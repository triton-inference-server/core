// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "model_lifecycle.h"

#include <algorithm>
#include <deque>
#include <future>
#include <stdexcept>
#include <thread>
#include "constants.h"
#include "filesystem.h"
#include "model.h"
#include "model_config_utils.h"
#include "repo_agent.h"
#include "triton/common/logging.h"
#include "triton/common/thread_pool.h"

#include "backend_model.h"
#ifdef TRITON_ENABLE_ENSEMBLE
#include "ensemble_model.h"
#endif  // TRITON_ENABLE_ENSEMBLE

namespace triton { namespace core {

const std::string&
ModelReadyStateString(ModelReadyState state)
{
  switch (state) {
    case ModelReadyState::UNKNOWN: {
      static std::string m("UNKNOWN");
      return m;
    }
    case ModelReadyState::READY: {
      static std::string m("READY");
      return m;
    }
    case ModelReadyState::UNAVAILABLE: {
      static std::string m("UNAVAILABLE");
      return m;
    }
    case ModelReadyState::LOADING: {
      static std::string m("LOADING");
      return m;
    }
    case ModelReadyState::UNLOADING: {
      static std::string m("UNLOADING");
      return m;
    }
  }

  static std::string m("<unknown>");
  return m;
}

namespace {

Status
VersionsToLoad(
    const std::string model_path, const std::string& name,
    const inference::ModelConfig& model_config, std::set<int64_t>* versions)
{
  versions->clear();

  // Get integral number of the version directory
  std::set<std::string> subdirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &subdirs));
  std::set<int64_t, std::greater<int64_t>> existing_versions;
  for (const auto& subdir : subdirs) {
    if (subdir == kWarmupDataFolder || subdir == kInitialStateFolder) {
      continue;
    }
    if ((subdir.length() > 1) && (subdir.front() == '0')) {
      LOG_WARNING << "ignore version directory '" << subdir
                  << "' which contains leading zeros in its directory name";
      continue;
    }
    try {
      int64_t version = std::stoll(subdir);
      existing_versions.insert(version);
    }
    catch (const std::invalid_argument& ia) {
      LOG_WARNING << "ignore version directory '" << subdir
                  << "' which fails to convert to integral number";
    }
  }

  if (model_config.version_policy().has_specific()) {
    for (const auto& v : model_config.version_policy().specific().versions()) {
      // Only load the specific versions that are presented in model directory
      bool version_not_exist = existing_versions.insert(v).second;
      if (!version_not_exist) {
        versions->emplace(v);
      } else {
        LOG_ERROR << "version " << v << " is specified for model '" << name
                  << "', but the version directory is not present";
      }
    }
  } else {
    if (model_config.version_policy().has_latest()) {
      // std::set is sorted with std::greater
      for (const auto& v : existing_versions) {
        if (versions->size() >=
            model_config.version_policy().latest().num_versions()) {
          break;
        }
        versions->emplace(v);
      }
    } else {
      // all
      versions->insert(existing_versions.begin(), existing_versions.end());
    }
  }

  return Status::Success;
}

// Use smart pointer with custom deleter so that model state will be updated
// to UNAVAILABLE if all smart pointer copies are out of scope
struct ModelDeleter {
  ModelDeleter(std::function<void()> OnDestroyModel)
      : OnDestroyModel_(std::move(OnDestroyModel))
  {
  }

  void operator()(Model* model)
  {
    // The actual model object must be destroyed in a different
    // thread. This thread could have a callstack that includes the
    // model itself because this deleter could be triggered by
    // a request release or response send in the model. Following
    // delete will lead to the model destructor which may wait on this
    // same thread... so deadlock if we don't use a different thread
    // here.
    std::function<void()> destroy_fn = OnDestroyModel_;
    std::thread dthd([model, destroy_fn]() {
      delete model;
      destroy_fn();
    });

    dthd.detach();
  }

  // Use to inform the ModelLifeCycle that the model handle is destroyed
  std::function<void()> OnDestroyModel_;
};

}  // namespace

Status
ModelLifeCycle::Create(
    InferenceServer* server, const double min_compute_capability,
    const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
    const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
    const unsigned int model_load_thread_count,
    std::unique_ptr<ModelLifeCycle>* life_cycle)
{
  std::unique_ptr<ModelLifeCycle> local_life_cycle(new ModelLifeCycle(
      min_compute_capability, server, backend_cmdline_config_map,
      host_policy_map, model_load_thread_count));

  *life_cycle = std::move(local_life_cycle);
  return Status::Success;
}

const ModelStateMap
ModelLifeCycle::LiveModelStates(bool strict_readiness)
{
  LOG_VERBOSE(2) << "LiveModelStates()";
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  ModelStateMap live_model_states;
  for (auto& model_version : map_) {
    bool live = false;
    VersionStateMap version_map;

    for (auto& version_model : model_version.second) {
      std::lock_guard<std::recursive_mutex> lock(
          version_model.second.first->mtx_);
      if (strict_readiness &&
          version_model.second.first->state_ != ModelReadyState::READY) {
        continue;
      }

      // At lease one version is live (ready / loading / unloading)
      if ((version_model.second.first->state_ != ModelReadyState::UNKNOWN) &&
          (version_model.second.first->state_ !=
           ModelReadyState::UNAVAILABLE)) {
        live = true;
        version_map[version_model.first] = std::make_pair(
            version_model.second.first->state_,
            version_model.second.first->state_reason_);
      }
    }

    if (live) {
      live_model_states[model_version.first] = std::move(version_map);
    }
  }
  return live_model_states;
}

Status
ModelLifeCycle::StopAllModels()
{
  LOG_VERBOSE(2) << "StopAllModels()";
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  for (auto& model_version : map_) {
    for (auto& version_model : model_version.second) {
      if (version_model.second.first != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(
            version_model.second.first->mtx_);
        if (version_model.second.first->model_ != nullptr) {
          version_model.second.first->model_->Stop();
        }
      }
    }
  }
  return Status::Success;
}

const std::set<std::tuple<std::string, int64_t, size_t>>
ModelLifeCycle::InflightStatus()
{
  LOG_VERBOSE(2) << "InflightStatus()";
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  std::set<std::tuple<std::string, int64_t, size_t>> inflight_status;
  for (auto& model_version : map_) {
    for (auto& version_model : model_version.second) {
      if (version_model.second.first != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(
            version_model.second.first->mtx_);
        if (version_model.second.first->model_ != nullptr) {
          const auto cnt =
              version_model.second.first->model_->InflightInferenceCount();
          if (cnt != 0) {
            inflight_status.emplace(
                model_version.first, version_model.first, cnt);
          }
        }
      }
    }
  }
  return inflight_status;
}

const ModelStateMap
ModelLifeCycle::ModelStates()
{
  LOG_VERBOSE(2) << "ModelStates()";
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  ModelStateMap model_states;
  for (auto& model_version : map_) {
    VersionStateMap version_map;

    for (auto& version_model : model_version.second) {
      std::lock_guard<std::recursive_mutex> lock(
          version_model.second.first->mtx_);
      version_map[version_model.first] = std::make_pair(
          version_model.second.first->state_,
          version_model.second.first->state_reason_);
    }

    model_states[model_version.first] = std::move(version_map);
  }

  return model_states;
}

const VersionStateMap
ModelLifeCycle::VersionStates(const std::string& model_name)
{
  LOG_VERBOSE(2) << "VersionStates() '" << model_name << "'";
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  VersionStateMap version_map;
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    for (auto& version_model : mit->second) {
      std::lock_guard<std::recursive_mutex> lock(
          version_model.second.first->mtx_);
      version_map[version_model.first] = std::make_pair(
          version_model.second.first->state_,
          version_model.second.first->state_reason_);
    }
  }

  return version_map;
}

Status
ModelLifeCycle::ModelState(
    const std::string& model_name, const int64_t model_version,
    ModelReadyState* state)
{
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    auto vit = mit->second.find(model_version);
    if (vit != mit->second.end()) {
      std::lock_guard<std::recursive_mutex> lock(vit->second.first->mtx_);
      *state = vit->second.first->state_;
      return Status::Success;
    }
  }

  return Status(
      Status::Code::NOT_FOUND, "model '" + model_name + "', version " +
                                   std::to_string(model_version) +
                                   " is not found");
}

Status
ModelLifeCycle::GetModel(
    const std::string& model_name, const int64_t version,
    std::shared_ptr<Model>* model)
{
  do {
    LOG_VERBOSE(2) << "GetModel() '" << model_name << "' version " << version;
    std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
    auto mit = map_.find(model_name);
    if (mit == map_.end()) {
      return Status(
          Status::Code::NOT_FOUND, "'" + model_name + "' is not found");
    }

    auto vit = mit->second.find(version);
    if (vit == mit->second.end()) {
      // In case the request is asking for latest version
      int64_t latest = -1;
      // Whether or not a re-attempt should be made to acquire stable model.
      bool retry = false;
      if (version == -1) {
        for (auto& version_model : mit->second) {
          if (version_model.first > latest) {
            // The LoadUnload thread may have already acquired version model
            // mutex to update its state. Here we attempt to acquire the lock
            // but don't block on it.
            std::unique_lock<std::recursive_mutex> lock(
                version_model.second.first->mtx_, std::try_to_lock);
            retry = !lock.owns_lock();
            if (retry) {
              // Skip this version as it is still being processed.
              continue;
            }
            if (version_model.second.first->state_ == ModelReadyState::READY) {
              latest = version_model.first;
              // Tedious, but have to set handle for any "latest" version
              // at the moment to avoid edge case like the following:
              // "versions : 1 3 2", version 3 is latest but is requested
              // to be unloaded when the iterator is examining version 2.
              *model = version_model.second.first->model_;
            }
          }
        }
        if (retry) {
          // Failed to find a stable model version to use. Will initiate a
          // re-attempt, which will release map_mtx_ for now. This is important
          // so that the LoadUnload thread which is processing the model version
          // can make progress and avoid a deadlock.
          continue;
        }
        if (latest == -1) {
          return Status(
              Status::Code::NOT_FOUND,
              "'" + model_name + "' has no available versions");
        }
      } else {
        return Status(
            Status::Code::NOT_FOUND, "'" + model_name + "' version " +
                                         std::to_string(version) +
                                         " is not found");
      }
    } else {
      // The LoadUnload thread may have already acquired version model
      // mutex to update its state. Here we attempt to acquire the lock
      // but don't block on it.
      std::unique_lock<std::recursive_mutex> lock(
          vit->second.first->mtx_, std::try_to_lock);
      if (!lock.owns_lock()) {
        // The model version is still being processed. Will initiate a
        // re-attempt, which will release map_mtx_ for now. This is important so
        // that the LoadUnload thread which is processing the model version can
        // make progress and avoid a deadlock.
        continue;
      }
      if (vit->second.first->state_ == ModelReadyState::READY) {
        *model = vit->second.first->model_;
      } else {
        return Status(
            Status::Code::UNAVAILABLE, "'" + model_name + "' version " +
                                           std::to_string(version) +
                                           " is not at ready state");
      }
    }
    return Status::Success;
  } while (true);
}

Status
ModelLifeCycle::AsyncUnload(const std::string& model_name)
{
  LOG_VERBOSE(2) << "AsyncUnload() '" << model_name << "'";
  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  auto it = map_.find(model_name);
  if (it == map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "Model to be unloaded has not been served");
  }

  // Get the existing agent models and notify the unload action
  for (auto& version : it->second) {
    ModelInfo* model_info = version.second.first.get();
    if (model_info->agent_model_list_ != nullptr) {
      auto unloading_agent_model_list = model_info->agent_model_list_;
      // Only log the error because the model should be unloaded regardless
      auto status = unloading_agent_model_list->InvokeAgentModels(
          TRITONREPOAGENT_ACTION_UNLOAD);
      if (!status.IsOk()) {
        LOG_ERROR
            << "Agent model returns error on TRITONREPOAGENT_ACTION_UNLOAD: "
            << status.AsString();
      }
      // [FIXME] move to deletor?
      model_info->OnComplete_ = [this, unloading_agent_model_list]() {
        auto status = unloading_agent_model_list->InvokeAgentModels(
            TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE);
        if (!status.IsOk()) {
          LOG_ERROR << "Agent model returns error on "
                       "TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE: "
                    << status.AsString();
        }
      };
      break;
    }
  }

  Status status = Status::Success;
  for (auto& version_model : it->second) {
    ModelInfo* model_info = version_model.second.first.get();
    model_info->next_action_ = ActionType::UNLOAD;
    Status action_status =
        TriggerNextAction(model_name, version_model.first, model_info);
    if (!action_status.IsOk()) {
      status = action_status;
    }
  }

  return status;
}

Status
ModelLifeCycle::AsyncLoad(
    const std::string& model_name, const std::string& model_path,
    const inference::ModelConfig& model_config,
    const std::shared_ptr<TritonRepoAgentModelList>& agent_model_list,
    std::function<void(Status)>&& OnComplete)
{
  LOG_VERBOSE(2) << "AsyncLoad() '" << model_name << "'";

  std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
  auto it = map_.find(model_name);
  if (it == map_.end()) {
    it = map_.emplace(std::make_pair(model_name, VersionMap())).first;
  }

  std::set<int64_t> versions;
  RETURN_IF_ERROR(
      VersionsToLoad(model_path, model_name, model_config, &versions));
  if (versions.empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "at least one version must be available under the version policy of "
        "model '" +
            model_name + "'");
  }


  const uint64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  std::shared_ptr<LoadTracker> load_tracker(
      new LoadTracker(versions.size(), now_ns));
  // [WIP] below trigger load directly
  for (const auto& version : versions) {
    std::unique_ptr<ModelInfo> linfo(
        new ModelInfo(model_path, model_config, now_ns));
    ModelInfo* model_info = linfo.get();

    auto res = it->second.emplace(
        std::make_pair(version, std::unique_ptr<ModelInfo>()));
    if (res.second) {
      res.first->second = std::move(linfo);
    } else {
      // There is already a record of this model version. Check if the version
      // model is being served, if so, the re-load of the version
      // should be performed in background to avoid version downtime.
      // Otherwise, swap and monitor state for newly loading model.
      auto& serving_model = res.first->second;
      std::lock_guard<std::recursive_mutex> lock(serving_model->mtx_);
      if (serving_model->state_ == ModelReadyState::READY) {
        background_models_[(uintptr_t)model_info] = std::move(linfo);
      } else {
        // swap the monitoring model info
        serving_model.swap(linfo);

        // further check the state, put to 'background_models_' to keep
        // the object valid if the model is LOADING / UNLOADING, because
        // the model info may be accessed by a different thread once the
        // operation is completed
        if ((linfo->state_ == ModelReadyState::LOADING) ||
            (linfo->state_ == ModelReadyState::UNLOADING)) {
          ModelInfo* key = linfo.get();
          background_models_[(uintptr_t)key] = std::move(linfo);
        }
      }
    }

    // [WIP] Trigger load and stuff
    model_info->agent_model_list_ = agent_model_list;
    model_info->latest_update_ns_ = now_ns;
    // set version-wise callback before triggering next action
    // [FIXME] 'OnComplete_' is used as handy callback placeholder for now,
    // should revisit
    model_info->OnComplete_ = std::bind(
        &ModelLifeCycle::OnLoadComplete, this, model_name, version, model_info,
        OnComplete, load_tracker);
    Status action_status = NewLoad(model_name, version, model_info);
    if (!action_status.IsOk()) {
      status = action_status;
    }
  }

  return status;
}

Status
ModelLifeCycle::TriggerNextAction(
    const std::string& model_name, const int64_t version, ModelInfo* model_info)
{
  LOG_VERBOSE(2) << "TriggerNextAction() '" << model_name << "' version "
                 << version << ": " << std::to_string(model_info->next_action_);
  ActionType next_action = model_info->next_action_;
  model_info->next_action_ = ActionType::NO_ACTION;
  Status status = Status::Success;
  switch (next_action) {
    case ActionType::LOAD:
      status = Load(model_name, version, model_info);
      break;
    case ActionType::UNLOAD:
      status = Unload(model_name, version, model_info);
      break;
    default:
      if (model_info->OnComplete_ != nullptr) {
        LOG_VERBOSE(2) << "no next action, trigger OnComplete()";
        model_info->OnComplete_();
        model_info->OnComplete_ = nullptr;
      }
      break;
  }

  // If status is not ok, "next action" path ends here and thus need to
  // invoke callback by this point
  if ((!status.IsOk()) && (model_info->OnComplete_ != nullptr)) {
    LOG_VERBOSE(2) << "failed to execute next action, trigger OnComplete()";
    // [FIXME] OnComplete_ shouldn't be called inplace, should be moved first
    model_info->OnComplete_();
    model_info->OnComplete_ = nullptr;
  }

  return status;
}

Status
ModelLifeCycle::Load(
    const std::string& model_name, const int64_t version, ModelInfo* model_info)
{
  LOG_VERBOSE(2) << "Load() '" << model_name << "' version " << version;
  Status status = Status::Success;

  model_info->next_action_ = ActionType::NO_ACTION;

  switch (model_info->state_) {
    case ModelReadyState::READY:
      LOG_INFO << "re-loading: " << model_name << ":" << version;
      model_info->state_ = ModelReadyState::UNLOADING;
      model_info->state_reason_.clear();
      model_info->next_action_ = ActionType::LOAD;
      // The load will be triggered once the unload is done (deleter is called)
      model_info->model_.reset();
      break;
    case ModelReadyState::LOADING:
    case ModelReadyState::UNLOADING:
      model_info->next_action_ = ActionType::LOAD;
      break;
    default:
      LOG_INFO << "loading: " << model_name << ":" << version;
      model_info->state_ = ModelReadyState::LOADING;
      model_info->state_reason_.clear();
      // Load model asynchronously via thread pool
      load_pool_->enqueue([this, model_name, version, model_info]() {
        CreateModel(model_name, version, model_info);
      });
      break;
  }

  return status;
}

Status
ModelLifeCycle::NewLoad(
    const std::string& model_name, const int64_t version, ModelInfo* model_info)
{
  LOG_VERBOSE(2) << "Load() '" << model_name << "' version " << version;

  LOG_INFO << "loading: " << model_name << ":" << version;
  model_info->state_ = ModelReadyState::LOADING;
  model_info->state_reason_.clear();
  // Load model asynchronously via thread pool
  load_pool_->enqueue([this, model_name, version, model_info]() {
    CreateModel(model_name, version, model_info);
  });

  return Status::Success;
}

Status
ModelLifeCycle::Unload(
    const std::string& model_name, const int64_t version, ModelInfo* model_info)
{
  LOG_VERBOSE(2) << "Unload() '" << model_name << "' version " << version;
  Status status = Status::Success;

  model_info->next_action_ = ActionType::NO_ACTION;

  switch (model_info->state_) {
    case ModelReadyState::READY:
      LOG_INFO << "unloading: " << model_name << ":" << version;
      model_info->state_ = ModelReadyState::UNLOADING;
      model_info->state_reason_.clear();
      model_info->model_.reset();
      model_info->agent_model_list_.reset();
      break;
    case ModelReadyState::LOADING:
    case ModelReadyState::UNLOADING:
      model_info->next_action_ = ActionType::UNLOAD;
      break;
    default:
      status = Status(
          Status::Code::NOT_FOUND,
          "tried to unload model '" + model_name + "' version " +
              std::to_string(version) + " which is at model state: " +
              ModelReadyStateString(model_info->state_));
      break;
  }

  return status;
}

Status
ModelLifeCycle::CreateModel(
    const std::string& model_name, const int64_t version, ModelInfo* model_info)
{
  LOG_VERBOSE(2) << "CreateModel() '" << model_name << "' version " << version;
  // make copy of the current model config in case model config in model info
  // is updated (another poll) during the creation of the model
  inference::ModelConfig model_config;
  {
    std::lock_guard<std::recursive_mutex> lock(model_info->mtx_);
    model_config = model_info->model_config_;
  }

  // Create model
  Status status;
  std::unique_ptr<Model> is;

  // If 'backend' is specified in the config then use the new triton
  // backend.
  if (!model_config.backend().empty()) {
    std::unique_ptr<TritonModel> model;
    status = TritonModel::Create(
        server_, model_info->model_path_, cmdline_config_map_, host_policy_map_,
        model_name, version, model_config, &model);
    is.reset(model.release());
  } else {
#ifdef TRITON_ENABLE_ENSEMBLE
    if (model_info->is_ensemble_) {
      status = EnsembleModel::Create(
          server_, model_info->model_path_, version, model_config,
          min_compute_capability_, &is);
      // Complete label provider with label information from involved models
      // Must be done here because involved models may not be able to
      // obtained from server because this may happen during server
      // initialization.
      if (status.IsOk()) {
        std::set<std::string> no_label_outputs;
        const auto& label_provider = is->GetLabelProvider();
        for (const auto& output : model_config.output()) {
          if (label_provider->GetLabel(output.name(), 0).empty()) {
            no_label_outputs.emplace(output.name());
          }
        }
        for (const auto& element : model_config.ensemble_scheduling().step()) {
          for (const auto& pair : element.output_map()) {
            // Found model that produce one of the missing output
            if (no_label_outputs.find(pair.second) != no_label_outputs.end()) {
              std::shared_ptr<Model> model;
              // Safe to obtain model because the ensemble can't be loaded
              // until the involved models are ready
              GetModel(element.model_name(), element.model_version(), &model);
              label_provider->AddLabels(
                  pair.second,
                  model->GetLabelProvider()->GetLabels(pair.first));
            }
          }
        }
      }
    } else
#endif  // TRITON_ENABLE_ENSEMBLE
    {
      status = Status(
          Status::Code::INVALID_ARG,
          "unknown platform '" + model_config.platform() + "'");
    }
  }

  // Update model state
  std::lock_guard<std::recursive_mutex> lock(model_info->mtx_);
  // Sanity check
  if (model_info->model_ != nullptr) {
    LOG_ERROR << "trying to load model '" << model_name << "' version "
              << version << " while it is being served";
  } else {
    if (status.IsOk()) {
      // Unless the handle is nullptr, always reset handle out of the mutex,
      // otherwise the handle's destructor will try to acquire the mutex and
      // cause deadlock.
      model_info->model_.reset(
          is.release(),
          ModelDeleter([this, model_name, version, model_info]() mutable {
            LOG_VERBOSE(2) << "OnDestroy callback() '" << model_name
                           << "' version " << version;
            LOG_INFO << "successfully unloaded '" << model_name << "' version "
                     << version;
            // Use recursive mutex as this deleter is likely to to be called
            // within ModelLifeCycle class where the same mutex is being hold.
            // However, mutex acquisition is needed here for the case where
            // the model is requested to be unloaded while there are inflight
            // requests, then the deleter will be called from the request thread
            {
              std::lock_guard<std::recursive_mutex> lock(model_info->mtx_);
              model_info->state_ = ModelReadyState::UNAVAILABLE;
              model_info->state_reason_ = "unloaded";
              // Check if next action is requested
              this->TriggerNextAction(model_name, version, model_info);
            }
          }));
      model_info->state_ = ModelReadyState::READY;
      model_info->state_reason_.clear();
      LOG_INFO << "successfully loaded '" << model_name << "' version "
               << version;
    } else {
      LOG_ERROR << "failed to load '" << model_name << "' version " << version
                << ": " << status.AsString();
      model_info->state_ = ModelReadyState::UNAVAILABLE;
      model_info->state_reason_ = status.AsString();
    }
  }

  // Check if next action is requested
  return TriggerNextAction(model_name, version, model_info);
}

void
ModelLifeCycle::NewCreateModel(
    const std::string& model_name, const int64_t version, ModelInfo* model_info)
{
  LOG_VERBOSE(2) << "CreateModel() '" << model_name << "' version " << version;
  const auto& model_config = model_info->model_config_;

  // Create model
  Status status;
  std::unique_ptr<Model> is;

  // If 'backend' is specified in the config then use the new triton
  // backend.
  if (!model_config.backend().empty()) {
    std::unique_ptr<TritonModel> model;
    status = TritonModel::Create(
        server_, model_info->model_path_, cmdline_config_map_, host_policy_map_,
        model_name, version, model_config, &model);
    is.reset(model.release());
  } else {
#ifdef TRITON_ENABLE_ENSEMBLE
    if (model_info->is_ensemble_) {
      status = EnsembleModel::Create(
          server_, model_info->model_path_, version, model_config,
          min_compute_capability_, &is);
      // Complete label provider with label information from involved models
      // Must be done here because involved models may not be able to
      // obtained from server because this may happen during server
      // initialization.
      if (status.IsOk()) {
        std::set<std::string> no_label_outputs;
        const auto& label_provider = is->GetLabelProvider();
        for (const auto& output : model_config.output()) {
          if (label_provider->GetLabel(output.name(), 0).empty()) {
            no_label_outputs.emplace(output.name());
          }
        }
        for (const auto& element : model_config.ensemble_scheduling().step()) {
          for (const auto& pair : element.output_map()) {
            // Found model that produce one of the missing output
            if (no_label_outputs.find(pair.second) != no_label_outputs.end()) {
              std::shared_ptr<Model> model;
              // Safe to obtain model because the ensemble can't be loaded
              // until the involved models are ready
              GetModel(element.model_name(), element.model_version(), &model);
              label_provider->AddLabels(
                  pair.second,
                  model->GetLabelProvider()->GetLabels(pair.first));
            }
          }
        }
      }
    } else
#endif  // TRITON_ENABLE_ENSEMBLE
    {
      status = Status(
          Status::Code::INVALID_ARG,
          "unknown platform '" + model_config.platform() + "'");
    }
  }

  // [FIXME] lock?
  {
    std::lock_guard<std::recursive_mutex> lock(model_info->mtx_);
    // Update model state
    // [FIXME] check if current state is still LOADING?
    // helper function to determine if the load should be
    // aborted (has newer state change / fail to load / self?)
    if (status.IsOk()) {
      // [FIXME] better way to manage agent model lifecycle
      auto agent_model_list = model_info->agent_model_list_;
      // Unless the handle is nullptr, always reset handle out of the mutex,
      // otherwise the handle's destructor will try to acquire the mutex and
      // cause deadlock.
      model_info->model_.reset(
          is.release(), ModelDeleter([this, model_name, version, model_info,
                                      agent_model_list]() mutable {
            LOG_VERBOSE(2) << "OnDestroy callback() '" << model_name
                           << "' version " << version;
            LOG_INFO << "successfully unloaded '" << model_name << "' version "
                     << version;
            // [FIXME] update deleter callback below
            // Use recursive mutex as this deleter is likely to to be called
            // within ModelLifeCycle class where the same mutex is being hold.
            // However, mutex acquisition is needed here for the case where
            // the model is requested to be unloaded while there are inflight
            // requests, then the deleter will be called from the request thread
            {
              std::lock_guard<std::recursive_mutex> lock(model_info->mtx_);
              model_info->state_ = ModelReadyState::UNAVAILABLE;
              model_info->state_reason_ = "unloaded";
              // Check if next action is requested
              this->TriggerNextAction(model_name, version, model_info);
            }
          }));
      // [WIP] defer the setting to READY state to OnLoadComplete callback
      // for simplicity (avoid serving subset of the versions while the load
      // is not fully finished)
      // model_info->state_ = ModelReadyState::READY;
      // model_info->state_reason_.clear();
      // LOG_INFO << "successfully loaded '" << model_name << "' version "
      //           << version;
    } else {
      LOG_ERROR << "failed to load '" << model_name << "' version " << version
                << ": " << status.AsString();
      model_info->state_ = ModelReadyState::UNAVAILABLE;
      model_info->state_reason_ = status.AsString();
    }
  }

  auto load_complete_fn = std::move(model_info->OnComplete_);
  load_complete_fn();
}

void
ModelLifeCycle::OnLoadComplete(
    const std::string& model_name, const int64_t version, ModelInfo* model_info,
    std::function<void(Status)> OnComplete,
    std::shared_ptr<LoadTracker> load_tracker)
{
  std::lock_guard<std::mutex> tracker_lock(load_tracker->mtx_);
  ++load_tracker->completed_version_cnt_;
  load_tracker->load_set_[version] = model_info;
  // [WIP] version will not be marked ready until all versions are
  // ready, this simplify the unloading when one version fails to load
  if (model_info->state_ != ModelReadyState::LOADING) {
    load_tracker->load_failed_ = true;
    load_tracker->reason_ +=
        ("version " + std::to_string(version) + " is at " +
         ModelReadyStateString(model_info->state_) +
         " state : " + model_info->state_reason_ + ";");
  }
  // Check if all versions are completed and finish the load
  if (load_tracker->completed_version_cnt_ ==
      load_tracker->affected_version_cnt_) {
    // hold 'map_mtx_' as there will be change onto the model info map
    std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
    auto it = map_.find(model_name);
    // Check if the load is the latest frontground action on the model
    for (const auto& version_info : it->second) {
      if (version_info.second->latest_update_ns_ >
          load_tracker->latest_update_ns_) {
        load_tracker->load_failed_ = true;
        load_tracker->reason_ =
            "Newer operation has been applied to the model lifecycle, current "
            "load operation is out-dated.";
      }
    }
    // [FIXME] load should fail when newer update on the model has taken affect
    if (load_tracker->load_failed_) {
      // Move agent list out of ModelInfo as it needs to be invoked
      // after all ModelInfos are reset
      std::shared_ptr<TritonRepoAgentModelList> lagent_list;
      if (model_info->agent_model_list_) {
        lagent_list = std::move(model_info->agent_model_list_);
      }
      // If any of the versions fails to load, abort the load and unload
      // all newly loaded versions
      for (auto& loaded : load_tracker->load_set_) {
        // Unload directly, the object is being managed either in frontground
        // or background
        std::lock_guard<std::recursive_mutex> lock(loaded.second->mtx_);
        if (loaded.second->model_ != nullptr) {
          model_info->state_ = ModelReadyState::UNLOADING;
          model_info->state_reason_.clear();
          model_info->agent_model_list_.reset();
          model_info->model_.reset();
        }
      }

      if (lagent_list) {
        auto status =
            lagent_list->InvokeAgentModels(TRITONREPOAGENT_ACTION_LOAD_FAIL);
        if (!status.IsOk()) {
          LOG_ERROR << "Agent model returns error on "
                       "TRITONREPOAGENT_ACTION_LOAD_FAIL: "
                    << status.AsString();
        }
      }
    } else {
      // Unload any previous loaded versions that are still available
      for (auto& version_info : it->second) {
        auto& mi = version_info.second;
        if ((mi->state_ == ModelReadyState::READY) &&
            (mi->latest_update_ns_ < load_tracker->latest_update_ns_)) {
          if ((mi->agent_model_list_ != nullptr) &&
              (mi->agent_model_list_->LastActionType() ==
               TRITONREPOAGENT_ACTION_LOAD_COMPLETE)) {
            auto status = mi->agent_model_list_->InvokeAgentModels(
                TRITONREPOAGENT_ACTION_UNLOAD);
            if (!status.IsOk()) {
              LOG_ERROR << "Agent model returns error on "
                           "TRITONREPOAGENT_ACTION_UNLOAD: "
                        << status.AsString();
            }
          }

          mi->state_ = ModelReadyState::UNLOADING;
          mi->state_reason_.clear();
          mi->agent_model_list_.reset();
          mi->model_.reset();
        }
      }

      // Mark current versions ready and track info in frontground
      for (auto& loaded : load_tracker->load_set_) {
        loaded.second->state_ = ModelReadyState::READY;

        auto bit = background_models_.find((uintptr_t)loaded.second);
        // Check if the version model is loaded in background, if so,
        // replace and unload the current serving version
        if (bit != background_models_.end()) {
          auto vit = it->second.find(loaded.first);

          // Need to lock the previous model info for in case the model is
          // loading / unloading, this ensure the model state tracked even when
          // the load / unload is completed.
          std::lock_guard<std::recursive_mutex> prev_info_lk(vit->second->mtx_);

          // swap previous info into local unique pointer
          auto linfo = std::move(bit->second);
          vit->second.swap(linfo);
          background_models_.erase(bit);

          // if previous info is under change, put into 'background_models_'
          if ((linfo->state_ == ModelReadyState::LOADING) ||
              (linfo->state_ == ModelReadyState::UNLOADING)) {
            ModelInfo* key = linfo.get();
            background_models_[(uintptr_t)key] = std::move(linfo);
          }
        }
      }
      if (model_info->agent_model_list_) {
        auto status = model_info->agent_model_list_->InvokeAgentModels(
            TRITONREPOAGENT_ACTION_LOAD_COMPLETE);
        if (!status.IsOk()) {
          LOG_ERROR << "Agent model returns error on "
                       "TRITONREPOAGENT_ACTION_LOAD_COMPLETE: "
                    << status.AsString();
        }
      }
    }
    if (OnComplete != nullptr) {
      OnComplete(
          load_tracker->load_failed_
              ? Status(Status::Code::INVALID_ARG, load_tracker->reason_)
              : Status::Success);
    }
  }
}

}}  // namespace triton::core
