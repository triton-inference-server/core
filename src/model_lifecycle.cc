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
    std::function<void(Status)> OnComplete)
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


  struct LoadTracker {
    LoadTracker(size_t affected_version_cnt)
        : load_failed_(false), completed_version_cnt_(0),
          affected_version_cnt_(affected_version_cnt)
    {
    }

    std::mutex mtx_;

    bool load_failed_;
    std::string reason_;
    size_t completed_version_cnt_;
    size_t affected_version_cnt_;
    std::map<int64_t, ModelInfo*> load_set_;
    // [WIP] the defer unload set will be deduced when all new versions are
    // successfully loaded, according to 'latest_update_ns_'

    // The set of model versions to be unloaded after the load is completed
    // std::set<int64_t> defer_unload_set_;
  };
  std::shared_ptr<LoadTracker> load_tracker(new LoadTracker(versions.size()));

  uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  // [WIP] below trigger load directly
  for (const auto& version : versions) {
    std::unique_ptr<ModelInfo> linfo(new ModelInfo(model_path, model_config));
    ModelInfo* model_info = linfo.get();

    auto res = it->second.emplace(
        std::make_pair(version, std::unique_ptr<ModelInfo>()));
    if (res.second) {
      res.first->second = std::move(linfo);
    } else {
      // There is already a record of this model version. Check if the version
      // model is being served, if so, the re-load of the version
      // should be performed in background to avoid version downtime
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
    if (OnComplete != nullptr) {
      model_info->OnComplete_ = [this, model_name, version, model_info,
                                 OnComplete, load_tracker]() {
        std::lock_guard<std::mutex> tracker_lock(load_tracker->mtx_);
        ++load_tracker->completed_version_cnt_;
        load_tracker->load_set_[version] = model_info;
        // [WIP] version will not be marked ready until all versions are
        // ready, this simplify the unloading when one version fails to load
        if (model_info->state_ != ModelReadyState::READY) {
          load_tracker->load_failed_ = true;
          load_tracker->reason_ +=
              ("version " + std::to_string(version) + ": " +
               model_info->state_reason_ + ";");
        }
        // Check if all versions are completed and finish the load
        if (load_tracker->completed_version_cnt_ ==
            load_tracker->affected_version_cnt_) {
          std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
          auto it = map_.find(model_name);
          if (load_tracker->load_failed_) {
            // If any of the versions fails to load, abort the load and unload
            // all newly loaded versions
            if (model_info->agent_model_list_) {
              auto status = model_info->agent_model_list_->InvokeAgentModels(
                  TRITONREPOAGENT_ACTION_LOAD_FAIL);
              if (!status.IsOk()) {
                LOG_ERROR << "Agent model returns error on "
                             "TRITONREPOAGENT_ACTION_LOAD_FAIL: "
                          << status.AsString();
              }
            }
            for (auto& loaded : load_tracker->load_set_) {
              std::lock_guard<std::recursive_mutex> lock(loaded.second->mtx_);
              if (loaded.second->state_ == ModelReadyState::READY) {
                auto vit = it->second.find(loaded.first);
                // Check if the version model is loaded in background, if so,
                // move the model to 'unloading_models_' and unload it.
                if (vit->second.second.get() == loaded.second) {
                  unloading_models_[(uintptr_t)loaded.second] =
                      std::move(vit->second.second);
                  auto unload_model = loaded.second;
                  // [FIXME] clean up can be moved to destructor
                  loaded.second->OnComplete_ = [this, unload_model]() {
                    std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
                    unloading_models_.erase((uintptr_t)unload_model);
                  };
                } else {
                  loaded.second->OnComplete_ = nullptr;
                }
                loaded.second->next_action_ = ActionType::UNLOAD;
                TriggerNextAction(model_name, loaded.first, loaded.second);
              }
            }
          } else {
            if (model_info->agent_model_list_) {
              auto status = model_info->agent_model_list_->InvokeAgentModels(
                  TRITONREPOAGENT_ACTION_LOAD_COMPLETE);
              if (!status.IsOk()) {
                LOG_ERROR << "Agent model returns error on "
                             "TRITONREPOAGENT_ACTION_LOAD_COMPLETE: "
                          << status.AsString();
              }
            }
            bool notified_agent = false;
            for (auto& loaded : load_tracker->load_set_) {
              auto vit = it->second.find(loaded.first);
              // Check if the version model is loaded in background, if so,
              // replace the current version model and unload it.
              if (vit->second.second.get() == loaded.second) {
                vit->second.second.swap(vit->second.first);
                auto unload_model = vit->second.second.get();
                unloading_models_[(uintptr_t)unload_model] =
                    std::move(vit->second.second);
                std::lock_guard<std::recursive_mutex> lock(unload_model->mtx_);
                unload_model->next_action_ = ActionType::UNLOAD;
                if (unload_model->agent_model_list_ && !notified_agent) {
                  auto unloading_agent_model_list =
                      unload_model->agent_model_list_;
                  auto status = unloading_agent_model_list->InvokeAgentModels(
                      TRITONREPOAGENT_ACTION_UNLOAD);
                  if (!status.IsOk()) {
                    LOG_ERROR << "Agent model returns error on "
                                 "TRITONREPOAGENT_ACTION_UNLOAD: "
                              << status.AsString();
                  }
                  unload_model->OnComplete_ = [this, unload_model,
                                               unloading_agent_model_list]() {
                    auto status = unloading_agent_model_list->InvokeAgentModels(
                        TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE);
                    if (!status.IsOk()) {
                      LOG_ERROR << "Agent model returns error on "
                                   "TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE: "
                                << status.AsString();
                    }
                    std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
                    unloading_models_.erase((uintptr_t)unload_model);
                  };
                  notified_agent = true;
                } else {
                  unload_model->OnComplete_ = [this, unload_model]() {
                    std::lock_guard<std::recursive_mutex> map_lock(map_mtx_);
                    unloading_models_.erase((uintptr_t)unload_model);
                  };
                }
                TriggerNextAction(model_name, version, unload_model);
              }
            }
            // Unload the deferred versions
            for (const auto deferred_version :
                 load_tracker->defer_unload_set_) {
              auto vit = it->second.find(deferred_version);
              auto unload_model = vit->second.first.get();
              std::lock_guard<std::recursive_mutex> lock(unload_model->mtx_);
              unload_model->next_action_ = ActionType::UNLOAD;
              if (unload_model->agent_model_list_ && !notified_agent) {
                auto unloading_agent_model_list =
                    unload_model->agent_model_list_;
                auto status = unloading_agent_model_list->InvokeAgentModels(
                    TRITONREPOAGENT_ACTION_UNLOAD);
                if (!status.IsOk()) {
                  LOG_ERROR << "Agent model returns error on "
                               "TRITONREPOAGENT_ACTION_UNLOAD: "
                            << status.AsString();
                }
                unload_model->OnComplete_ = [this,
                                             unloading_agent_model_list]() {
                  auto status = unloading_agent_model_list->InvokeAgentModels(
                      TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE);
                  if (!status.IsOk()) {
                    LOG_ERROR << "Agent model returns error on "
                                 "TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE: "
                              << status.AsString();
                  }
                };
                notified_agent = true;
              } else {
                unload_model->OnComplete_ = nullptr;
              }
              TriggerNextAction(model_name, deferred_version, unload_model);
            }
          }
          OnComplete(
              load_tracker->load_failed_
                  ? Status(Status::Code::INVALID_ARG, load_tracker->reason_)
                  : Status::Success);
        }
      };
    }
    Status action_status = TriggerNextAction(model_name, version, model_info);
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

}}  // namespace triton::core
