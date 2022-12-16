// Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_repository_manager.h"

#include <algorithm>
#include <deque>
#include <future>
#include <stdexcept>
#include <thread>
#include "constants.h"
#include "ensemble_utils.h"
#include "filesystem.h"
#include "model.h"
#include "model_config_utils.h"
#include "triton/common/logging.h"

#include "backend_model.h"
#ifdef TRITON_ENABLE_ENSEMBLE
#include "ensemble_model.h"
#endif  // TRITON_ENABLE_ENSEMBLE

namespace triton { namespace core {

namespace {

static std::string file_prefix = "file:";

// Internal repo agent used for model file override
class LocalizeRepoAgent : public TritonRepoAgent {
 public:
  LocalizeRepoAgent()
      : TritonRepoAgent("ModelRepositoryManager::LocalizeRepoAgent")
  {
    // Callbacks below interact with TritonRepoAgentModel directly knowing that
    // it is the internal implementation of TRITONREPOAGENT_AgentModel
    model_action_fn_ = [](TRITONREPOAGENT_Agent* agent,
                          TRITONREPOAGENT_AgentModel* model,
                          const TRITONREPOAGENT_ActionType action_type)
        -> TRITONSERVER_Error* {
      auto agent_model = reinterpret_cast<TritonRepoAgentModel*>(model);
      switch (action_type) {
        case TRITONREPOAGENT_ACTION_LOAD: {
          // localize the override files for model loading,
          // as currently the model is expected to load from local directory
          const char* temp_dir_cstr = nullptr;
          RETURN_TRITONSERVER_ERROR_IF_ERROR(
              agent_model->AcquireMutableLocation(
                  TRITONREPOAGENT_ARTIFACT_FILESYSTEM, &temp_dir_cstr));
          const std::string temp_dir = temp_dir_cstr;
          const auto& files =
              *reinterpret_cast<std::vector<const InferenceParameter*>*>(
                  agent_model->State());
          bool found_config = false;
          for (const auto& file : files) {
            if (file->Name() == "config") {
              if (file->Type() != TRITONSERVER_PARAMETER_STRING) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "Config parameter 'config' must have string type for its "
                    "value");
              }
              inference::ModelConfig config;
              RETURN_TRITONSERVER_ERROR_IF_ERROR(JsonToModelConfig(
                  file->ValueString(), 1 /* config_version */, &config));
              RETURN_TRITONSERVER_ERROR_IF_ERROR(WriteTextProto(
                  JoinPath({temp_dir, kModelConfigPbTxt}), config));
              found_config = true;
            } else if (file->Name().rfind(file_prefix, 0) == 0) {
              if (file->Type() != TRITONSERVER_PARAMETER_BYTES) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string("File parameter '") + file->Name() +
                     "' must have bytes type for its value")
                        .c_str());
              }

              // Save model file to the instructed directory
              // mkdir
              const std::string file_path =
                  JoinPath({temp_dir, file->Name().substr(file_prefix.size())});
              const std::string dir = DirName(file_path);
              bool dir_exist = false;
              RETURN_TRITONSERVER_ERROR_IF_ERROR(FileExists(dir, &dir_exist));
              if (dir_exist) {
                bool is_dir = false;
                RETURN_TRITONSERVER_ERROR_IF_ERROR(IsDirectory(dir, &is_dir));
                if (!is_dir) {
                  return TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INVALID_ARG,
                      (std::string("Invalid file parameter '") + file->Name() +
                       "', directory has been created as a file")
                          .c_str());
                }
              } else {
                RETURN_TRITONSERVER_ERROR_IF_ERROR(
                    MakeDirectory(dir, true /* recursive */));
              }

              // write
              RETURN_TRITONSERVER_ERROR_IF_ERROR(WriteBinaryFile(
                  file_path,
                  reinterpret_cast<const char*>(file->ValuePointer()),
                  file->ValueByteSize()));
            }
          }
          if (!found_config) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "Load parameter 'config' must be specified for model file "
                "override");
          }
          // Commit the temporary directory
          RETURN_TRITONSERVER_ERROR_IF_ERROR(agent_model->SetLocation(
              TRITONREPOAGENT_ARTIFACT_FILESYSTEM, temp_dir_cstr));
          break;
        }
        default:
          break;
      }
      return nullptr;  // success
    };

    model_fini_fn_ =
        [](TRITONREPOAGENT_Agent* agent,
           TRITONREPOAGENT_AgentModel* model) -> TRITONSERVER_Error* {
      auto agent_model = reinterpret_cast<TritonRepoAgentModel*>(model);
      RETURN_TRITONSERVER_ERROR_IF_ERROR(agent_model->DeleteMutableLocation());
      return nullptr;  // success
    };
  }
};

Status
CreateAgentModelListWithLoadAction(
    const inference::ModelConfig& original_model_config,
    const std::string& original_model_path,
    std::shared_ptr<TritonRepoAgentModelList>* agent_model_list)
{
  if (original_model_config.has_model_repository_agents()) {
    // Trick to append user specified repo agent on top of internal ones
    std::shared_ptr<TritonRepoAgentModelList> lagent_model_list;
    if (*agent_model_list != nullptr) {
      lagent_model_list = std::move(*agent_model_list);
    } else {
      lagent_model_list.reset(new TritonRepoAgentModelList());
    }

    FileSystemType filesystem_type;
    RETURN_IF_ERROR(GetFileSystemType(original_model_path, &filesystem_type));
    TRITONREPOAGENT_ArtifactType artifact_type =
        TRITONREPOAGENT_ARTIFACT_FILESYSTEM;
    if (filesystem_type != FileSystemType::LOCAL) {
      artifact_type = TRITONREPOAGENT_ARTIFACT_REMOTE_FILESYSTEM;
    }
    const char* location = original_model_path.c_str();
    inference::ModelConfig model_config = original_model_config;
    for (const auto& agent_config :
         original_model_config.model_repository_agents().agents()) {
      std::shared_ptr<TritonRepoAgent> agent;
      RETURN_IF_ERROR(
          TritonRepoAgentManager::CreateAgent(agent_config.name(), &agent));
      TritonRepoAgent::Parameters agent_params;
      for (const auto& parameter : agent_config.parameters()) {
        agent_params.emplace_back(parameter.first, parameter.second);
      }
      std::unique_ptr<TritonRepoAgentModel> agent_model;
      if (lagent_model_list->Size() != 0) {
        lagent_model_list->Back()->Location(&artifact_type, &location);
        const auto config_path = JoinPath({location, kModelConfigPbTxt});
        if (!ReadTextProto(config_path, &model_config).IsOk()) {
          model_config.Clear();
        }
      }
      RETURN_IF_ERROR(TritonRepoAgentModel::Create(
          artifact_type, location, model_config, agent, agent_params,
          &agent_model));
      RETURN_IF_ERROR(agent_model->InvokeAgent(TRITONREPOAGENT_ACTION_LOAD));
      lagent_model_list->AddAgentModel(std::move(agent_model));
    }
    *agent_model_list = std::move(lagent_model_list);
  }
  return Status::Success;
}

int64_t
GetModifiedTime(const std::string& path)
{
  // If there is an error in any step the fall-back default
  // modification time is 0. This means that in error cases 'path'
  // will show as not modified. This is the safe fall-back to avoid
  // assuming a model is constantly being modified.
  bool path_is_dir;
  Status status = IsDirectory(path, &path_is_dir);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status.AsString();
    return 0;
  }

  // If 'path' is a file return its mtime. Otherwise, using the modification
  // time of the directory as baseline in case of file deletion
  int64_t mtime = 0;
  status = FileModificationTime(path, &mtime);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status.AsString();
    return 0;
  }
  if (!path_is_dir) {
    return mtime;
  }

  // 'path' is a directory. Return the most recent mtime of the
  // contents of the directory.
  std::set<std::string> contents;
  status = GetDirectoryContents(path, &contents);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status.AsString();
    return 0;
  }

  for (const auto& child : contents) {
    const auto full_path = JoinPath({path, child});
    mtime = std::max(mtime, GetModifiedTime(full_path));
  }

  return mtime;
}
// Return true if any file in the subdirectory root at 'path' has been
// modified more recently than 'last'. Return the most-recent modified
// time in 'last'.
bool
IsModified(const std::string& path, int64_t* last_ns)
{
  const int64_t repo_ns = GetModifiedTime(path);
  bool modified = repo_ns > *last_ns;
  *last_ns = repo_ns;
  return modified;
}

}  // namespace

struct ModelRepositoryManager::ModelInfo {
  ModelInfo(
      const int64_t mtime_nsec, const int64_t prev_mtime_ns,
      const std::string& model_path)
      : mtime_nsec_(mtime_nsec), prev_mtime_ns_(prev_mtime_ns),
        explicitly_load_(true), model_path_(model_path),
        is_config_provided_(false)
  {
  }
  ModelInfo()
      : mtime_nsec_(0), prev_mtime_ns_(0), explicitly_load_(true),
        is_config_provided_(false)
  {
  }
  int64_t mtime_nsec_;
  int64_t prev_mtime_ns_;
  bool explicitly_load_;
  inference::ModelConfig model_config_;
  std::string model_path_;
  // Temporary location to hold agent model list before creating the model
  // the ownership must transfer to ModelLifeCycle to ensure
  // the agent model life cycle is handled properly.
  std::shared_ptr<TritonRepoAgentModelList> agent_model_list_;
  bool is_config_provided_;
};

ModelRepositoryManager::ModelRepositoryManager(
    const std::set<std::string>& repository_paths, const bool autofill,
    const bool polling_enabled, const bool model_control_enabled,
    const double min_compute_capability,
    std::unique_ptr<ModelLifeCycle> life_cycle)
    : repository_paths_(repository_paths), autofill_(autofill),
      polling_enabled_(polling_enabled),
      model_control_enabled_(model_control_enabled),
      min_compute_capability_(min_compute_capability),
      model_life_cycle_(std::move(life_cycle))
{
}

ModelRepositoryManager::~ModelRepositoryManager() {}

Status
ModelRepositoryManager::Create(
    InferenceServer* server, const std::string& server_version,
    const std::set<std::string>& repository_paths,
    const std::set<std::string>& startup_models, const bool strict_model_config,
    const bool polling_enabled, const bool model_control_enabled,
    const ModelLifeCycleOptions& life_cycle_options,
    std::unique_ptr<ModelRepositoryManager>* model_repository_manager)
{
  // The rest only matters if repository path is valid directory
  for (const auto& path : repository_paths) {
    bool path_is_dir;
    RETURN_IF_ERROR(IsDirectory(path, &path_is_dir));
    if (!path_is_dir) {
      return Status(
          Status::Code::INVALID_ARG,
          "repository path is not a valid directory");
    }
  }

  if (polling_enabled && model_control_enabled) {
    return Status(
        Status::Code::INVALID_ARG,
        "cannot enable both polling and explicit model control");
  }

  std::unique_ptr<ModelLifeCycle> life_cycle;
  RETURN_IF_ERROR(
      ModelLifeCycle::Create(server, life_cycle_options, &life_cycle));

  // Not setting the smart pointer directly to simplify clean up
  std::unique_ptr<ModelRepositoryManager> local_manager(
      new ModelRepositoryManager(
          repository_paths, !strict_model_config, polling_enabled,
          model_control_enabled, life_cycle_options.min_compute_capability_,
          std::move(life_cycle)));
  *model_repository_manager = std::move(local_manager);

  // Support loading all models on startup in explicit model control mode with
  // special startup_model name "*". This does not imply support for pattern
  // matching in model names.
  bool load_all_models_on_startup = false;
  if ((startup_models.find("*") != startup_models.end()) &&
      model_control_enabled) {
    if (startup_models.size() > 1) {
      return Status(
          Status::Code::INVALID_ARG,
          "Wildcard model name '*' must be the ONLY startup model "
          "if specified at all.");
    }

    load_all_models_on_startup = true;
  }

  bool all_models_polled = true;
  if (!model_control_enabled || load_all_models_on_startup) {
    // only error happens before model load / unload will be return
    // model loading / unloading error will be printed but ignored
    RETURN_IF_ERROR(
        (*model_repository_manager)->PollAndUpdateInternal(&all_models_polled));
  } else {
    // Load each specified startup_model
    std::unordered_map<std::string, std::vector<const InferenceParameter*>>
        models;
    for (const auto& model_name : startup_models) {
      models[model_name];
    }
    RETURN_IF_ERROR(
        (*model_repository_manager)
            ->LoadUnloadModels(
                models, ActionType::LOAD, false, &all_models_polled));
  }


  if (!all_models_polled) {
    return Status(Status::Code::INTERNAL, "failed to load all models");
  }
  // Some models may failed to be loaded after model manager is created,
  // return proper error and let function caller decide whether to proceed.
  for (const auto& model : (*model_repository_manager)->infos_) {
    const auto version_states =
        (*model_repository_manager)
            ->model_life_cycle_->VersionStates(model.first);
    // Return general error message, detail of each model's loading state
    // is logged separately.
    if (version_states.empty()) {
      return Status(Status::Code::INTERNAL, "failed to load all models");
    }
    for (const auto& state : version_states) {
      if (state.second.first != ModelReadyState::READY) {
        return Status(Status::Code::INTERNAL, "failed to load all models");
      }
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::PollAndUpdate()
{
  if (!polling_enabled_) {
    return Status(Status::Code::UNAVAILABLE, "polling is disabled");
  }

  bool all_models_polled;
  return PollAndUpdateInternal(&all_models_polled);
}

Status
ModelRepositoryManager::PollAndUpdateInternal(bool* all_models_polled)
{
  // Serialize all operations that change model state
  std::lock_guard<std::mutex> lock(poll_mu_);

  std::set<std::string> added, deleted, modified, unmodified;

  // We don't modify 'infos_' in place to minimize how long we need to
  // hold the lock and also prevent any partial changes to do an error
  // during processing.
  ModelInfoMap new_infos;

  // Each subdirectory of repository path is a model directory from
  // which we read the model configuration.
  std::unordered_map<std::string, std::vector<const InferenceParameter*>>
      subdirs;
  RETURN_IF_ERROR(Poll(
      subdirs, &added, &deleted, &modified, &unmodified, &new_infos,
      all_models_polled));

  // Anything in 'infos_' that is not in "added", "modified", or
  // "unmodified" is deleted.
  for (const auto& pr : infos_) {
    if ((added.find(pr.first) == added.end()) &&
        (modified.find(pr.first) == modified.end()) &&
        (unmodified.find(pr.first) == unmodified.end())) {
      deleted.insert(pr.first);
    }
  }

  // Nothing to do if no model adds, deletes or modifies.
  if (added.empty() && deleted.empty() && modified.empty()) {
    return Status::Success;
  }

  infos_.swap(new_infos);

  UpdateDependencyGraph(added, deleted, modified);

  for (const auto& name : deleted) {
    model_life_cycle_->AsyncUnload(name);
  }

  // model loading / unloading error will be printed but ignored
  LoadModelByDependency();

  return Status::Success;
}

std::map<std::string, Status>
ModelRepositoryManager::LoadModelByDependency()
{
  std::map<std::string, Status> res;
  struct ModelState {
    ModelState(DependencyNode* node) : node_(node), status_(Status::Success) {}
    DependencyNode* node_;
    Status status_;
    std::promise<void> ready_;
  };
  NodeSet loaded_models;
  auto set_pair = ModelsToLoadUnload(loaded_models);
  // Loop until all model are loaded / unloaded
  while ((!set_pair.first.empty()) || (!set_pair.second.empty())) {
    loaded_models.clear();
    // Unload invalid models first
    for (auto& invalid_model : set_pair.second) {
      model_life_cycle_->AsyncUnload(invalid_model->model_name_);
      LOG_ERROR << invalid_model->status_.AsString();
      invalid_model->loaded_versions_ = std::set<int64_t>();
      loaded_models.emplace(invalid_model);
    }
    // load valid models and wait for load results
    std::vector<std::unique_ptr<ModelState>> model_states;
    for (auto& valid_model : set_pair.first) {
      model_states.emplace_back(new ModelState(valid_model));
      auto model_state = model_states.back().get();
      const auto itr = infos_.find(valid_model->model_name_);
      auto status = model_life_cycle_->AsyncLoad(
          valid_model->model_name_, itr->second->model_path_,
          valid_model->model_config_, itr->second->is_config_provided_,
          itr->second->agent_model_list_, [model_state](Status load_status) {
            model_state->status_ = load_status;
            model_state->ready_.set_value();
          });
      if (!status.IsOk()) {
        model_state->status_ = status;
        model_state->ready_.set_value();
        LOG_ERROR << "failed to load model '" << valid_model->model_name_
                  << "': " << status.Message();
      }
      loaded_models.emplace(valid_model);
    }
    for (auto& model_state : model_states) {
      model_state->ready_.get_future().wait();
      res[model_state->node_->model_name_] = model_state->status_;
      const auto version_state =
          model_life_cycle_->VersionStates(model_state->node_->model_name_);
      model_state->node_->loaded_versions_.clear();
      for (const auto& vs : version_state) {
        if (vs.second.first == ModelReadyState::READY) {
          model_state->node_->loaded_versions_.emplace(vs.first);
        }
      }
      // If the model failed to load, should revert the timestamp to
      // ensure the next load request will attempt to load the model again
      // for operation consistency.
      if (!model_state->status_.IsOk()) {
        auto& model_info = infos_.find(model_state->node_->model_name_)->second;
        model_info->mtime_nsec_ = model_info->prev_mtime_ns_;
      }
    }
    set_pair = ModelsToLoadUnload(loaded_models);
  }
  // Clear temporary stored agent model list after all loads are triggerred
  for (auto& info : infos_) {
    info.second->agent_model_list_.reset();
  }
  return res;
}

Status
ModelRepositoryManager::LoadUnloadModel(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models,
    const ActionType type, const bool unload_dependents)
{
  if (!model_control_enabled_) {
    return Status(
        Status::Code::UNAVAILABLE,
        "explicit model load / unload is not allowed if polling is enabled");
  }

  if (models.size() > 1) {
    return Status(
        Status::Code::UNSUPPORTED,
        "explicit load / unload multiple models is not currently supported");
  }

  // Serialize all operations that change model state
  std::lock_guard<std::mutex> lock(poll_mu_);

  bool polled = true;
  RETURN_IF_ERROR(LoadUnloadModels(models, type, unload_dependents, &polled));
  // Check if model is loaded / unloaded properly
  const auto& model_name = models.begin()->first;
  if (!polled) {
    return Status(
        Status::Code::INTERNAL, "failed to load '" + model_name +
                                    "', failed to poll from model repository");
  }

  const auto version_states = model_life_cycle_->VersionStates(model_name);
  if (type == ActionType::LOAD) {
    if (version_states.empty()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to load '" + model_name + "', no version is available");
    }
    auto it = infos_.find(model_name);
    if (it == infos_.end()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to load '" + model_name +
              "', failed to poll from model repository");
    }
  } else {
    std::string ready_version_str;
    for (const auto& version_state : version_states) {
      if (version_state.second.first == ModelReadyState::READY) {
        ready_version_str += std::to_string(version_state.first);
        ready_version_str += ",";
      }
    }
    if (!ready_version_str.empty()) {
      ready_version_str.pop_back();
      return Status(
          Status::Code::INTERNAL,
          "failed to unload '" + model_name +
              "', versions that are still available: " + ready_version_str);
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::LoadUnloadModels(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models,
    const ActionType type, const bool unload_dependents,
    bool* all_models_polled)
{
  auto status = Status::Success;
  *all_models_polled = true;
  // Update ModelInfo related to file system accordingly
  std::set<std::string> added, deleted, modified, unmodified;
  {
    if (type == ActionType::UNLOAD) {
      for (const auto& model : models) {
        deleted.insert(model.first);
      }
    }
    // ActionType::LOAD and in model control mode
    else {
      std::set<std::string> checked_models;
      auto current_models = models;
      for (const auto& model : models) {
        checked_models.emplace(model.first);
      }

      ModelInfoMap new_infos;
#ifdef TRITON_ENABLE_ENSEMBLE
      bool first_iteration = true;
#endif  // TRITON_ENABLE_ENSEMBLE
      while (!current_models.empty()) {
        bool polled = true;
        RETURN_IF_ERROR(Poll(
            current_models, &added, &deleted, &modified, &unmodified,
            &new_infos, &polled));
        *all_models_polled &= polled;

        // More models should be polled if the polled models are ensembles
        std::unordered_map<std::string, std::vector<const InferenceParameter*>>
            next_models;
#ifdef TRITON_ENABLE_ENSEMBLE
        for (const auto& model : current_models) {
          auto it = new_infos.find(model.first);
          // Some models may be marked as deleted and not in 'new_infos'
          if (it != new_infos.end()) {
            it->second->explicitly_load_ = first_iteration;
            const auto& config = it->second->model_config_;
            if (config.has_ensemble_scheduling()) {
              for (const auto& step : config.ensemble_scheduling().step()) {
                bool need_poll =
                    checked_models.emplace(step.model_name()).second;
                if (need_poll) {
                  next_models[step.model_name()];
                }
              }
            }
          }
        }
        first_iteration = false;
#endif  // TRITON_ENABLE_ENSEMBLE
        current_models.swap(next_models);
      }

      // Only update the infos when all validation is completed
      for (const auto& model_name : added) {
        auto nitr = new_infos.find(model_name);
        infos_.emplace(model_name, std::move(nitr->second));
      }
      for (const auto& model_name : modified) {
        auto nitr = new_infos.find(model_name);
        auto itr = infos_.find(model_name);
        itr->second = std::move(nitr->second);
      }
    }
  }
  std::set<std::string> deleted_dependents;

  // Update dependency graph and load
  UpdateDependencyGraph(
      added, deleted, modified,
      unload_dependents ? &deleted_dependents : nullptr);

  // The models are in 'deleted' either when they are asked to be unloaded or
  // they are not found / are duplicated across all model repositories.
  // In all cases, should unload them and remove from 'infos_' explicitly.
  for (const auto& name : (unload_dependents ? deleted_dependents : deleted)) {
    infos_.erase(name);
    model_life_cycle_->AsyncUnload(name);
  }

  // load / unload the models affected, and check the load status of
  // the requested models
  const auto& load_status = LoadModelByDependency();
  if (status.IsOk() && (type == ActionType::LOAD)) {
    std::string load_error_message = "";
    for (const auto& model : models) {
      auto it = load_status.find(model.first);
      // If 'model.first' not in load status, it means the (re-)load is not
      // necessary because there is no change in the model's directory
      if ((it != load_status.end()) && !it->second.IsOk()) {
        load_error_message +=
            ("load failed for model '" + model.first +
             "': " + it->second.Message() + "\n");
      }
    }
    if (!load_error_message.empty()) {
      status = Status(Status::Code::INVALID_ARG, load_error_message);
    }
  }

  return status;
}

Status
ModelRepositoryManager::UnloadAllModels()
{
  Status status;
  for (const auto& name_info : infos_) {
    Status unload_status = model_life_cycle_->AsyncUnload(name_info.first);
    if (!unload_status.IsOk()) {
      status = Status(
          unload_status.ErrorCode(),
          "Failed to gracefully unload models: " + unload_status.Message());
    }
  }
  return Status::Success;
}

Status
ModelRepositoryManager::StopAllModels()
{
  return model_life_cycle_->StopAllModels();
}

const std::set<std::tuple<std::string, int64_t, size_t>>
ModelRepositoryManager::InflightStatus()
{
  return model_life_cycle_->InflightStatus();
}

const ModelStateMap
ModelRepositoryManager::LiveModelStates(bool strict_readiness)
{
  return model_life_cycle_->LiveModelStates(strict_readiness);
}

const ModelStateMap
ModelRepositoryManager::ModelStates()
{
  return model_life_cycle_->ModelStates();
}

const VersionStateMap
ModelRepositoryManager::VersionStates(const std::string& model_name)
{
  return model_life_cycle_->VersionStates(model_name);
}

Status
ModelRepositoryManager::ModelState(
    const std::string& model_name, const int64_t model_version,
    ModelReadyState* state)
{
  return model_life_cycle_->ModelState(model_name, model_version, state);
}

Status
ModelRepositoryManager::RepositoryIndex(
    const bool ready_only, std::vector<ModelIndex>* index)
{
  std::set<std::string> seen_models;
  std::set<std::string> duplicate_models;
  for (const auto& repository_path : repository_paths_) {
    // For any mapped models in this repository, save the mapping
    // from their subdirectory name to model name.
    std::map<std::string, std::string> models_in_repo;
    for (const auto& mapping_it : model_mappings_) {
      if (mapping_it.second.first == repository_path) {
        models_in_repo.emplace(
            BaseName(mapping_it.second.second), mapping_it.first);
      }
    }
    std::set<std::string> subdirs;
    RETURN_IF_ERROR(GetDirectorySubdirs(repository_path, &subdirs));
    for (const auto& subdir : subdirs) {
      auto model = subdir;
      auto model_it = models_in_repo.find(subdir);
      if (model_it != models_in_repo.end()) {
        model = model_it->second;
      }

      if (seen_models.find(model) != seen_models.end()) {
        duplicate_models.insert(model);
      }

      seen_models.insert(model);
    }
  }

  ModelStateMap states = ModelStates();

  for (const auto& model : seen_models) {
    // If the same model appears in multiple repostories then show it
    // as unavailable since duplicate models are not allowed to load.
    if (duplicate_models.find(model) != duplicate_models.end()) {
      index->emplace_back(
          model, -1 /* version */, ModelReadyState::UNAVAILABLE,
          MODEL_READY_REASON_DUPLICATE);
      continue;
    }

    // If there is any version/state/reason associated with the model
    // then include that in the index.
    auto sitr = states.find(model);
    if (sitr == states.end()) {
      if (!ready_only) {
        index->emplace_back(model);
      }
    } else {
      for (const auto& pr : sitr->second) {
        if (!ready_only || (pr.second.first == ModelReadyState::READY)) {
          index->emplace_back(
              model, pr.first, pr.second.first, pr.second.second);
        }
      }
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::GetModel(
    const std::string& model_name, const int64_t model_version,
    std::shared_ptr<Model>* model)
{
  Status status = model_life_cycle_->GetModel(model_name, model_version, model);
  if (!status.IsOk()) {
    model->reset();
    status = Status(
        status.ErrorCode(), "Request for unknown model: " + status.Message());
  }
  return status;
}

Status
ModelRepositoryManager::Poll(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models,
    std::set<std::string>* added, std::set<std::string>* deleted,
    std::set<std::string>* modified, std::set<std::string>* unmodified,
    ModelInfoMap* updated_infos, bool* all_models_polled)
{
  *all_models_polled = true;
  // empty path is the special case to indicate the model should be loaded
  // from override file content in 'models'.
  std::map<std::string, std::string> model_to_path;

  // If no model is specified, poll all models in all model repositories.
  // Otherwise, only poll the specified models
  if (models.empty()) {
    std::set<std::string> duplicated_models;
    for (const auto& repository_path : repository_paths_) {
      std::set<std::string> subdirs;
      Status status = GetDirectorySubdirs(repository_path, &subdirs);
      if (!status.IsOk()) {
        LOG_ERROR << "failed to poll model repository '" << repository_path
                  << "': " << status.Message();
        *all_models_polled = false;
      } else {
        for (const auto& subdir : subdirs) {
          if (!model_to_path
                   .emplace(subdir, JoinPath({repository_path, subdir}))
                   .second) {
            duplicated_models.insert(subdir);
            *all_models_polled = false;
          }
        }
      }
    }
    // If the model is not unique, mark as deleted to unload it
    for (const auto& model : duplicated_models) {
      model_to_path.erase(model);
      deleted->insert(model);
      LOG_ERROR << "failed to poll model '" << model
                << "': not unique across all model repositories";
    }
  }
  // If models are specified, this is explicit model control mode.
  else {
    for (const auto& model : models) {
      // Skip repository polling if override model files
      if (ModelDirectoryOverride(model.second)) {
        model_to_path.emplace(model.first, "");
        continue;
      }
      // Check model mapping first to see if matching model to load.
      bool exists = false;
      auto model_it = model_mappings_.find(model.first);
      if (model_it != model_mappings_.end()) {
        bool exists_in_this_repo = false;
        auto full_path = model_it->second.second;
        Status status = FileExists(full_path, &exists_in_this_repo);
        if (!status.IsOk()) {
          LOG_ERROR << "failed to poll mapped path '" << full_path
                    << "' for model '" << model.first
                    << "': " << status.Message();
          *all_models_polled = false;
        }
        if (exists_in_this_repo) {
          model_to_path.emplace(model.first, model_it->second.second);
          exists = true;
        } else {
          LOG_ERROR << "mapped path '" << full_path
                    << "' does not exist for model '" << model.first << "'";
          exists = false;
        }
      } else {
        for (const auto repository_path : repository_paths_) {
          bool exists_in_this_repo = false;
          const auto full_path = JoinPath({repository_path, model.first});
          Status status = FileExists(full_path, &exists_in_this_repo);
          if (!status.IsOk()) {
            LOG_ERROR << "failed to poll model repository '" << repository_path
                      << "' for model '" << model.first
                      << "': " << status.Message();
            *all_models_polled = false;
          } else if (exists_in_this_repo) {
            // Check to make sure this directory is not mapped.
            // If mapped, continue to next repository path.
            bool mapped = false;
            for (auto const& mapping : model_mappings_) {
              if (mapping.second.second == full_path) {
                mapped = true;
                break;
              }
            }
            if (mapped) {
              continue;
            }

            auto res = model_to_path.emplace(
                model.first, JoinPath({repository_path, model.first}));
            if (res.second) {
              exists = true;
            } else {
              exists = false;
              model_to_path.erase(res.first);
              LOG_ERROR << "failed to poll model '" << model.first
                        << "': not unique across all model repositories";
              break;
            }
          }
        }
      }
      // For an explicitly specified model that doesn't exist, we don't mark it
      // as deleted, we simply mark that we couldn't poll all models.
      if (!exists) {
        *all_models_polled = false;
      }
    }
  }

  // Poll each of the models. If error happens during polling the model,
  // its state will fallback to the state before the polling.
  for (const auto& pair : model_to_path) {
    std::unique_ptr<ModelInfo> model_info;
    const auto& mit = models.find(pair.first);
    static std::vector<const InferenceParameter*> empty_params;
    auto status = InitializeModelInfo(
        pair.first, pair.second,
        ((mit == models.end()) ? empty_params : mit->second), &model_info);

    const auto& iitr = infos_.find(pair.first);
    const bool invalid_add = (!status.IsOk()) && (iitr == infos_.end());
    if (!invalid_add) {
      const auto& ret = updated_infos->emplace(pair.first, nullptr);
      if (!ret.second) {
        return Status(
            Status::Code::ALREADY_EXISTS,
            "unexpected model info for model '" + pair.first + "'");
      }

      // Classify load state and set updated info
      if (model_info == nullptr) {
        ret.first->second.reset(new ModelInfo(*iitr->second));
        unmodified->insert(pair.first);
      } else {
        ret.first->second = std::move(model_info);
        if (iitr != infos_.end()) {
          modified->insert(pair.first);
        } else {
          added->insert(pair.first);
        }
      }
    }

    if (!status.IsOk()) {
      LOG_ERROR << "Poll failed for model directory '" << pair.first
                << "': " << status.Message();
      *all_models_polled = false;
    }
  }

  return Status::Success;
}

bool
ModelRepositoryManager::ModelDirectoryOverride(
    const std::vector<const InferenceParameter*>& model_params)
{
  for (const auto& param : model_params) {
    if (param->Name().rfind(file_prefix, 0) == 0) {
      // param name starts with prefix if user provides override file
      return true;
    }
  }
  return false;
}

Status
ModelRepositoryManager::InitializeModelInfo(
    const std::string& name, const std::string& path,
    const std::vector<const InferenceParameter*>& params,
    std::unique_ptr<ModelInfo>* info)
{
  std::unique_ptr<ModelInfo> linfo(new ModelInfo());
  linfo->model_path_ = path;

  bool unmodified = false;

  const auto iitr = infos_.find(name);
  // Set 'prev_mtime_ns_' if there is existing ModelInfo
  if (iitr != infos_.end()) {
    linfo->prev_mtime_ns_ = iitr->second->mtime_nsec_;
  } else {
    linfo->prev_mtime_ns_ = 0;
  }

  // Set 'mtime_nsec_' and override 'model_path_' if current path is empty
  // (file override is specified)
  if (linfo->model_path_.empty()) {
    // Need to localize the override files, use repo agent to manage
    // the lifecycle of the localized files
    std::shared_ptr<TritonRepoAgent> localize_agent(new LocalizeRepoAgent());
    std::unique_ptr<TritonRepoAgentModel> localize_agent_model;
    RETURN_IF_ERROR(TritonRepoAgentModel::Create(
        TRITONREPOAGENT_ARTIFACT_FILESYSTEM, "", inference::ModelConfig(),
        localize_agent, {}, &localize_agent_model));

    // Set agent model state so the repo agent can access the encoded files
    // Using const_cast here but we are safe as the RepoAgent will not
    // modify the state
    localize_agent_model->SetState(
        const_cast<void*>(reinterpret_cast<const void*>(&params)));
    RETURN_IF_ERROR(
        localize_agent_model->InvokeAgent(TRITONREPOAGENT_ACTION_LOAD));

    const char* location;
    TRITONREPOAGENT_ArtifactType type;
    RETURN_IF_ERROR(localize_agent_model->Location(&type, &location));

    // For file override, set 'mtime_nsec_' to minimum value so that
    // the next load without override will trigger re-load to undo
    // the override while the local files may still be unchanged.
    linfo->mtime_nsec_ = 0;
    linfo->model_path_ = location;
    linfo->agent_model_list_.reset(new TritonRepoAgentModelList());
    linfo->agent_model_list_->AddAgentModel(std::move(localize_agent_model));
  } else {
    if (iitr == infos_.end()) {
      linfo->mtime_nsec_ = GetModifiedTime(std::string(linfo->model_path_));
    } else {
      // Check the current timestamps to determine if model actually has been
      // modified
      linfo->mtime_nsec_ = linfo->prev_mtime_ns_;
      unmodified =
          !IsModified(std::string(linfo->model_path_), &linfo->mtime_nsec_);
    }
  }

  // Set 'model_config_'
  bool parsed_config = false;
  // Check if there is config override
  for (const auto& override_parameter : params) {
    if ((override_parameter->Name() == "config") &&
        (override_parameter->Type() == TRITONSERVER_PARAMETER_STRING)) {
      // When override happens, set 'mtime_nsec_' to minimum value so that
      // the next load without override will trigger re-load to undo
      // the override while the local files may still be unchanged.
      linfo->mtime_nsec_ = 0;
      unmodified = false;

      const std::string& override_config = override_parameter->ValueString();
      auto err = JsonToModelConfig(
          override_config, 1 /* config_version */, &linfo->model_config_);
      if (!err.IsOk()) {
        return Status(
            Status::Code::INVALID_ARG,
            "Invalid config override: " + std::string(err.Message()));
      }
      parsed_config = true;
      break;
    } else if (override_parameter->Name().rfind(file_prefix, 0) != 0) {
      return Status(
          Status::Code::INVALID_ARG,
          "Unrecognized load parameter '" + override_parameter->Name() +
              "' with type '" +
              TRITONSERVER_ParameterTypeString(override_parameter->Type()) +
              "'");
    }
  }

  // Polling model is considered unmodified by this point and can be returned
  // with info == nullptr
  if (unmodified) {
    return Status::Success;
  }

  // Create the associated repo agent models when a model is to be loaded,
  // this must be done before normalizing model config as agents might
  // redirect to use the model config at a different location
  if (!parsed_config) {
    const auto config_path = JoinPath({linfo->model_path_, kModelConfigPbTxt});
    bool model_config_exists = false;
    RETURN_IF_ERROR(FileExists(config_path, &model_config_exists));
    // model config can be missing if auto fill is set
    if (autofill_ && !model_config_exists) {
      linfo->model_config_.Clear();
    } else {
      RETURN_IF_ERROR(ReadTextProto(config_path, &linfo->model_config_));
      parsed_config = true;
    }
  }
  if (parsed_config) {
    RETURN_IF_ERROR(CreateAgentModelListWithLoadAction(
        linfo->model_config_, linfo->model_path_, &linfo->agent_model_list_));
    if (linfo->agent_model_list_ != nullptr) {
      // Get the latest repository path
      const char* location;
      TRITONREPOAGENT_ArtifactType artifact_type;
      RETURN_IF_ERROR(linfo->agent_model_list_->Back()->Location(
          &artifact_type, &location));
      auto latest_path = std::string(location);
      linfo->model_path_ = latest_path;
    }
  }
  linfo->is_config_provided_ = parsed_config;

  // Try to automatically generate missing parts of the model
  // configuration (autofill) that don't require model detail
  RETURN_IF_ERROR(GetNormalizedModelConfig(
      name, linfo->model_path_, min_compute_capability_,
      &linfo->model_config_));

  // Note that the model inputs and outputs are not validated until
  // the model model is intialized as they may not be auto-completed
  // until model is intialized.
  RETURN_IF_ERROR(
      ValidateModelConfig(linfo->model_config_, min_compute_capability_));
  if (!autofill_) {
    RETURN_IF_ERROR(ValidateModelIOConfig(linfo->model_config_));
  }

  // If the model is mapped, update its config name based on the
  // mapping.
  if (model_mappings_.find(name) != model_mappings_.end()) {
    linfo->model_config_.set_name(name);
  } else {
    // If there is no model mapping, make sure the name of the model
    // matches the name of the directory. This is a somewhat arbitrary
    // requirement but seems like good practice to require it of the user.
    // It also acts as a check to make sure we don't have two different
    // models with the same name.
    if (linfo->model_config_.name() != name) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected directory name '" + name + "' for model '" +
              linfo->model_config_.name() +
              "', directory name must equal model name");
    }
  }

  *info = std::move(linfo);
  return Status::Success;
}

Status
ModelRepositoryManager::UpdateDependencyGraph(
    const std::set<std::string>& added, const std::set<std::string>& deleted,
    const std::set<std::string>& modified,
    std::set<std::string>* deleted_dependents)
{
  // update dependency graph, if the state of a node is changed, all its
  // downstreams will be affected

  // deleted, drop from dependency_graph, add to missing_nodes if downstreams is
  // not empty affected_nodes are all ensembles as only ensembles are depending
  // on other models
  std::set<DependencyNode*> affected_nodes;
  std::set<DependencyNode*> updated_nodes;
  std::set<std::string> current_deleted = deleted;
  while (!current_deleted.empty()) {
    std::set<std::string> next_deleted;
    for (const auto& model_name : current_deleted) {
      auto it = dependency_graph_.find(model_name);
      if (it != dependency_graph_.end()) {
        // remove this node from its upstreams
        for (auto& upstream : it->second->upstreams_) {
          upstream.first->downstreams_.erase(it->second.get());
          // Check if the upstream should be removed as well
          if ((deleted_dependents != nullptr) &&
              (upstream.first->downstreams_.empty()) &&
              (!upstream.first->explicitly_load_)) {
            next_deleted.emplace(upstream.first->model_name_);
          }
        }
        it->second->upstreams_.clear();

        if (!it->second->downstreams_.empty()) {
          UncheckDownstream(&it->second->downstreams_, &affected_nodes);
          // mark this node as missing upstream in its downstreams
          for (auto& downstream : it->second->downstreams_) {
            downstream->missing_upstreams_.emplace(it->second.get());
          }
          missing_nodes_.emplace(
              std::make_pair(model_name, std::move(it->second)));
        }

        // Make sure deleted node will not be in affected nodes
        affected_nodes.erase(it->second.get());
        dependency_graph_.erase(it);
      }
      if (deleted_dependents != nullptr) {
        deleted_dependents->emplace(model_name);
      }
    }
    current_deleted.swap(next_deleted);
  }

  // modified, invalidate (uncheck) all downstreams
  for (const auto& model_name : modified) {
    auto it = dependency_graph_.find(model_name);
    if (it != dependency_graph_.end()) {
      UncheckDownstream(&it->second->downstreams_, &affected_nodes);
      ModelInfo* info = nullptr;
      GetModelInfo(model_name, &info);
      it->second->model_config_ = info->model_config_;
      it->second->explicitly_load_ = info->explicitly_load_;
      // remove this node from its upstream node
      for (auto& upstream : it->second->upstreams_) {
        upstream.first->downstreams_.erase(it->second.get());
      }
      it->second->upstreams_.clear();
      it->second->checked_ = false;
      it->second->status_ = Status::Success;
      updated_nodes.emplace(it->second.get());
    }
  }

  // added, add to dependency_graph, if in missing_node, invalidate (uncheck)
  // and associate all downstreams, remove from missing_node
  for (const auto& model_name : added) {
    std::unique_ptr<DependencyNode> added_node;
    auto it = missing_nodes_.find(model_name);
    if (it != missing_nodes_.end()) {
      UncheckDownstream(&it->second->downstreams_, &affected_nodes);
      // remove this node from missing upstream node in its downstream nodes
      for (auto& downstream : it->second->downstreams_) {
        downstream->missing_upstreams_.erase(it->second.get());
      }

      it->second->checked_ = false;
      added_node = std::move(it->second);
      missing_nodes_.erase(it);
    } else {
      // Right now, nothing is going to be filled until validation
      added_node.reset(new DependencyNode(model_name));
    }
    ModelInfo* info = nullptr;
    GetModelInfo(model_name, &info);
    added_node->model_config_ = info->model_config_;
    added_node->explicitly_load_ = info->explicitly_load_;
    updated_nodes.emplace(added_node.get());
    dependency_graph_.emplace(
        std::make_pair(model_name, std::move(added_node)));
  }

  auto& affected_ensembles = affected_nodes;
  for (auto& updated_node : updated_nodes) {
    bool is_ensemble = ConnectDependencyGraph(updated_node);
    if (is_ensemble) {
      affected_ensembles.emplace(updated_node);
    }
  }

#ifdef TRITON_ENABLE_ENSEMBLE
  // After the dependency graph is updated, check ensemble dependencies
  for (auto& ensemble : affected_ensembles) {
    if (ensemble->status_.IsOk()) {
      if (!ensemble->missing_upstreams_.empty()) {
        std::string name_list;
        for (auto it = ensemble->missing_upstreams_.begin();
             it != ensemble->missing_upstreams_.end(); it++) {
          if (it != ensemble->missing_upstreams_.begin()) {
            name_list += ", ";
          }
          name_list += (*it)->model_name_;
        }
        ensemble->status_ = Status(
            Status::Code::INVALID_ARG,
            "ensemble " + ensemble->model_name_ +
                " contains models that are not available: " + name_list);
      } else {
        ensemble->status_ = CircularcyCheck(ensemble, ensemble);
      }
    }
  }
#endif  // TRITON_ENABLE_ENSEMBLE
  return Status::Success;
}

Status
ModelRepositoryManager::RegisterModelRepository(
    const std::string& repository,
    const std::unordered_map<std::string, std::string>& model_mapping)
{
  if (!model_control_enabled_) {
    return Status(
        Status::Code::UNSUPPORTED,
        "repository registration is not allowed if model control mode is not "
        "EXPLICIT");
  }
  bool is_directory = false;
  auto status = IsDirectory(repository, &is_directory);
  if (!status.IsOk() || !is_directory) {
    return Status(
        Status::Code::INVALID_ARG, (std::string("failed to register '") +
                                    repository + "', repository not found")
                                       .c_str());
  }

  {
    // Serialize all operations that change model state
    std::lock_guard<std::mutex> lock(poll_mu_);

    // Check repository and mapped models do not yet exist.
    if (repository_paths_.find(repository) != repository_paths_.end()) {
      return Status(
          Status::Code::ALREADY_EXISTS,
          "model repository '" + repository + "' has already been registered");
    }

    for (const auto& mapping : model_mapping) {
      if (model_mappings_.find(mapping.first) != model_mappings_.end()) {
        return Status(
            Status::Code::ALREADY_EXISTS,
            (std::string("failed to register '") + mapping.first +
             "', there is a conflicting mapping for '" +
             std::string(mapping.first) + "'")
                .c_str());
      }
    }

    repository_paths_.emplace(repository);
    for (const auto& mapping : model_mapping) {
      model_mappings_.emplace(
          mapping.first,
          std::make_pair(repository, JoinPath({repository, mapping.second})));
    }
  }

  LOG_INFO << "Model repository registered: " << repository;
  return Status::Success;
}

Status
ModelRepositoryManager::UnregisterModelRepository(const std::string& repository)
{
  if (!model_control_enabled_) {
    return Status(
        Status::Code::UNSUPPORTED,
        "repository unregistration is not allowed if model control mode is not "
        "EXPLICIT");
  }
  {
    std::lock_guard<std::mutex> lock(poll_mu_);
    if (repository_paths_.erase(repository) != 1) {
      return Status(
          Status::Code::INVALID_ARG,
          "failed to unregister '" + repository + "', repository not found");
    }

    std::set<std::string> models_to_delete;
    for (auto const& mapping : model_mappings_) {
      if (mapping.second.first == repository) {
        models_to_delete.insert(mapping.first);
      }
    }
    for (auto const& model : models_to_delete) {
      model_mappings_.erase(model);
    }
  }

  LOG_INFO << "Model repository unregistered: " << repository;
  return Status::Success;
}

Status
ModelRepositoryManager::CircularcyCheck(
    DependencyNode* current_node, const DependencyNode* start_node)
{
  for (auto& downstream : current_node->downstreams_) {
    if (downstream->model_name_ == start_node->model_name_) {
      return Status(
          Status::Code::INVALID_ARG,
          "circular dependency between ensembles: " + start_node->model_name_ +
              " -> ... -> " + current_node->model_name_ + " -> " +
              start_node->model_name_);
    } else {
      const auto status = CircularcyCheck(downstream, start_node);
      if (!status.IsOk() && current_node->status_.IsOk()) {
        current_node->status_ = status;
        return status;
      }
    }
  }
  return Status::Success;
}

void
ModelRepositoryManager::UncheckDownstream(
    NodeSet* downstreams, NodeSet* updated_nodes)
{
  // Mark downstream nodes as unchecked recursively
  for (auto& node : *downstreams) {
    if (node->checked_) {
      node->checked_ = false;
      node->status_ = Status::Success;
      UncheckDownstream(&node->downstreams_, updated_nodes);
      updated_nodes->emplace(node);
    }
  }
}

bool
ModelRepositoryManager::ConnectDependencyGraph(DependencyNode* updated_node)
{
  // Check the node's model config to determine if it depends on other models
  // and if those models are present
  updated_node->upstreams_.clear();
  updated_node->missing_upstreams_.clear();
  if (updated_node->model_config_.has_ensemble_scheduling()) {
    for (const auto& step :
         updated_node->model_config_.ensemble_scheduling().step()) {
      DependencyNode* upstream_node = nullptr;
      const auto& model_name = step.model_name();
      auto dit = dependency_graph_.find(model_name);
      if (dit == dependency_graph_.end()) {
        auto mit = missing_nodes_.find(model_name);
        if (mit == missing_nodes_.end()) {
          std::unique_ptr<DependencyNode> node(new DependencyNode(model_name));
          updated_node->missing_upstreams_.emplace(node.get());
          mit = missing_nodes_.emplace(model_name, std::move(node)).first;
        }
        // Add the node to missing node's downstream so that when the missing
        // node is added, the downstreams can be found easily.
        mit->second->downstreams_.emplace(updated_node);
        upstream_node = mit->second.get();
      } else {
        dit->second->downstreams_.emplace(updated_node);
        upstream_node = dit->second.get();
      }
      auto res = updated_node->upstreams_.emplace(
          upstream_node, std::set<int64_t>({step.model_version()}));
      // If map insertion doesn't happen, the same model is required in
      // different step, insert the version to existing required version set.
      if (!res.second) {
        res.first->second.insert(step.model_version());
      }
    }
    return true;
  }
  return false;
}

Status
ModelRepositoryManager::GetModelInfo(
    const std::string& name, ModelInfo** model_info)
{
  const auto itr = infos_.find(name);
  if (itr == infos_.end()) {
    return Status(
        Status::Code::NOT_FOUND, "no configuration for model '" + name + "'");
  }

  *model_info = itr->second.get();
  return Status::Success;
}

std::pair<ModelRepositoryManager::NodeSet, ModelRepositoryManager::NodeSet>
ModelRepositoryManager::ModelsToLoadUnload(const NodeSet& loaded_models)
{
  // <valid model set, invalid model set>
  std::pair<NodeSet, NodeSet> res;
  // first call to this function
  if (loaded_models.empty()) {
    for (auto& pair : dependency_graph_) {
      auto node = pair.second.get();
      // only care about nodes that are affected by the update
      if (!node->checked_) {
        if (CheckNode(node)) {
          if (node->status_.IsOk()) {
            res.first.emplace(node);
          } else {
            res.second.emplace(node);
          }
        }
      }
    }
  } else {
    for (const auto& model : loaded_models) {
      for (auto node : model->downstreams_) {
        // only care about nodes that are affected by the update
        if (!node->checked_) {
          if (CheckNode(node)) {
            if (node->status_.IsOk()) {
              res.first.emplace(node);
            } else {
              res.second.emplace(node);
            }
          }
        }
      }
    }
  }
  for (auto& node : res.first) {
    node->checked_ = true;
  }
  for (auto& node : res.second) {
    node->checked_ = true;
  }
  return res;
}

bool
ModelRepositoryManager::CheckNode(DependencyNode* node)
{
  bool node_ready = true;
  // if the node is in invalid status, mark as ready as we know
  // it should not be loaded
  if (node->status_.IsOk()) {
    for (auto& upstream : node->upstreams_) {
      if (!upstream.first->checked_) {
        node_ready = false;
        break;
      }
      if (!upstream.first->status_.IsOk()) {
        node->status_ = Status(
            Status::Code::INVALID_ARG,
            "ensemble '" + node->model_name_ + "' depends on '" +
                upstream.first->model_name_ + "' which is not valid");
      } else if (upstream.first->loaded_versions_.empty()) {
        node->status_ = Status(
            Status::Code::INVALID_ARG,
            "ensemble '" + node->model_name_ + "' depends on '" +
                upstream.first->model_name_ + "' which has no loaded version");
      } else {
        for (const auto& required_version : upstream.second) {
          if (required_version == -1) {
            continue;
          }

          auto it = upstream.first->loaded_versions_.find(required_version);
          if (it == upstream.first->loaded_versions_.end()) {
            node->status_ = Status(
                Status::Code::INVALID_ARG,
                "ensemble '" + node->model_name_ + "' depends on '" +
                    upstream.first->model_name_ + "' whose required version " +
                    std::to_string(required_version) + " is not loaded");
          }
        }
      }
      if (!node->status_.IsOk()) {
        break;
      }
    }
#ifdef TRITON_ENABLE_ENSEMBLE
    // Validate ensemble config if the node is ready. By this point, the
    // depending models are loaded and their configs are completed
    if (node_ready && node->status_.IsOk()) {
      node->status_ = ValidateEnsembleConfig(this, node);
    }
#endif  // TRITON_ENABLE_ENSEMBLE
  }
  return node_ready;
}

}}  // namespace triton::core
