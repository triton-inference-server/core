// Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <filesystem>
#include <future>
#include <stdexcept>
#include <thread>

#include "backend_model.h"
#include "constants.h"
#include "ensemble_scheduler/ensemble_utils.h"
#include "filesystem/api.h"
#include "model.h"
#include "model_config_utils.h"
#include "triton/common/logging.h"
#ifdef TRITON_ENABLE_ENSEMBLE
#include "ensemble_scheduler/ensemble_model.h"
#endif  // TRITON_ENABLE_ENSEMBLE

namespace triton { namespace core {

namespace {

template <typename T>
void
AddToSet(const std::set<T>& src, std::set<T>* dest)
{
  // std::set::merge() can be used if move to >= C++17,
  // note this is different from merge: copy item instead of move
  for (const auto& item : src) {
    dest->emplace(item);
  }
}

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
          const std::string temp_dir =
              std::filesystem::canonical(temp_dir_cstr).string();
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

              // Resolve any relative paths or symlinks, and enforce that target
              // directory stays within model directory for security.
              const std::string file_path =
                  std::filesystem::weakly_canonical(
                      JoinPath(
                          {temp_dir, file->Name().substr(file_prefix.size())}))
                      .string();
              if (file_path.rfind(temp_dir, 0) != 0) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string("Invalid file parameter '") + file->Name() +
                     "' with normalized path '" + file_path +
                     "' must stay within model directory.")
                        .c_str());
              }

              // Save model override file to the instructed directory using the
              // temporary model directory as the basepath.
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

              // Write file contents at specified path
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

/// Get the model config path to load for the model.
const std::string
GetModelConfigFullPath(
    const std::string& model_dir_path, const std::string& custom_config_name)
{
  // "--model-config-name" is set. Select custom config from
  // "<model_dir_path>/configs" folder if config file exists.
  if (!custom_config_name.empty()) {
    bool custom_config_exists = false;
    const std::string custom_config_path = JoinPath(
        {model_dir_path, kModelConfigFolder,
         custom_config_name + kPbTxtExtension});

    Status status = FileExists(custom_config_path, &custom_config_exists);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to get model configuration full path for '"
                << model_dir_path << "': " << status.AsString();
      return "";
    }

    if (custom_config_exists) {
      return custom_config_path;
    }
  }
  // "--model-config-name" is not set or custom config file does not exist.
  return JoinPath({model_dir_path, kModelConfigPbTxt});
}

Status
CreateAgentModelListWithLoadAction(
    const inference::ModelConfig& original_model_config,
    const std::string& original_model_path,
    const std::string& model_config_name,
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
        const auto config_path =
            GetModelConfigFullPath(location, model_config_name);
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

// Return the latest modification time in ns for all files/folders starting
// from the path. The modification time will be 0 if there is an error.
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
// Return the latest modification time in ns for '<config.pbtxt, model files>'
// in a model directory path. The time for "config.pbtxt" will be 0 if not
// found at "[model_dir_path]/config.pbtxt" or "[model_dir_path]/configs/
// <custom-config-name>.pbtxt" if "--model-config-name" is set. The time for
// "model files" includes the time for 'model_dir_path'.
std::pair<int64_t, int64_t>
GetDetailedModifiedTime(
    const std::string& model_dir_path, const std::string& model_config_path)
{
  // Check if 'model_dir_path' is a directory.
  bool is_dir;
  Status status = IsDirectory(model_dir_path, &is_dir);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << model_dir_path
              << "': " << status.AsString();
    return std::make_pair(0, 0);
  }
  if (!is_dir) {
    LOG_ERROR << "Failed to determine modification time for '" << model_dir_path
              << "': Model directory path is not a directory";
    return std::make_pair(0, 0);
  }

  std::pair<int64_t, int64_t> mtime(0, 0);  // <config.pbtxt, model files>

  // Populate 'model files' time to the model directory time
  status = FileModificationTime(model_dir_path, &mtime.second);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << model_dir_path
              << "': " << status.AsString();
    return std::make_pair(0, 0);
  }

  // Get files/folders in model_dir_path.
  std::set<std::string> contents;
  status = GetDirectoryContents(model_dir_path, &contents);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << model_dir_path
              << "': " << status.AsString();
    return std::make_pair(0, 0);
  }
  // Get latest modification time for each files/folders, and place it at the
  // correct category.
  for (const auto& child : contents) {
    const auto full_path = JoinPath({model_dir_path, child});
    if (full_path == model_config_path) {
      // config.pbtxt or customized config file in configs folder
      mtime.first = GetModifiedTime(full_path);
    } else {
      // model files
      mtime.second = std::max(mtime.second, GetModifiedTime(full_path));
    }
  }

  return mtime;
}
// Return true if any file in the 'model_dir_path' has been modified more
// recently than 'last_ns', which represents last modified time for
// '<config.pbtxt, model files>'. Update the 'last_ns' to the most-recent
// modified time.
bool
IsModified(
    const std::string& model_dir_path, const std::string& model_config_path,
    std::pair<int64_t, int64_t>* last_ns)
{
  auto new_ns = GetDetailedModifiedTime(model_dir_path, model_config_path);
  bool modified = std::max(new_ns.first, new_ns.second) >
                  std::max(last_ns->first, last_ns->second);
  last_ns->swap(new_ns);
  return modified;
}

}  // namespace

ModelRepositoryManager::ModelRepositoryManager(
    const std::set<std::string>& repository_paths, const bool autofill,
    const std::string& model_config_name, const bool polling_enabled,
    const bool model_control_enabled, const double min_compute_capability,
    const bool enable_model_namespacing,
    std::unique_ptr<ModelLifeCycle> life_cycle)
    : autofill_(autofill), model_config_name_(model_config_name),
      polling_enabled_(polling_enabled),
      model_control_enabled_(model_control_enabled),
      min_compute_capability_(min_compute_capability),
      dependency_graph_(&global_map_),
      enable_model_namespacing_(enable_model_namespacing),
      repository_paths_(repository_paths),
      model_life_cycle_(std::move(life_cycle))
{
  if (enable_model_namespacing_) {
    find_identifier_fn_ = [this](const std::string& n, ModelIdentifier* i) {
      return FindModelIdentifier(n, i);
    };
  } else {
    find_identifier_fn_ = [this](const std::string& n, ModelIdentifier* i) {
      return Status::Success;
    };
  }
}

ModelRepositoryManager::~ModelRepositoryManager() {}

Status
ModelRepositoryManager::Create(
    InferenceServer* server, const std::string& server_version,
    const std::set<std::string>& repository_paths,
    const std::set<std::string>& startup_models, const bool strict_model_config,
    const std::string& model_config_name, const bool polling_enabled,
    const bool model_control_enabled,
    const ModelLifeCycleOptions& life_cycle_options,
    const bool enable_model_namespacing,
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
          repository_paths, !strict_model_config, model_config_name,
          polling_enabled, model_control_enabled,
          life_cycle_options.min_compute_capability, enable_model_namespacing,
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
  std::lock_guard<std::mutex> lock(mu_);

  std::set<ModelIdentifier> added, deleted, modified, unmodified;

  // We don't modify 'infos_' in place to minimize how long we need to
  // hold the lock and also prevent any partial changes to do an error
  // during processing.
  ModelInfoMap new_infos;

  // Each subdirectory of repository path is a model directory from
  // which we read the model configuration.
  std::unordered_map<std::string, std::vector<const InferenceParameter*>>
      model_names;
  RETURN_IF_ERROR(Poll(
      model_names, &added, &deleted, &modified, &unmodified, &new_infos,
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

  infos_.Swap(new_infos);

  dependency_graph_.UpdateGraph(infos_, added, deleted, modified);

  for (const auto& name : deleted) {
    model_life_cycle_->AsyncUnload(name);
  }

  // model loading / unloading error will be printed but ignored
  LoadModelByDependency(&dependency_graph_, &infos_);

  return Status::Success;
}

std::map<ModelIdentifier, Status>
ModelRepositoryManager::LoadModelByDependency(
    DependencyGraph* dependency_graph, ModelInfoMap* infos)
{
  // [FIXME] This function involves iterative interaction between
  // ModelLifeCycle object and DependencyGraph object, should better
  // encapsulate the interaction:
  // Each iteration:
  //  - Check dependency graph for nodes that are ready for lifecycle changes:
  //      - load if all dependencies are satisfied and the node is 'healthy'
  //      - unload otherwise (should revisit this, logically will only happen in
  //        ensemble, the ensemble is requested to be re-loaded, at this point
  //        it is too late to revert model changes so the ensemble will not be
  //        available afterwards. In other words, ensemble is not covered in
  //        the model availability claim: if re-load results in model not
  //        available, the model state will be reverted)
  //  - Apply changes to lifecycle
  //  - Reflect the lifecycle change result to the dependency graph so that the
  //    downstream nodes may be ready for lifecycle changes.
  // Repeat until no more changes can be made.
  std::map<ModelIdentifier, Status> res;
  struct ModelState {
    ModelState(DependencyNode* node) : node_(node), status_(Status::Success) {}
    DependencyNode* node_;
    Status status_;
    std::promise<void> ready_;
  };
  NodeSet loaded_models;
  auto set_pair = ModelsToLoadUnload(loaded_models, res, dependency_graph);
  // Loop until all model are loaded / unloaded
  while ((!set_pair.first.empty()) || (!set_pair.second.empty())) {
    loaded_models.clear();
    // Unload invalid models first
    for (auto& invalid_model : set_pair.second) {
      model_life_cycle_->AsyncUnload(invalid_model->model_id_);
      res[invalid_model->model_id_] = invalid_model->status_;
      LOG_ERROR << invalid_model->status_.AsString();
      invalid_model->loaded_versions_ = std::set<int64_t>();
      loaded_models.emplace(invalid_model);
    }
    // load valid models and wait for load results
    std::vector<std::unique_ptr<ModelState>> model_states;
    for (auto& valid_model : set_pair.first) {
      model_states.emplace_back(new ModelState(valid_model));
      auto model_state = model_states.back().get();
      const auto itr = infos->Find(valid_model->model_id_);
      auto status = model_life_cycle_->AsyncLoad(
          valid_model->model_id_, itr->second->model_path_,
          valid_model->model_config_, itr->second->is_config_provided_,
          itr->second->mtime_nsec_.second > itr->second->prev_mtime_ns_.second,
          itr->second->agent_model_list_, [model_state](Status load_status) {
            model_state->status_ = load_status;
            model_state->ready_.set_value();
          });
      if (!status.IsOk()) {
        model_state->status_ = status;
        model_state->ready_.set_value();
        LOG_ERROR << "failed to load model '" << valid_model->model_id_.str()
                  << "': " << status.Message();
      }
      loaded_models.emplace(valid_model);
    }
    for (auto& model_state : model_states) {
      model_state->ready_.get_future().wait();
      res[model_state->node_->model_id_] = model_state->status_;
      const auto version_state =
          model_life_cycle_->VersionStates(model_state->node_->model_id_);
      model_state->node_->loaded_versions_.clear();
      for (const auto& vs : version_state) {
        if (vs.second.first == ModelReadyState::READY) {
          model_state->node_->loaded_versions_.emplace(vs.first);
        }
      }
      // If the model failed to load, should revert the timestamp to
      // ensure the next load request will attempt to load the model again
      // for operation idempotence. See comment on 'infos_'
      if (!model_state->status_.IsOk()) {
        auto& model_info = infos->Find(model_state->node_->model_id_)->second;
        model_info->mtime_nsec_ = model_info->prev_mtime_ns_;
      }
    }
    set_pair = ModelsToLoadUnload(loaded_models, res, dependency_graph);
  }
  // Clear temporary stored agent model list after all loads are triggered
  for (auto& info : *infos) {
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

  const auto& model_name = models.begin()->first;

  // Need ModelIdentifier to retrieve model state in lifecycle object,
  // which will not be available after graph update. So make a copy first
  // to check if unload works as intended.
  std::set<ModelIdentifier> to_be_unloaded_ids;
  if (type == ActionType::UNLOAD) {
    std::lock_guard<std::mutex> lock(mu_);  // guard global_map_
    const auto& git = global_map_.find(model_name);
    if (git != global_map_.end()) {
      to_be_unloaded_ids = git->second;
    }
  }

  bool polled = true;
  bool no_parallel_conflict = true;
  do {
    // Function will block during a conflict, so safe to retry immediately.
    RETURN_IF_ERROR(LoadUnloadModels(
        models, type, unload_dependents, &polled, &no_parallel_conflict));
  } while (!no_parallel_conflict);
  // Check if model is loaded / unloaded properly
  if (!polled) {
    return Status(
        Status::Code::INTERNAL, "failed to load '" + model_name +
                                    "', failed to poll from model repository");
  }

  // Use global map to retrieve the set of models being updated with the given
  // name
  if (type == ActionType::LOAD) {
    std::lock_guard<std::mutex> lock(mu_);  // guard global_map_ and infos_
    const auto& git = global_map_.find(model_name);
    if (git == global_map_.end()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to load '" + model_name + "', unexpected miss in global map");
    }
    for (const auto& model_id : git->second) {
      const auto version_states = model_life_cycle_->VersionStates(model_id);
      if (version_states.empty()) {
        return Status(
            Status::Code::INTERNAL,
            "failed to load '" + model_name + "', no version is available");
      }
      auto it = infos_.Find(model_id);
      if (it == infos_.end()) {
        return Status(
            Status::Code::INTERNAL,
            "failed to load '" + model_name +
                "', failed to poll from model repository");
      }
    }
  } else {
    for (const auto& model_id : to_be_unloaded_ids) {
      std::string ready_version_str;
      const auto version_states = model_life_cycle_->VersionStates(model_id);
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
  }

  return Status::Success;
}

Status
ModelRepositoryManager::LoadUnloadModels(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models,
    const ActionType type, const bool unload_dependents,
    bool* all_models_polled, bool* no_parallel_conflict)
{
  *all_models_polled = true;
  if (no_parallel_conflict != nullptr) {
    *no_parallel_conflict = true;
  }

  std::unique_lock<std::mutex> lock(mu_);

  // Prepare new model infos related to file system accordingly, do not write
  // yet until committed to load/unload the models.
  // Also poll the added, deleted, modified, unmodified models.
  ModelInfoMap new_infos(infos_);
  std::set<ModelIdentifier> added, deleted, modified, unmodified;
  switch (type) {
    case ActionType::LOAD:
      RETURN_IF_ERROR(PollModels(
          models, &added, &deleted, &modified, &unmodified, &new_infos,
          all_models_polled));
      break;
    case ActionType::UNLOAD:
      for (const auto& model : models) {
        // Within the class, model is referred by ModelIdentifier, so it needs
        // to check global map to find identifier associated with the name.
        auto git = global_map_.find(model.first);
        if (git != global_map_.end()) {
          for (const auto& mid : git->second) {
            deleted.insert(mid);
          }
        }
      }
      break;
    default:
      return Status(Status::Code::INTERNAL, "Invalid action type");
  }
  std::set<ModelIdentifier> deleted_dependents;

  // Make a copy of the dependency graph for rollback if there is a collision
  // between the models to be loaded/unloaded by this thread with another
  // thread actively loading/unloading models. The global map referenced by the
  // dependency graph is managed by this object, which is the responsibility of
  // this object to make a copy.
  std::unordered_map<std::string, std::set<ModelIdentifier>> new_global_map(
      global_map_);
  DependencyGraph new_dependency_graph(dependency_graph_, &new_global_map);

  // Update the new dependency graph, and get models affected by the update.
  std::set<ModelIdentifier> affected_models = new_dependency_graph.UpdateGraph(
      new_infos, added, deleted, modified,
      unload_dependents ? &deleted_dependents : nullptr);
  affected_models.insert(unmodified.begin(), unmodified.end());

  // Check for collision in affected models and try to lock those models.
  std::shared_ptr<std::condition_variable> retry_notify_cv;
  auto conflict_model =
      new_dependency_graph.LockNodes(affected_models, &retry_notify_cv);
  if (conflict_model) {
    // A collision is found. Since info is only written to local copy, so it is
    // safe to rollback by simply returning the conflict information.
    LOG_VERBOSE(2) << "Load/Unload conflict '" << conflict_model->str() << "'";
    if (no_parallel_conflict != nullptr) {
      *no_parallel_conflict = false;
      retry_notify_cv->wait(lock);
      return Status::Success;
    }
    return Status(
        Status::Code::INTERNAL,
        "a related model '" + conflict_model->str() +
            "' to a load/unload request is currently loading or unloading");
  }

  // The models are in 'deleted' either when they are asked to be unloaded or
  // they are not found / are duplicated across all model repositories.
  // Explicitly remove them from new infos.
  for (const auto& name : (unload_dependents ? deleted_dependents : deleted)) {
    new_infos.Erase(name);
  }

  // At this point, it is committed to load/unload the models, so write the
  // local copy into this object. The state of the models cannot be modified by
  // another thread while they are locked. Thus, it is safe to release the
  // mutex while actively loading/unloading models, but the mutex must be
  // re-held when reading/updating any model info to avoid any corruption.
  infos_ = new_infos;
  global_map_ = new_global_map;
  dependency_graph_.Assign(new_dependency_graph, &global_map_);

  // Unlock the affected models.
  // This serve 2 purposes:
  // 1. The unlocked state will be writeback to the dependency graph owned by
  //    this object, after the load/unload is completed.
  // 2. It doubles as an indicator that all the locked models will be handled
  //    by another thread, and this thread must not attempt to load/unload
  //    those locked models.
  new_dependency_graph.UnlockNodes(affected_models);

  // Release the mutex for parallel loading/unloading
  lock.unlock();
  // Unload 'deleted' models.
  for (const auto& name : (unload_dependents ? deleted_dependents : deleted)) {
    model_life_cycle_->AsyncUnload(name);
  }
  // Load/Unload the models affected
  const auto& load_status =
      LoadModelByDependency(&new_dependency_graph, &new_infos);
  // Re-held the mutex after loading/unloading
  lock.lock();

  // Write updated info back to this object after model load/unload.
  infos_.Writeback(new_infos, affected_models);
  dependency_graph_.Writeback(new_dependency_graph, affected_models);

  // Check the load status of the requested models.
  if (type == ActionType::LOAD) {
    std::string load_error_message = "";
    for (const auto& model : models) {
      auto git = global_map_.find(model.first);
      if (git == global_map_.end()) {
        // skip the model name that is not found, requested name didn't pass
        // polling
        continue;
      }
      for (const auto& model_id : git->second) {
        auto it = load_status.find(model_id);
        // If 'model.first' not in load status, it means the (re-)load is not
        // necessary because there is no change in the model's directory
        if ((it != load_status.end()) && !it->second.IsOk()) {
          load_error_message +=
              ("load failed for model '" + model_id.str() +
               "': " + it->second.Message() + "\n");
        }
      }
    }
    if (!load_error_message.empty()) {
      return Status(Status::Code::INVALID_ARG, load_error_message);
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::PollModels(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models,
    std::set<ModelIdentifier>* added, std::set<ModelIdentifier>* deleted,
    std::set<ModelIdentifier>* modified, std::set<ModelIdentifier>* unmodified,
    ModelInfoMap* infos, bool* all_models_polled)
{
  *all_models_polled = true;

  std::set<std::string> checked_models;
  auto current_models = models;

  ModelInfoMap new_infos;
  while (!current_models.empty()) {
    bool polled = true;
    RETURN_IF_ERROR(Poll(
        current_models, added, deleted, modified, unmodified, &new_infos,
        &polled));
    *all_models_polled &= polled;

    // A polling chain is needed if the polled model is ensemble, so that all
    // (up-to-date) composing models are exposed to Triton.
    std::set<std::string> next_models;
    // The intention here is to iteratively expand the collection of models
    // to be polled for this load operation, if ensemble is involved, until
    // the collection is self-contained.
    // [FIXME] Every Poll() above add new entries in existing 'new_infos'
    // (and 'added' etc.), so iterating on the whole structure means
    // re-examination on certain entries.
    std::set<std::string> temp_checked;
    for (auto& info : new_infos) {
      // if the model is not "checked", it is model polled at current
      // iteration, check config for more models to be polled
      const bool checked =
          (checked_models.find(info.first.name_) != checked_models.end());
      if (!checked) {
        // Don't add the model name to 'checked_models' as there may be same
        // name model polled but with different namespace
        temp_checked.emplace(info.first.name_);
        const auto& config = info.second->model_config_;
        if (config.has_ensemble_scheduling()) {
          for (const auto& step : config.ensemble_scheduling().step()) {
            next_models.emplace(step.model_name());
          }
        }
      }
    }
    AddToSet(temp_checked, &checked_models);
    // trim the model if it has been checked
    current_models.clear();
    for (const auto& name : next_models) {
      const bool checked = (checked_models.find(name) != checked_models.end());
      if (!checked) {
        current_models[name];
      }
    }
  }

  // After all polls, only models in the initial set ('models') are
  // explicitly loaded.
  for (auto& info : new_infos) {
    info.second->explicitly_load_ =
        (models.find(info.first.name_) != models.end());
  }

  // Only update the infos when all validation is completed
  for (const auto& model_name : *added) {
    auto nitr = new_infos.Find(model_name);
    infos->Emplace(model_name, std::move(nitr->second));
  }
  for (const auto& model_name : *modified) {
    auto nitr = new_infos.Find(model_name);
    auto itr = infos->Find(model_name);
    // Also check the 'explicitly_load_' in previous snapshot, if the model
    // has been explicitly loaded, we shouldn't change its state as it may be
    // polled as composing model and flag is set to false.
    nitr->second->explicitly_load_ |= itr->second->explicitly_load_;
    itr->second = std::move(nitr->second);
  }

  return Status::Success;
}

Status
ModelRepositoryManager::UnloadAllModels()
{
  // Find all the models to be unloaded.
  std::unordered_map<std::string, std::vector<const InferenceParameter*>>
      models;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& pair : infos_) {
      // Only the model name is needed.
      models[pair.first.name_];
    }
  }
  // Unload all models found.
  bool polled;
  return LoadUnloadModels(
      models, ActionType::UNLOAD, true /* unload_dependents */, &polled,
      nullptr /* no_parallel_conflict */);
}

Status
ModelRepositoryManager::StopAllModels()
{
  return model_life_cycle_->StopAllModels();
}

const std::set<std::tuple<ModelIdentifier, int64_t, size_t>>
ModelRepositoryManager::InflightStatus()
{
  return model_life_cycle_->InflightStatus();
}

size_t
ModelRepositoryManager::BackgroundModelsSize()
{
  return model_life_cycle_->BackgroundModelsSize();
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
  ModelIdentifier model_id("", model_name);
  if (find_identifier_fn_(model_name, &model_id).IsOk()) {
    return model_life_cycle_->VersionStates(model_id);
  }
  return {};
}

Status
ModelRepositoryManager::ModelState(
    const std::string& model_name, const int64_t model_version,
    ModelReadyState* state)
{
  ModelIdentifier model_id("", model_name);
  RETURN_IF_ERROR(find_identifier_fn_(model_name, &model_id));
  return model_life_cycle_->ModelState(model_id, model_version, state);
}

Status
ModelRepositoryManager::RepositoryIndex(
    const bool ready_only, std::vector<ModelIndex>* index)
{
  std::set<ModelIdentifier> seen_models;
  std::set<ModelIdentifier> duplicate_models;
  for (const auto& repository_path : repository_paths_) {
    const std::string model_namespace =
        (enable_model_namespacing_ ? repository_path : "");
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
      auto model_id = ModelIdentifier(model_namespace, subdir);
      auto model_it = models_in_repo.find(subdir);
      if (model_it != models_in_repo.end()) {
        model_id.name_ = model_it->second;
      }

      if (seen_models.find(model_id) != seen_models.end()) {
        duplicate_models.insert(model_id);
      }

      seen_models.insert(model_id);
    }
  }

  ModelStateMap states = ModelStates();

  for (const auto& model_id : seen_models) {
    // If the same model appears in multiple repositories then show it
    // as unavailable since duplicate models are not allowed to load.
    if (duplicate_models.find(model_id) != duplicate_models.end()) {
      index->emplace_back(
          model_id, -1 /* version */, ModelReadyState::UNAVAILABLE,
          MODEL_READY_REASON_DUPLICATE);
      continue;
    }

    // If there is any version/state/reason associated with the model
    // then include that in the index.
    auto sitr = states.find(model_id);
    if (sitr == states.end()) {
      if (!ready_only) {
        index->emplace_back(model_id);
      }
    } else {
      for (const auto& pr : sitr->second) {
        if (!ready_only || (pr.second.first == ModelReadyState::READY)) {
          index->emplace_back(
              model_id, pr.first, pr.second.first, pr.second.second);
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
  ModelIdentifier model_id("", model_name);
  RETURN_IF_ERROR(find_identifier_fn_(model_name, &model_id));
  return GetModel(model_id, model_version, model);
}

Status
ModelRepositoryManager::GetModel(
    const ModelIdentifier& model_id, const int64_t model_version,
    std::shared_ptr<Model>* model)
{
  Status status = model_life_cycle_->GetModel(model_id, model_version, model);
  if (!status.IsOk()) {
    model->reset();
    status = Status(
        status.ErrorCode(), "Request for unknown model: " + status.Message());
  }
  return status;
}

Status
ModelRepositoryManager::FindModelIdentifier(
    const std::string& model_name, ModelIdentifier* model_id)
{
  auto git = global_map_.find(model_name);
  if (git == global_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "Request for unknown model: '" + model_name + "' is not found");
  }
  switch (git->second.size()) {
    case 0:
      return Status(
          Status::Code::NOT_FOUND, "Identifier of model '" + model_name +
                                       "' is not found in global map");
      break;
    case 1: {
      *model_id = *git->second.begin();
      break;
    }
    default:
      return Status(
          Status::Code::INVALID_ARG,
          "There are " + std::to_string(git->second.size()) +
              " identifiers of model '" + model_name +
              "' in global map, model namespace must be provided to resolve "
              "ambiguity.");
      break;
  }
  return Status::Success;
}

Status
ModelRepositoryManager::Poll(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models,
    std::set<ModelIdentifier>* added, std::set<ModelIdentifier>* deleted,
    std::set<ModelIdentifier>* modified, std::set<ModelIdentifier>* unmodified,
    ModelInfoMap* updated_infos, bool* all_models_polled)
{
  *all_models_polled = true;
  // empty path is the special case to indicate the model should be loaded
  // from override file content in 'models'.
  std::map<ModelIdentifier, std::string> model_to_path;

  // If no model is specified, poll all models in all model repositories.
  // Otherwise, only poll the specified models
  if (models.empty()) {
    std::set<ModelIdentifier> duplicated_models;
    for (const auto& repository_path : repository_paths_) {
      // collapse namespace based on whether it is enabled
      const std::string model_namespace =
          (enable_model_namespacing_ ? repository_path : "");

      // Model name mapping is not checked in direct polling, because
      // it is set when registering additional model repository which is limited
      // to explicit mode, so there is no intersection of the code paths.
      // [TODO] revisit the above statement.
      std::set<std::string> subdirs;
      Status status = GetDirectorySubdirs(repository_path, &subdirs);
      if (!status.IsOk()) {
        LOG_ERROR << "failed to poll model repository '" << repository_path
                  << "': " << status.Message();
        *all_models_polled = false;
      } else {
        for (const auto& subdir : subdirs) {
          const ModelIdentifier model_identifier(model_namespace, subdir);
          if (!model_to_path
                   .emplace(
                       model_identifier, JoinPath({repository_path, subdir}))
                   .second) {
            duplicated_models.insert(model_identifier);
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
        // [TODO] once namespace is exposed to external user, should allow and
        // check if the namespace is provided as well,.
        model_to_path.emplace(ModelIdentifier("", model.first), "");
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
          // collapse namespace based on whether it is enabled
          const std::string model_namespace =
              (enable_model_namespacing_ ? model_it->second.first : "");
          model_to_path.emplace(
              ModelIdentifier(model_namespace, model.first),
              model_it->second.second);
          exists = true;
        } else {
          LOG_ERROR << "mapped path '" << full_path
                    << "' does not exist for model '" << model.first << "'";
          exists = false;
        }
      } else {
        for (const auto& repository_path : repository_paths_) {
          bool exists_in_this_repo = false;
          const auto full_path = JoinPath({repository_path, model.first});
          Status status = FileExists(full_path, &exists_in_this_repo);
          if (!status.IsOk()) {
            LOG_ERROR << "failed to poll model repository '" << repository_path
                      << "' for model '" << model.first
                      << "': " << status.Message();
            *all_models_polled = false;
          } else if (exists_in_this_repo) {
            // [FIXME] Revisit model_mapping: The intention is to check if
            // the model name is mapped to something other than the folder name.
            // If so, we shouldn't consider the model is found in this repo.
            // If the intention is correct, the current check is not accurate,
            // it skips the repo as long as at least one model within is mapped.
            //
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

            // collapse namespace based on whether it is enabled
            const std::string model_namespace =
                (enable_model_namespacing_ ? repository_path : "");
            auto res = model_to_path.emplace(
                ModelIdentifier(model_namespace, model.first),
                JoinPath({repository_path, model.first}));
            if (res.second) {
              exists = true;
            } else {
              exists = false;
              model_to_path.erase(res.first);
              LOG_ERROR << "failed to poll model '" << model.first
                        << "': not unique across all model repositories";
              break;
            }
          }  // else: model doesn't exist in this repository_path
        }
      }
      // For an explicitly specified model that does not exist, we do not mark
      // it as deleted, we simply mark that we could not poll all models.
      if (!exists) {
        *all_models_polled = false;
        LOG_ERROR << "failed to poll model '" << model.first
                  << "': model not found in any model repository.";
      }
    }
  }

  // Poll each of the models. If error happens during polling the model,
  // its state will fallback to the state before the polling.
  for (const auto& pair : model_to_path) {
    std::unique_ptr<ModelInfo> model_info;
    const auto& model_name = pair.first.name_;
    // Load with parameters will be appiled to all models with the same
    // name (namespace can be different), unless namespace is specified
    // in the future.
    const auto& mit = models.find(model_name);
    static std::vector<const InferenceParameter*> empty_params;
    auto status = InitializeModelInfo(
        pair.first, pair.second,
        ((mit == models.end()) ? empty_params : mit->second), &model_info);

    const auto& iitr = infos_.Find(pair.first);
    const bool invalid_add = (!status.IsOk()) && (iitr == infos_.end());
    if (!invalid_add) {
      const auto& ret = updated_infos->Emplace(pair.first, nullptr);
      if (!ret.second) {
        return Status(
            Status::Code::ALREADY_EXISTS,
            "unexpected model info for model '" + pair.first.str() + "'");
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
    const ModelIdentifier& model_id, const std::string& path,
    const std::vector<const InferenceParameter*>& params,
    std::unique_ptr<ModelInfo>* info)
{
  std::unique_ptr<ModelInfo> linfo(new ModelInfo());
  linfo->model_path_ = path;

  bool unmodified = false;

  const auto iitr = infos_.Find(model_id);
  // Set 'prev_mtime_ns_' if there is existing ModelInfo
  if (iitr != infos_.end()) {
    linfo->prev_mtime_ns_ = iitr->second->mtime_nsec_;
  } else {
    linfo->prev_mtime_ns_ = std::make_pair(0, 0);
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
    linfo->mtime_nsec_ = std::make_pair(0, 0);
    linfo->model_path_ = location;
    linfo->model_config_path_ = JoinPath({location, kModelConfigPbTxt});
    linfo->agent_model_list_.reset(new TritonRepoAgentModelList());
    linfo->agent_model_list_->AddAgentModel(std::move(localize_agent_model));
  } else {
    linfo->model_config_path_ =
        GetModelConfigFullPath(linfo->model_path_, model_config_name_);
    // Model is not loaded.
    if (iitr == infos_.end()) {
      linfo->mtime_nsec_ = GetDetailedModifiedTime(
          linfo->model_path_, linfo->model_config_path_);
    } else {
      // Check the current timestamps to determine if model actually has been
      // modified
      linfo->mtime_nsec_ = linfo->prev_mtime_ns_;
      unmodified = !IsModified(
          linfo->model_path_, linfo->model_config_path_, &linfo->mtime_nsec_);
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
      auto time_since_epoch =
          std::chrono::system_clock::now().time_since_epoch();
      linfo->mtime_nsec_.first =
          std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch)
              .count();
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
    const auto config_path = linfo->model_config_path_;
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
        linfo->model_config_, linfo->model_path_, model_config_name_,
        &linfo->agent_model_list_));
    if (linfo->agent_model_list_ != nullptr) {
      // Get the latest repository path
      const char* location;
      TRITONREPOAGENT_ArtifactType artifact_type;
      RETURN_IF_ERROR(linfo->agent_model_list_->Back()->Location(
          &artifact_type, &location));
      auto latest_path = std::string(location);
      linfo->model_path_ = latest_path;
      // [TODO] should try to read the config again at the latest location?
    }
  }
  linfo->is_config_provided_ = parsed_config;

  // [FIXME] better document / interact with config, the actual config to be
  // associated with the model is not finalized and valid for read until now.
  // Repo agent / config overwrite may provide different configs than
  // the one in storage.

  // Try to automatically generate missing parts of the model
  // configuration (autofill) that don't require model detail
  RETURN_IF_ERROR(GetNormalizedModelConfig(
      model_id.name_, linfo->model_path_, min_compute_capability_,
      &linfo->model_config_));

  // Note that the model inputs and outputs are not validated until
  // the model model is initialized as they may not be auto-completed
  // until model is initialized.
  RETURN_IF_ERROR(
      ValidateModelConfig(linfo->model_config_, min_compute_capability_));
  if (!autofill_) {
    RETURN_IF_ERROR(ValidateModelIOConfig(linfo->model_config_));
  }

  // If the model is mapped, update its config name based on the
  // mapping.
  if (model_mappings_.find(model_id.name_) != model_mappings_.end()) {
    linfo->model_config_.set_name(model_id.name_);
  } else {
    // If there is no model mapping, make sure the name of the model
    // matches the name of the directory. This is a somewhat arbitrary
    // requirement but seems like good practice to require it of the user.
    // It also acts as a check to make sure we don't have two different
    // models with the same name.
    if (linfo->model_config_.name() != model_id.name_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected directory name '" + model_id.name_ + "' for model '" +
              linfo->model_config_.name() +
              "', directory name must equal model name");
    }
  }

  *info = std::move(linfo);
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
    std::lock_guard<std::mutex> lock(mu_);

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
    std::lock_guard<std::mutex> lock(mu_);
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

std::pair<ModelRepositoryManager::NodeSet, ModelRepositoryManager::NodeSet>
ModelRepositoryManager::ModelsToLoadUnload(
    const NodeSet& loaded_models,
    const std::map<ModelIdentifier, Status>& model_load_status,
    DependencyGraph* dependency_graph)
{
  // <valid model set, invalid model set>
  std::pair<NodeSet, NodeSet> res;
  // first call to this function
  if (loaded_models.empty()) {
    for (auto& pair : *(dependency_graph->MutableNodes())) {
      auto node = pair.second.get();
      // only care about nodes that are affected by the update
      // nodes is locked if and only if it is being updated by another thread
      if (!node->checked_ && !node->is_locked_) {
        if (CheckNode(node, model_load_status)) {
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
        // nodes is locked if and only if it is being updated by another thread
        if (!node->checked_ && !node->is_locked_) {
          if (CheckNode(node, model_load_status)) {
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
ModelRepositoryManager::CheckNode(
    DependencyNode* node,
    const std::map<ModelIdentifier, Status>& model_load_status)
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
            "ensemble '" + node->model_id_.str() + "' depends on '" +
                upstream.first->model_id_.str() +
                "' which is not valid. Model '" +
                upstream.first->model_id_.str() +
                "' failed with error: " + upstream.first->status_.Message());
      } else if (upstream.first->loaded_versions_.empty()) {
        auto model_load_stat =
            model_load_status.find(upstream.first->model_id_);
        if ((model_load_stat != model_load_status.end()) &&
            model_load_stat->second.IsOk()) {
          node->status_ = Status(
              Status::Code::INVALID_ARG, "ensemble '" + node->model_id_.str() +
                                             "' depends on '" +
                                             upstream.first->model_id_.str() +
                                             "' which has no loaded version.");
        } else {
          // [FIXME] See https://jirasw.nvidia.com/browse/DLIS-4459
          node->status_ = Status(
              Status::Code::INVALID_ARG,
              "ensemble '" + node->model_id_.str() + "' depends on '" +
                  upstream.first->model_id_.str() +
                  "' which has no loaded version. Model '" +
                  upstream.first->model_id_.str() +
                  "' loading failed with error: " +
                  model_load_stat->second.Message());
        }
      } else {
        for (const auto& required_version : upstream.second) {
          if (required_version == -1) {
            continue;
          }

          auto it = upstream.first->loaded_versions_.find(required_version);
          if (it == upstream.first->loaded_versions_.end()) {
            node->status_ = Status(
                Status::Code::INVALID_ARG,
                "ensemble '" + node->model_id_.str() + "' depends on '" +
                    upstream.first->model_id_.str() +
                    "' whose required version " +
                    std::to_string(required_version) + " is not loaded.");
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


std::set<ModelIdentifier>
ModelRepositoryManager::DependencyGraph::UpdateGraph(
    const ModelInfoMap& model_infos, const std::set<ModelIdentifier>& added,
    const std::set<ModelIdentifier>& deleted,
    const std::set<ModelIdentifier>& modified,
    std::set<ModelIdentifier>* deleted_dependents)
{
  // Update dependency graph in following steps:
  //   1. Apply all node changes.
  //   2. Connect the edges of all affected nodes.
  //   3. Connectivity check.

  // If the state of a node is changed, all its downstreams will be affected.
  // deleted, drop from dependency_graph, add to missing_nodes if downstreams is
  // not empty affected_nodes are all ensembles as only ensembles are depending
  // on other models
  std::set<ModelIdentifier> affected_nodes, deleted_nodes;
  const bool cascading_removal = (deleted_dependents != nullptr);
  std::tie(affected_nodes, deleted_nodes) =
      RemoveNodes(deleted, cascading_removal);
  if (cascading_removal) {
    deleted_dependents->swap(deleted_nodes);
  }
  AddToSet(UpdateNodes(modified, model_infos), &affected_nodes);
  AddToSet(AddNodes(added, model_infos), &affected_nodes);

  for (auto& node_id : affected_nodes) {
    ConnectDependencyGraph(node_id);
  }

  for (auto& node_id : affected_nodes) {
    CircularDependencyCheck(node_id);
  }

  // Add deleted nodes to affected nodes
  const auto& deleted_ref =
      cascading_removal ? *deleted_dependents : deleted_nodes;
  affected_nodes.insert(deleted_ref.begin(), deleted_ref.end());

  return affected_nodes;
}

std::pair<std::set<ModelIdentifier>, std::set<ModelIdentifier>>
ModelRepositoryManager::DependencyGraph::RemoveNodes(
    const std::set<ModelIdentifier>& nodes, const bool cascading_removal)
{
  std::set<ModelIdentifier> all_affected_nodes, all_removed_nodes;
  std::set<ModelIdentifier> curr_removal = nodes;
  while (!curr_removal.empty()) {
    std::set<ModelIdentifier> next_removal;
    for (const auto& model_id : curr_removal) {
      const auto affected_nodes = RemoveNode(model_id);

      // Check if the upstream should be removed as well,
      // a node should be removed if cascading removal is requested,
      // was not explicitly loaded and now doesn't have any downstreams.
      // For the concern that the node may then have downstreams from
      // 'added' / 'modified' nodes, it is not possible based on the situations
      // where dependency graph will be updated:
      //  - POLL/NONE : There can be additions and deletions within single
      //                operation, but all nodes are marked explicitly loaded.
      //  - EXPLICIT : Each operation can either be "load" or "unload", so there
      //               will not be bi-directional changes regarding downstreams.
      if (cascading_removal) {
        const auto& upstreams = affected_nodes.first;
        for (const auto& upstream : upstreams) {
          auto unode = FindNode(upstream, false /* allow_fuzzy_matching */);
          if (unode && (unode->downstreams_.empty()) &&
              (!unode->explicitly_load_)) {
            next_removal.emplace(upstream);
          }
        }
      }

      // The downstreams will need to be re-evaluated once the node
      // changes are in place
      const auto& downstreams = affected_nodes.second;
      // std::set::merge() can be used if move to >= C++17
      for (const auto& id : downstreams) {
        all_affected_nodes.emplace(id);
      }

      all_removed_nodes.emplace(model_id);
    }

    curr_removal.swap(next_removal);
  }
  // Exclude removed nodes from affected nodes.
  for (const auto& removed_node : all_removed_nodes) {
    all_affected_nodes.erase(removed_node);
  }
  return {std::move(all_affected_nodes), std::move(all_removed_nodes)};
}

std::set<ModelIdentifier>
ModelRepositoryManager::DependencyGraph::UpdateNodes(
    const std::set<ModelIdentifier>& nodes, const ModelInfoMap& model_infos)
{
  std::set<ModelIdentifier> updated_nodes;
  // modified, invalidate (uncheck) all downstreams
  for (const auto& model_id : nodes) {
    auto it = nodes_.find(model_id);
    if (it != nodes_.end()) {
      auto& node = it->second;
      UncheckDownstream(node->downstreams_);

      // remove all upstream reference, because the
      // config may be changed and should rebuild dependency
      for (auto& upstream : node->upstreams_) {
        upstream.first->DisconnectDownstream(node.get());
      }
      for (const auto& model_name : node->missing_upstreams_) {
        auto mit = missing_nodes_.find(model_name);
        mit->second.erase(it->first);
      }

      // Update model info stored in the node
      ModelInfo* info = model_infos.At(model_id).get();
      node->model_config_ = info->model_config_;
      node->explicitly_load_ = info->explicitly_load_;
      node->upstreams_.clear();
      node->checked_ = false;
      node->status_ = Status::Success;

      updated_nodes.emplace(model_id);
    }
  }
  return updated_nodes;
}

std::set<ModelIdentifier>
ModelRepositoryManager::DependencyGraph::AddNodes(
    const std::set<ModelIdentifier>& nodes, const ModelInfoMap& model_infos)
{
  std::set<ModelIdentifier> affected_nodes;
  // added, add to dependency_graph, if in missing_node, invalidate (uncheck)
  // and associate all downstreams, remove from missing_node
  for (const auto& model_id : nodes) {
    std::unique_ptr<DependencyNode> added_node(new DependencyNode(model_id));
    ModelInfo* info = model_infos.At(model_id).get();
    added_node->model_config_ = info->model_config_;
    added_node->explicitly_load_ = info->explicitly_load_;

    // Check if this model name is needed for some nodes, simply mark those
    // nodes affected to re-evaluate them later
    auto it = missing_nodes_.find(model_id.name_);
    if (it != missing_nodes_.end()) {
      for (const auto& dependent_node_id : it->second) {
        auto dependent_node =
            FindNode(dependent_node_id, false /* allow_fuzzy_matching */);
        if (dependent_node) {
          UncheckDownstream({dependent_node});
          affected_nodes.emplace(dependent_node_id);
        }
      }
    }

    // Add nodes to all reference
    affected_nodes.emplace(model_id);
    (*global_map_ptr_)[model_id.name_].emplace(model_id);
    nodes_.emplace(std::make_pair(model_id, std::move(added_node)));
  }
  return affected_nodes;
}

void
ModelRepositoryManager::DependencyGraph::ConnectDependencyGraph(
    const ModelIdentifier& node_id)
{
  // Check the node's model config to determine if it depends on other models
  // and if those models are present
  auto updated_node = FindNode(node_id, false /* allow_fuzzy_matching */);
  updated_node->upstreams_.clear();
  updated_node->missing_upstreams_.clear();
  updated_node->connected_ = true;
  if (updated_node->model_config_.has_ensemble_scheduling()) {
    auto mutable_steps =
        updated_node->model_config_.mutable_ensemble_scheduling()
            ->mutable_step();
    for (auto& step : *mutable_steps) {
      // Build model identifier to look for the composing model,
      // currently using the same namespace as the 'updated_node' for
      // preferred identifier.
      // [enhancement] May allow user-specified namespace once allowed in
      // ensemble config.
      const ModelIdentifier preferred_id(
          updated_node->model_id_.namespace_, step.model_name());
      auto node = FindNode(preferred_id, true /* allow_fuzzy_matching */);
      if (node) {
        // Set 'model_namespace' field in model config to allow direct model
        // lookup at later stage.
        step.set_model_namespace(node->model_id_.namespace_);

        // Add the current node to its upstream
        node->downstreams_.emplace(updated_node);

        // Add the upstream to the current node
        auto res = updated_node->upstreams_.emplace(
            node, std::set<int64_t>({step.model_version()}));
        // If map insertion doesn't happen, the same model is required in
        // different step, insert the version to existing required version set.
        if (!res.second) {
          res.first->second.insert(step.model_version());
        }
      } else {
        updated_node->connected_ = false;
        updated_node->missing_upstreams_.emplace(step.model_name());
      }

      // If no node is found or the found node is not exact matched,
      // just record that in 'missing_nodes_' so the current node will
      // be rechecked whenever a new candidate is added.
      if (!node || (node->model_id_ != preferred_id)) {
        missing_nodes_[preferred_id.name_].emplace(updated_node->model_id_);
      }

      // Only update status if there is no failure from earlier stages
      if (updated_node->status_.IsOk() && !updated_node->connected_) {
        std::string name_list;
        for (const auto& missing_name : updated_node->missing_upstreams_) {
          if (!name_list.empty()) {
            name_list += ", ";
          }
          name_list += missing_name;
        }
        updated_node->status_ = Status(
            Status::Code::INVALID_ARG,
            "ensemble " + updated_node->model_id_.str() +
                " contains models that are not available or ambiguous: " +
                name_list);
      }
    }
  }
}

std::pair<std::set<ModelIdentifier>, std::set<ModelIdentifier>>
ModelRepositoryManager::DependencyGraph::RemoveNode(
    const ModelIdentifier& model_id)
{
  auto it = nodes_.find(model_id);
  // no-op if not found: "node" has been removed
  if (it == nodes_.end()) {
    return {{}, {}};
  }

  std::set<ModelIdentifier> upstreams, downstreams;
  auto& node = it->second;
  // remove this node from its upstreams
  for (auto& upstream : node->upstreams_) {
    auto& upstream_node = upstream.first;
    upstream_node->DisconnectDownstream(node.get());
    upstreams.emplace(upstream_node->model_id_);
  }

  // remove this node from its downstreams
  UncheckDownstream(node->downstreams_);
  for (auto& downstream : node->downstreams_) {
    downstream->DisconnectUpstream(node.get());
    downstreams.emplace(downstream->model_id_);
  }

  // Drop the node from all references and only track it on removed nodes
  (*global_map_ptr_)[model_id.name_].erase(model_id);
  for (const auto& model_name : node->missing_upstreams_) {
    auto mit = missing_nodes_.find(model_name);
    mit->second.erase(model_id);
  }
  removed_nodes_.emplace(std::make_pair(model_id, std::move(node)));
  nodes_.erase(it);

  return {std::move(upstreams), std::move(downstreams)};
}

ModelRepositoryManager::DependencyNode*
ModelRepositoryManager::DependencyGraph::FindNode(
    const ModelIdentifier& model_id, const bool allow_fuzzy_matching) const
{
  const auto git = nodes_.find(model_id);
  if (git != nodes_.end()) {
    return git->second.get();
  } else if (allow_fuzzy_matching) {
    const auto gmit = global_map_ptr_->find(model_id.name_);
    if ((gmit != global_map_ptr_->end()) && (gmit->second.size() == 1)) {
      const auto git = nodes_.find(*gmit->second.begin());
      if (git != nodes_.end()) {
        return git->second.get();
      }
    }
  }
  return nullptr;
}

void
ModelRepositoryManager::DependencyGraph::UncheckDownstream(
    const std::set<DependencyNode*>& downstreams)
{
  for (auto& node : downstreams) {
    if (node->checked_) {
      node->checked_ = false;
      node->status_ = Status::Success;
      UncheckDownstream(node->downstreams_);
    }
  }
}

void
ModelRepositoryManager::DependencyGraph::CircularDependencyCheck(
    const ModelIdentifier& node_id)
{
  auto node = FindNode(node_id, false /* allow_fuzzy_matching */);
  if (node->status_.IsOk()) {
    node->status_ = CircularDependencyCheck(node, node);
  }
}

Status
ModelRepositoryManager::DependencyGraph::CircularDependencyCheck(
    DependencyNode* current_node, const DependencyNode* start_node)
{
  // Note that traverse order is towards upstream for the correct dependency
  // ordering.
  for (auto& upstream : current_node->upstreams_) {
    auto& upstream_node = upstream.first;
    if (upstream_node == start_node) {
      return Status(
          Status::Code::INVALID_ARG, "circular dependency between ensembles: " +
                                         start_node->model_id_.str() +
                                         " -> ... -> " +
                                         current_node->model_id_.str() +
                                         " -> " + start_node->model_id_.str());
    } else {
      // Only return error when detect circular dependency, propagate the error
      // so all node in the chain will be set.
      const auto status = CircularDependencyCheck(upstream_node, start_node);
      if (!status.IsOk()) {
        current_node->status_ = status;
        return status;
      }
    }
  }
  return Status::Success;
}

ModelRepositoryManager::DependencyGraph::DependencyGraph(
    const DependencyGraph& rhs,
    std::unordered_map<std::string, std::set<ModelIdentifier>>* global_map)
    : global_map_ptr_(global_map), missing_nodes_(rhs.missing_nodes_)
{
  // Copy nodes_
  for (auto& pair : rhs.nodes_) {
    nodes_.emplace(pair.first, std::make_unique<DependencyNode>(*pair.second));
  }
  // Re-map pointers from rhs to this
  for (auto& pair : nodes_) {
    // upstreams_
    std::unordered_map<DependencyNode*, std::set<int64_t>> new_upstreams;
    for (auto& p : pair.second->upstreams_) {
      new_upstreams.emplace(nodes_.at(p.first->model_id_).get(), p.second);
    }
    pair.second->upstreams_.swap(new_upstreams);
    // downstreams_
    std::set<DependencyNode*> new_downstreams;
    for (auto* node : pair.second->downstreams_) {
      new_downstreams.emplace(nodes_.at(node->model_id_).get());
    }
    pair.second->downstreams_.swap(new_downstreams);
  }
  // Copy removed_nodes_
  for (auto& pair : rhs.removed_nodes_) {
    removed_nodes_.emplace(
        pair.first, std::make_unique<DependencyNode>(*pair.second));
  }
}

void
ModelRepositoryManager::DependencyGraph::Assign(
    const DependencyGraph& rhs,
    std::unordered_map<std::string, std::set<ModelIdentifier>>* global_map)
{
  DependencyGraph tmp(rhs, global_map);
  Swap(tmp);
}

void
ModelRepositoryManager::DependencyGraph::Swap(DependencyGraph& rhs)
{
  std::swap(global_map_ptr_, rhs.global_map_ptr_);
  nodes_.swap(rhs.nodes_);
  missing_nodes_.swap(rhs.missing_nodes_);
  removed_nodes_.swap(rhs.removed_nodes_);
}

ModelRepositoryManager::DependencyNode*
ModelRepositoryManager::DependencyGraph::GetNode(
    const ModelIdentifier& model_id) const
{
  const auto it = removed_nodes_.find(model_id);
  if (it != removed_nodes_.end()) {
    return it->second.get();
  }
  auto* node = FindNode(model_id, false /* allow_fuzzy_matching */);
  if (node == nullptr) {
    throw std::invalid_argument("model_id '" + model_id.str() + "' not found");
  }
  return node;
}

std::unique_ptr<ModelIdentifier>
ModelRepositoryManager::DependencyGraph::LockNodes(
    const std::set<ModelIdentifier>& nodes,
    std::shared_ptr<std::condition_variable>* retry_notify_cv)
{
  for (const auto& model_id : nodes) {
    auto* node = GetNode(model_id);
    if (node->is_locked_) {
      if (retry_notify_cv != nullptr) {
        *retry_notify_cv = node->retry_notify_cv_;
      }
      return std::make_unique<ModelIdentifier>(model_id);
    }
    node->is_locked_ = true;
  }
  return std::unique_ptr<ModelIdentifier>(nullptr);
}

std::unique_ptr<ModelIdentifier>
ModelRepositoryManager::DependencyGraph::UnlockNodes(
    const std::set<ModelIdentifier>& nodes)
{
  for (const auto& model_id : nodes) {
    auto* node = GetNode(model_id);
    if (!node->is_locked_) {
      return std::make_unique<ModelIdentifier>(model_id);
    }
    node->is_locked_ = false;
  }
  return std::unique_ptr<ModelIdentifier>(nullptr);
}

void
ModelRepositoryManager::DependencyGraph::Writeback(
    const DependencyGraph& updated_dependency_graph,
    const std::set<ModelIdentifier>& affected_models)
{
  for (const auto& model_id : affected_models) {
    auto* node = GetNode(model_id);
    auto* updated_node = updated_dependency_graph.GetNode(model_id);
    // Writeback
    node->Writeback(*updated_node);
    // Erase removed nodes at last to complete its lifecycle
    removed_nodes_.erase(model_id);
  }
}


void
ModelRepositoryManager::DependencyNode::Writeback(
    const DependencyNode& updated_dependency_node)
{
  // Writeback
  status_ = updated_dependency_node.status_;
  checked_ = updated_dependency_node.checked_;
  loaded_versions_ = updated_dependency_node.loaded_versions_;
  is_locked_ = updated_dependency_node.is_locked_;
  // Notify retry(s)
  retry_notify_cv_->notify_all();
}


ModelRepositoryManager::ModelInfoMap::ModelInfoMap(const ModelInfoMap& rhs)
{
  for (auto& pair : rhs.map_) {
    map_.emplace(pair.first, std::make_unique<ModelInfo>(*pair.second));
  }
}

ModelRepositoryManager::ModelInfoMap&
ModelRepositoryManager::ModelInfoMap::operator=(const ModelInfoMap& rhs)
{
  ModelInfoMap tmp(rhs);
  Swap(tmp);
  return *this;
}

void
ModelRepositoryManager::ModelInfoMap::Writeback(
    const ModelInfoMap& updated_model_info,
    const std::set<ModelIdentifier>& affected_models)
{
  for (auto& model_id : affected_models) {
    // An affected model can be deleted, so the presence of the model must be
    // checked.
    auto itr = updated_model_info.map_.find(model_id);
    if (itr != updated_model_info.map_.end()) {
      auto* updated_info = itr->second.get();
      auto* info = map_.at(model_id).get();
      // Writeback
      info->mtime_nsec_ = updated_info->mtime_nsec_;
    }
  }
}

}}  // namespace triton::core
