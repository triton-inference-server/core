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
#pragma once

#include <functional>
#include <map>
#include <mutex>
#include "infer_parameter.h"
#include "model_config.pb.h"
#include "repo_agent.h"
#include "status.h"
#include "triton/common/model_config.h"
#include "triton/common/thread_pool.h"

namespace triton { namespace core {

// [FIXME] shouldn't need this
enum ActionType { NO_ACTION, LOAD, UNLOAD };

/// Readiness status for models.
enum class ModelReadyState {
  // The model is in an unknown state. The model is not available for
  // inferencing.
  UNKNOWN,

  // The model is ready and available for inferencing.
  READY,

  // The model is unavailable, indicating that the model failed to
  // load or has been implicitly or explicitly unloaded. The model is
  // not available for inferencing.
  UNAVAILABLE,

  // The model is being loaded by the inference server. The model is
  // not available for inferencing.
  LOADING,

  // The model is being unloaded by the inference server. The model is
  // not available for inferencing.
  UNLOADING
};

/// Get the string representation for a ModelReadyState
const std::string& ModelReadyStateString(ModelReadyState state);

using VersionStateMap =
    std::map<int64_t, std::pair<ModelReadyState, std::string>>;
using ModelStateMap = std::map<std::string, VersionStateMap>;

// Helper class to manage the lifecycle of a list of associated agent models
class TritonRepoAgentModelList {
 public:
  TritonRepoAgentModelList()
      : last_action_type_(TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE){};
  ~TritonRepoAgentModelList()
  {
    // [WIP] Use destructor to make sure the unload lifecycle is complete
    // without explicitly managing the last step in ModelLifecycle.
    if (last_action_type_ == TRITONREPOAGENT_ACTION_UNLOAD) {
      InvokeAgentModels(TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE);
    }
  }
  Status AddAgentModel(std::unique_ptr<TritonRepoAgentModel>&& agent_model)
  {
    agent_models_.emplace_back(std::move(agent_model));
    return Status::Success;
  }

  size_t Size() { return agent_models_.size(); }

  TritonRepoAgentModel* Back() { return agent_models_.back().get(); }

  TRITONREPOAGENT_ActionType LastActionType() { return last_action_type_; }

  Status InvokeAgentModels(const TRITONREPOAGENT_ActionType action_type)
  {
    last_action_type_ = action_type;
    switch (action_type) {
      case TRITONREPOAGENT_ACTION_LOAD:
      case TRITONREPOAGENT_ACTION_UNLOAD: {
        for (size_t idx = 0; idx < agent_models_.size(); ++idx) {
          RETURN_IF_ERROR(agent_models_[idx]->InvokeAgent(action_type));
        }
        break;
      }
      case TRITONREPOAGENT_ACTION_LOAD_COMPLETE:
      case TRITONREPOAGENT_ACTION_LOAD_FAIL:
      case TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE: {
        // reverse order
        for (size_t one_pass_idx = agent_models_.size(); one_pass_idx > 0;
             --one_pass_idx) {
          RETURN_IF_ERROR(
              agent_models_[one_pass_idx - 1]->InvokeAgent(action_type));
        }
        break;
      }
    }
    return Status::Success;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonRepoAgentModelList);

  std::vector<std::unique_ptr<TritonRepoAgentModel>> agent_models_;
  TRITONREPOAGENT_ActionType last_action_type_;
};

class InferenceServer;
class Model;

class ModelLifeCycle {
 public:
  static Status Create(
      InferenceServer* server, const double min_compute_capability,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const unsigned int model_load_thread_count,
      std::unique_ptr<ModelLifeCycle>* life_cycle);

  ~ModelLifeCycle()
  {
    // Explicitly clean up thread pool first to clean up any pending callbacks
    // that may modify model lifecycle members
    load_pool_.reset();
    map_.clear();
  }

  // Start loading model with specified versions asynchronously.
  // All versions that are being served will be unloaded only after
  // the load is finished sucessfully.
  Status AsyncLoad(
      const std::string& model_name, const std::string& model_path,
      const inference::ModelConfig& model_config,
      const std::shared_ptr<TritonRepoAgentModelList>& agent_model_list,
      std::function<void(Status)>&& OnComplete);

  // Unload model asynchronously.
  Status AsyncUnload(const std::string& model_name);

  // Get specified version of the model. Latest ready version will
  // be retrieved if 'version' is -1. Return error if the version specified is
  // not found or it is not ready.
  Status GetModel(
      const std::string& model_name, const int64_t version,
      std::shared_ptr<Model>* model);

  // Get the ModelStateMap representation of the live models. A model is
  // live if at least one of the versions is not unknown nor unavailable.
  // If 'strict_readiness' is true, a model is only live if
  // at least one of the versions is ready.
  const ModelStateMap LiveModelStates(bool strict_readiness = false);

  // Get the ModelStateMap representation of the models.
  const ModelStateMap ModelStates();

  // Get the VersionStateMap representation of the specified model.
  const VersionStateMap VersionStates(const std::string& model_name);

  // Get the state of a specific model version.
  Status ModelState(
      const std::string& model_name, const int64_t model_version,
      ModelReadyState* state);

  // Instruct the model to stop accepting new inference requests.
  Status StopAllModels();

  // Return the number of in-flight inference if any, model versions
  // that don't have in-flight inferences will not be included.
  const std::set<std::tuple<std::string, int64_t, size_t>> InflightStatus();

 private:
  struct ModelInfo {
    ModelInfo(
        const std::string& model_path,
        const inference::ModelConfig& model_config,
        const uint64_t latest_update_ns)
        : model_config_(model_config), model_path_(model_path),
          is_ensemble_(false), latest_update_ns_(latest_update_ns),
          state_(ModelReadyState::UNKNOWN)
    {
#ifdef TRITON_ENABLE_ENSEMBLE
      is_ensemble_ = (model_config.platform() == kEnsemblePlatform);
#endif  // TRITON_ENABLE_ENSEMBLE
    }

    inference::ModelConfig model_config_;
    std::string model_path_;
    bool is_ensemble_;

    std::recursive_mutex mtx_;
    uint64_t latest_update_ns_;
    ModelReadyState state_;
    std::string state_reason_;

    // callback function that will be triggered when there is no next action
    std::function<void()> OnComplete_;

    // [FIXME] better way to manage the lifecycle (unload / unload complete)
    std::shared_ptr<TritonRepoAgentModelList> agent_model_list_;
    std::shared_ptr<Model> model_;
  };

  struct LoadTracker {
    LoadTracker(
        const size_t affected_version_cnt, const uint64_t latest_update_ns)
        : load_failed_(false), completed_version_cnt_(0),
          affected_version_cnt_(affected_version_cnt),
          latest_update_ns_(latest_update_ns)
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
    uint64_t latest_update_ns_;

    // The set of model versions to be unloaded after the load is completed
    // std::set<int64_t> defer_unload_set_;
  };

  ModelLifeCycle(
      const double min_compute_capability, InferenceServer* server,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const unsigned int model_load_thread_count)
      : min_compute_capability_(min_compute_capability), server_(server),
        cmdline_config_map_(backend_cmdline_config_map),
        host_policy_map_(host_policy_map)
  {
    load_pool_.reset(
        new triton::common::ThreadPool(std::max(1u, model_load_thread_count)));
  }

  // Function called after model state / next action is updated.
  // Caller must obtain the mutex of 'model_info' before calling this function
  Status TriggerNextAction(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info);

  // Helper function called by TriggerNextAction()
  Status Load(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info);

  // Helper function called by TriggerNextAction()
  Status Unload(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info);

  Status CreateModel(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info);

  // [WIP] replace existing functions
  // [FIXME] lock?
  // Load is easy, always operating on fresh ModelInfo
  Status NewLoad(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info);
  void NewCreateModel(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info);
  // Callback function template for model load.
  // 'OnComplete' needs to be passed by value for now as there can be
  // multiple versions to be loaded and each holds a copy of
  // the 'OnComplete' callback.
  void OnLoadComplete(
      const std::string& model_name, const int64_t version,
      ModelInfo* model_info, std::function<void(Status)> OnComplete,
      std::shared_ptr<LoadTracker> load_tracker);

  const double min_compute_capability_;

  using VersionMap = std::map<int64_t, std::unique_ptr<ModelInfo>>;
  using ModelMap = std::map<std::string, VersionMap>;
  ModelMap map_;

  // Models that are being loaded / unloaded (?) in background
  std::map<uintptr_t, std::unique_ptr<ModelInfo>> background_models_;
  std::recursive_mutex map_mtx_;

  InferenceServer* server_;
  const triton::common::BackendCmdlineConfigMap cmdline_config_map_;
  const triton::common::HostPolicyCmdlineConfigMap host_policy_map_;

  // Fixed-size thread pool to load models at specified concurrency
  std::unique_ptr<triton::common::ThreadPool> load_pool_;
};

}}  // namespace triton::core
