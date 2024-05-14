// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <map>
#include <mutex>

#include "infer_parameter.h"
#include "model.h"
#include "model_config.pb.h"
#include "repo_agent.h"
#include "status.h"
#include "triton/common/model_config.h"
#include "triton/common/thread_pool.h"

namespace triton { namespace core {
struct ModelLifeCycleOptions {
  explicit ModelLifeCycleOptions(
      const double min_compute_capability,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const unsigned int model_load_thread_count, const size_t load_retry)
      : min_compute_capability(min_compute_capability),
        backend_cmdline_config_map(backend_cmdline_config_map),
        host_policy_map(host_policy_map),
        model_load_thread_count(model_load_thread_count), load_retry(load_retry)
  {
  }
  // The minimum supported CUDA compute capability.
  const double min_compute_capability;
  // The backend configuration settings specified on the command-line
  const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map;
  // The host policy setting used when loading models.
  const triton::common::HostPolicyCmdlineConfigMap& host_policy_map;
  // Number of the threads to use for concurrently loading models
  const unsigned int model_load_thread_count;
  // Number of retry on model loading before considering the load has failed.
  const size_t load_retry{0};
};


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
using ModelStateMap = std::map<ModelIdentifier, VersionStateMap>;

// Helper class to manage the lifecycle of a list of associated agent models
class TritonRepoAgentModelList {
 public:
  TritonRepoAgentModelList()
      : last_action_type_(TRITONREPOAGENT_ACTION_UNLOAD_COMPLETE){};
  ~TritonRepoAgentModelList()
  {
    // Using destructor to finish the unload lifecycle without
    // explicitly managing the last step in ModelLifecycle.
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

  Status InvokeAgentModels(const TRITONREPOAGENT_ActionType action_type)
  {
    // Special handling for the current model lifecycle implementation,
    // the repo agent may be asked to perform UNLOAD action multiple times,
    // and the requests after the first should be ignored.
    const bool duplicate_unload =
        (action_type == TRITONREPOAGENT_ACTION_UNLOAD) &&
        (last_action_type_ == TRITONREPOAGENT_ACTION_UNLOAD);
    if (duplicate_unload) {
      return Status::Success;
    }

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

class ModelLifeCycle {
 public:
  static Status Create(
      InferenceServer* server, const ModelLifeCycleOptions& options,
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
  // the load is finished successfully.
  Status AsyncLoad(
      const ModelIdentifier& model_id, const std::string& model_path,
      const inference::ModelConfig& model_config, const bool is_config_provided,
      const bool is_model_file_updated,
      const std::shared_ptr<TritonRepoAgentModelList>& agent_model_list,
      std::function<void(Status)>&& OnComplete);

  // Unload model asynchronously.
  Status AsyncUnload(const ModelIdentifier& model_id);

  // Get specified version of the model. Latest ready version will
  // be retrieved if 'version' is -1. Return error if the version specified is
  // not found or it is not ready.
  Status GetModel(
      const ModelIdentifier& model_id, const int64_t version,
      std::shared_ptr<Model>* model);

  // Get the ModelStateMap representation of the live models. A model is
  // live if at least one of the versions is not unknown nor unavailable.
  // If 'strict_readiness' is true, a model is only live if
  // at least one of the versions is ready.
  const ModelStateMap LiveModelStates(bool strict_readiness = false);

  // Get the ModelStateMap representation of the models.
  const ModelStateMap ModelStates();

  // Get the VersionStateMap representation of the specified model.
  const VersionStateMap VersionStates(const ModelIdentifier& model_id);

  // Get the state of a specific model version.
  Status ModelState(
      const ModelIdentifier& model_id, const int64_t model_version,
      ModelReadyState* state);

  // Instruct the model to stop accepting new inference requests.
  Status StopAllModels();

  // Return the number of in-flight inference if any, model versions
  // that don't have in-flight inferences will not be included.
  const std::set<std::tuple<ModelIdentifier, int64_t, size_t>> InflightStatus();

  // Return the number of model(s) in the background.
  size_t BackgroundModelsSize();

 private:
  struct ModelInfo {
    ModelInfo(
        const std::string& model_path,
        const inference::ModelConfig& model_config,
        const uint64_t last_update_ns)
        : model_config_(model_config), model_path_(model_path),
#ifdef TRITON_ENABLE_ENSEMBLE
          is_ensemble_(model_config.platform() == kEnsemblePlatform),
#else
          is_ensemble_(false),
#endif  // TRITON_ENABLE_ENSEMBLE
          last_update_ns_(last_update_ns), state_(ModelReadyState::UNKNOWN)
    {
    }

    // Release the flyweight in ModelInfo object, reflect as 'UNLOADING' in
    // model state. Note that 'mtx_' should be acquired before invoking this
    // function to prevent possible data race.
    void Release()
    {
      state_ = ModelReadyState::UNLOADING;
      state_reason_.clear();
      agent_model_list_.reset();
      model_.reset();
    }

    inference::ModelConfig model_config_;
    const std::string model_path_;
    const bool is_ensemble_;

    std::mutex mtx_;

    uint64_t last_update_ns_;

    ModelReadyState state_;
    std::string state_reason_;

    // flyweight
    std::shared_ptr<TritonRepoAgentModelList> agent_model_list_;
    std::shared_ptr<Model> model_;
  };

  struct LoadTracker {
    LoadTracker(
        const size_t affected_version_cnt, const uint64_t last_update_ns)
        : last_update_ns_(last_update_ns),
          affected_version_cnt_(affected_version_cnt), load_failed_(false),
          completed_version_cnt_(0)
    {
    }

    const uint64_t last_update_ns_;
    const size_t affected_version_cnt_;

    std::mutex mtx_;

    bool load_failed_;
    std::string reason_;
    size_t completed_version_cnt_;
    std::map<int64_t, ModelInfo*> load_set_;
  };

  ModelLifeCycle(InferenceServer* server, const ModelLifeCycleOptions& options)
      : server_(server), options_(options)
  {
    load_pool_.reset(new triton::common::ThreadPool(
        std::max(1u, options_.model_load_thread_count)));
  }

  // Create a new model, the 'model_id' can either be a new or existing model.
  void CreateModel(
      const ModelIdentifier& model_id, const int64_t version,
      ModelInfo* model_info, const bool is_config_provided);
  // Update model to the new config. It is the responsibility of the caller to
  // ensure the model can be updated in-place without a complete reload.
  // Currently, only model instances can be updated.
  void UpdateModelConfig(
      const ModelIdentifier& model_id, const int64_t version,
      ModelInfo* model_info, const inference::ModelConfig& new_model_config);
  // Update 'load_tracker' to the latest info in 'model_info' after loading
  // **each** model version.
  void OnLoadComplete(
      const ModelIdentifier& model_id, const int64_t version,
      ModelInfo* model_info, const bool is_update,
      const std::function<void(Status)>& OnComplete,
      std::shared_ptr<LoadTracker> load_tracker);
  // Helper function for 'OnLoadComplete()' to finish final operations after
  // loading **all** model versions.
  void OnLoadFinal(
      const ModelIdentifier& model_id, ModelInfo* model_info,
      const std::function<void(Status)>& OnComplete,
      std::shared_ptr<LoadTracker> load_tracker);


  // Mutex for 'map_' and 'background_models_'
  std::mutex map_mtx_;

  using VersionMap = std::map<int64_t, std::unique_ptr<ModelInfo>>;
  using ModelMap = std::map<ModelIdentifier, VersionMap>;
  ModelMap map_;
  // Models that are being loaded / unloaded in background
  std::map<uintptr_t, std::unique_ptr<ModelInfo>> background_models_;

  InferenceServer* server_;
  const ModelLifeCycleOptions options_;

  // Fixed-size thread pool to load models at specified concurrency
  std::unique_ptr<triton::common::ThreadPool> load_pool_;
};

}}  // namespace triton::core
