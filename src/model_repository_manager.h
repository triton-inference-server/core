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
#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <set>
#include "infer_parameter.h"
#include "model_config.pb.h"
#include "model_lifecycle.h"
#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

class InferenceServer;
class Model;

// [FIXME] should have separated load / unload functions for clarity
enum ActionType { NO_ACTION, LOAD, UNLOAD };

/// Predefined reason strings
#define MODEL_READY_REASON_DUPLICATE "model appears in two or more repositories"

/// An object to manage the model repository active in the server.
class ModelRepositoryManager {
 public:
  // Index information for a model.
  struct ModelIndex {
    ModelIndex(const std::string& n)
        : name_only_(true), name_(n), version_(-1),
          state_(ModelReadyState::UNKNOWN)
    {
    }
    ModelIndex(
        const std::string& n, const int64_t v, const ModelReadyState s,
        const std::string& r)
        : name_only_(false), name_(n), version_(v), state_(s), reason_(r)
    {
    }
    const bool name_only_;
    const std::string name_;
    const int64_t version_;
    const ModelReadyState state_;
    const std::string reason_;
  };

  /// A basic unit in dependency graph that records the models seen by the model
  /// repository manager.
  struct DependencyNode {
    DependencyNode(const std::string& model_name)
        : model_name_(model_name), status_(Status::Success), checked_(false)
    {
    }

    std::string model_name_;
    Status status_;
    bool checked_;
    bool explicitly_load_;
    inference::ModelConfig model_config_;
    std::set<int64_t> loaded_versions_;
    std::set<DependencyNode*> missing_upstreams_;
    std::unordered_map<DependencyNode*, std::set<int64_t>> upstreams_;
    std::set<DependencyNode*> downstreams_;
  };

  ~ModelRepositoryManager();

  /// Create a manager for a repository.
  /// \param server The pointer to the inference server.
  /// \param server_version The version of the inference server.
  /// \param repository_paths A set of file-system paths of the repositories.
  /// \param startup_models A set of models to be loaded at startup
  /// if model control is enabled.
  /// \param strict_model_config If false attempt to autofill missing required
  /// information in each model configuration.
  /// \param polling_enabled If true, then PollAndUpdate() is allowed.
  /// Otherwise, it is not allowed.
  /// \param model_control_enabled If true, then LoadUnloadModel() is allowed
  /// and the models in the model repository will not be loaded at startup.
  /// Otherwise, LoadUnloadModel() is not allowed and the models will be loaded.
  /// Cannot be set to true if polling_enabled is true.
  /// \param life_cycle_options The options to configure ModelLifeCycle.
  /// \param model_repository_manager Return the model repository manager.
  /// \return The error status.
  static Status Create(
      InferenceServer* server, const std::string& server_version,
      const std::set<std::string>& repository_paths,
      const std::set<std::string>& startup_models,
      const bool strict_model_config, const bool polling_enabled,
      const bool model_control_enabled,
      const ModelLifeCycleOptions& life_cycle_options,
      std::unique_ptr<ModelRepositoryManager>* model_repository_manager);

  /// Poll the model repository to determine the new set of models and
  /// compare with the current set. And serve the new set of models based
  /// on their version policy.
  Status PollAndUpdate();

  /// Load or unload a specified model.
  /// \param models The models and the parameters to be loaded or unloaded
  /// \param type The type action to be performed. If the action is LOAD and
  /// the model has been loaded, the model will be re-loaded.
  /// \return error status. Return "NOT_FOUND" if it tries to load
  /// a non-existing model or if it tries to unload a model that hasn't been
  /// loaded.
  Status LoadUnloadModel(
      const std::unordered_map<
          std::string, std::vector<const InferenceParameter*>>& models,
      const ActionType type, const bool unload_dependents);

  /// Unload all models. This function should be called before shutting down
  /// the model repository manager.
  /// \return error status.
  Status UnloadAllModels();

  /// Instruct all models to stop accepting new inference requests. However,
  /// the models are still capable of processing inference requests
  /// if the model considers them as part of the in-flight inference.
  /// \return error status.
  Status StopAllModels();

  /// \return the number of in-flight inferences for the all versions of all
  /// models. The set element will be a tuple of <model_name, model_version,
  /// in-flight inference count>. Note that a model version will not be included
  /// if it doesn't have in-flight inferences.
  const std::set<std::tuple<std::string, int64_t, size_t>> InflightStatus();

  /// \param strict_readiness If true, only models that have at least one
  /// ready version will be considered as live. Otherwise, the models that
  /// have loading / unloading versions will also be live.
  /// \return the state of all versions of all live models.
  const ModelStateMap LiveModelStates(bool strict_readiness = false);

  /// \return the state of all versions of all models that have every
  /// been (attempted) loaded over the lifetime of the server.
  const ModelStateMap ModelStates();

  /// \return the states of all versions of a specific model.
  const VersionStateMap VersionStates(const std::string& model_name);

  /// \return the ready-state of a specific model version.
  Status ModelState(
      const std::string& model_name, const int64_t model_version,
      ModelReadyState* state);

  /// Get the index of all models in all repositories.
  /// \param ready_only If true return only index of models that are ready.
  /// \param index Returns the index.
  /// \return error status.
  Status RepositoryIndex(const bool ready_only, std::vector<ModelIndex>* index);

  /// Obtain the specified model.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model.
  /// \param model Return the model object.
  /// \return error status.
  Status GetModel(
      const std::string& model_name, const int64_t model_version,
      std::shared_ptr<Model>* model);

  // Register model repository path.
  /// \param repository Path to model repository.
  /// \param model_mapping Mapping with (overridden) model name as key, subdir
  /// name as value.
  /// \return error status
  Status RegisterModelRepository(
      const std::string& repository,
      const std::unordered_map<std::string, std::string>& model_mapping);

  // Unregister model repository path.
  /// \param repository Path to model repository.
  /// \return error status
  Status UnregisterModelRepository(const std::string& repository);

 private:
  struct ModelInfo;

  // Map from model name to information about the model.
  using ModelInfoMap =
      std::unordered_map<std::string, std::unique_ptr<ModelInfo>>;

  // Set of DependencyNode
  using NodeSet = std::set<DependencyNode*>;

  ModelRepositoryManager(
      const std::set<std::string>& repository_paths, const bool autofill,
      const bool polling_enabled, const bool model_control_enabled,
      const double min_compute_capability,
      std::unique_ptr<ModelLifeCycle> life_cycle);

  /// The internal function that are called in Create() and PollAndUpdate().
  Status PollAndUpdateInternal(bool* all_models_polled);

  /// The internal function that load or unload a set of models.
  Status LoadUnloadModels(
      const std::unordered_map<
          std::string, std::vector<const InferenceParameter*>>& models,
      const ActionType type, const bool unload_dependents,
      bool* all_models_polled);

  /// Poll the requested models in the model repository and
  /// compare with the current set. Return the additions, deletions,
  /// and modifications that have occurred. This function will not updated
  /// the current model info, it is caller's responsibility to do so.
  /// \param models The map from models to be polled to their associated
  /// parameters.
  /// \param added The names of the models added to the repository.
  /// \param deleted The names of the models removed from the repository.
  /// \param modified The names of the models remaining in the
  /// repository that have been changed.
  /// \param unmodified The names of the models remaining in the
  /// repository that have not changed.
  /// \param updated_infos The model infos retrieved from the poll.
  /// \param all_models_polled Return true if all models are polled and
  /// their model configuration are validated successfully. Instead of aborting
  /// the polling, the models that fail will be ignored and their model infos
  /// will stay in the previous state.
  /// \return The error status.
  Status Poll(
      const std::unordered_map<
          std::string, std::vector<const InferenceParameter*>>& models,
      std::set<std::string>* added, std::set<std::string>* deleted,
      std::set<std::string>* modified, std::set<std::string>* unmodified,
      ModelInfoMap* updated_infos, bool* all_models_polled);

  /// Helper function for Poll() to initialize ModelInfo for the model.
  /// \param name The name of the model.
  /// \param path The model path. Empty path means the model is provided via
  /// 'params'
  /// \param params The model parameters provided for polling model.
  /// \param info Return the updated ModelInfo. 'nullptr' will be returned if
  /// existing ModelInfo for the model should be reused.
  /// \return The error status.
  Status InitializeModelInfo(
      const std::string& name, const std::string& path,
      const std::vector<const InferenceParameter*>& params,
      std::unique_ptr<ModelInfo>* info);

  /// Load models based on the dependency graph. The function will iteratively
  /// load models that all the models they depend on has been loaded, and unload
  /// models if their dependencies are no longer satisfied.
  /// \return The status of the model loads.
  std::map<std::string, Status> LoadModelByDependency();

  /// Helper function to update the dependency graph based on the poll result
  /// \param added The names of the models added to the repository.
  /// \param deleted The names of the models removed from the repository.
  /// \param modified The names of the models remaining in the
  /// repository that have been changed.
  /// \param deleted_dependents The names of dependent models to be removed
  /// from the repository.
  /// \return The error status.
  Status UpdateDependencyGraph(
      const std::set<std::string>& added, const std::set<std::string>& deleted,
      const std::set<std::string>& modified,
      std::set<std::string>* deleted_dependents = nullptr);

  /// Helper function to uncheck the nodes because the model that they depends
  /// on has changed. The unchecked nodes will be validated again.
  /// The function will be call recursively to uncheck all downstreams.
  /// \param downstreams The nodes to be unchecked.
  /// \param updated_nodes Return the nodes that have been unchecked
  void UncheckDownstream(NodeSet* downstreams, NodeSet* updated_nodes);

  /// Helper function to construct the edges between nodes in dependency graph.
  /// \param updated_node The node that is newly added or modified.
  /// \return True if the node represents an ensemble model. False otherwise.
  bool ConnectDependencyGraph(DependencyNode* updated_node);

  /// Get the model info for a named model.
  /// \param name The model name.
  /// \param model_info Returns the model information.
  /// \return OK if found, NOT_FOUND otherwise.
  Status GetModelInfo(const std::string& name, ModelInfo** model_info);

  /// Get the models to be loaded / unloaded based on the model loaded in
  /// previous iteration.
  /// \param loaded_models The models loaded / unloaded in previous iteration.
  /// Unloaded models will be represented as models with no loaded versions.
  /// \return A pair of node set containing models to be loaded and models to be
  /// unloaded for the next iteration.
  std::pair<NodeSet, NodeSet> ModelsToLoadUnload(const NodeSet& loaded_models);

  /// Check if the node is ready for the next iteration. A node is ready if the
  /// node is invalid (containing invalid model config or its depdencies failed
  /// to load) or all of its dependencies are satisfied.
  /// \param node The node to be checked.
  /// \return True if the node is ready. False otherwise.
  bool CheckNode(DependencyNode* node);

  Status CircularcyCheck(
      DependencyNode* current_node, const DependencyNode* start_node);

  bool ModelDirectoryOverride(
      const std::vector<const InferenceParameter*>& model_params);

  std::set<std::string> repository_paths_;
  const bool autofill_;
  const bool polling_enabled_;
  const bool model_control_enabled_;
  const double min_compute_capability_;

  std::mutex poll_mu_;
  ModelInfoMap infos_;

  std::unordered_map<std::string, std::unique_ptr<DependencyNode>>
      dependency_graph_;
  std::unordered_map<std::string, std::unique_ptr<DependencyNode>>
      missing_nodes_;

  // Mappings from (overridden) model names to a pair of their repository and
  // absolute path
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      model_mappings_;

  std::unique_ptr<ModelLifeCycle> model_life_cycle_;
};

}}  // namespace triton::core
