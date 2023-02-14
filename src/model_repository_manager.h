// Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ModelIndex(const ModelIdentifier& n)
        : name_only_(true), namespace_(n.namespace_), name_(n.name_), version_(-1),
          state_(ModelReadyState::UNKNOWN)
    {
    }
    ModelIndex(
        const ModelIdentifier& n, const int64_t v, const ModelReadyState s,
        const std::string& r)
        : name_only_(false), namespace_(n.namespace_), name_(n.name_), version_(v), state_(s), reason_(r)
    {
    }
    const bool name_only_;
    const std::string namespace_;
    const std::string name_;
    const int64_t version_;
    const ModelReadyState state_;
    const std::string reason_;
  };

  /// A basic unit in dependency graph that records the models seen by the model
  /// repository manager.
  struct DependencyNode {
    DependencyNode(const ModelIdentifier& model_id)
        : status_(Status::Success), model_id_(model_id), checked_(false),
          connected_(false)
    {
    }

    void DisconnectUpstream(DependencyNode* upstream) { upstreams_.erase(upstream); }
    void DisconnectDownstream(DependencyNode* downstream) { downstreams_.erase(downstream); }

    // Overall status
    Status status_;

    // Poll info
    ModelIdentifier model_id_;
    bool explicitly_load_;
    inference::ModelConfig model_config_;

    // Graph info
    bool checked_;
    bool connected_;
    // store only the model names for missing upstreams, and remove from it
    // only if the upstream is exactly matched. This variable works with
    // 'missing_nodes_' in DependencyGraph to provide bi-directional lookup for
    // dependency resolution (exact / fuzzy match).
    // i.e. the node will look for upstream node that has matching identifier,
    // but upstream node with different namspace can still be used if not found.
    std::set<std::string> missing_upstreams_;
    std::unordered_map<DependencyNode*, std::set<int64_t>> upstreams_;
    std::set<DependencyNode*> downstreams_;

    // Lifecycle info
    std::set<int64_t> loaded_versions_;
  };

  // Interface for dependency graph operations
  class DependencyGraph {
   public:
    DependencyGraph() = delete;

    // Passing reference of 'global_map'.
    // There is coupling between the dependency graph and global map: global map
    // will be used to resolve dependency (node connectivity), and global map
    // will be updated to reflect node changes.
    DependencyGraph(std::unordered_map<std::string, std::set<ModelIdentifier>>& global_map)
     : global_map_ref_(global_map) {}

    std::unordered_map<ModelIdentifier, std::unique_ptr<DependencyNode>>*
    MutableNodes() {return &nodes_; }

    // Remove the given set of nodes, return two sets of nodes: The first set
    // contains existing nodes to be re-evaluated, because they depend on
    // the nodes removed; the second set contains all the nodes removed in this
    // operation.
    std::pair<std::set<ModelIdentifier>, std::set<ModelIdentifier>> RemoveNodes(const std::set<ModelIdentifier>& nodes, const bool cascading_removal);

    // Update the given set of nodes to reflect the latest model information polled,
    // returns existing nodes to be re-evaluated, including the modified node.
    std::set<ModelIdentifier> UpdateNodes(const std::set<ModelIdentifier>& nodes, const ModelRepositoryManager* model_manager);

    // Add the given set of nodes to the dependency graph,
    // returns existing nodes to be re-evaluated, including the added node.
    std::set<ModelIdentifier> AddNodes(const std::set<ModelIdentifier>& nodes, const ModelRepositoryManager* model_manager);

    // Helper function on the 'node_id' to construct the edges between nodes
    // in the dependency graph. The node status will be updated if the node
    // has missing edges. This function should be called after all node changes
    // are made to the dependency graph.
    void ConnectDependencyGraph(const ModelIdentifier& node_id);

    // Look up node in dependency graph with matching model identifier. If not found and fuzzy match
    // is allowed, a node in different namespace will be returned if it is the only node with the same
    // name
    DependencyNode* FindNode(const ModelIdentifier& model_id, const bool allow_fuzzy_matching);

    // Check if there is circular dependency on the given node, the node
    // status will be updated if the node has circular dependency.
    void CircularDependencyCheck(const ModelIdentifier& node_id);

   private:
    // Recursively uncheck the downstream, so the downstreams will need to be
    // re-checked at later stage to propagate the impact of upstream changes.
    void UncheckDownstream(const std::set<DependencyNode*>& downstreams);

    // Remove the node of the given identifier from dependency graph,
    // and its reference in other nodes.
    // Returns two sets of identifiers of the existing nodes that were linked to
    // the removed node. The first set of the returned identifier is the
    // "upstreams" of the node (i.e. composing models of the ensemble), the
    // second set is the "downstreams" of the node (i.e. the model is required
    // by other ensembles)
    std::pair<std::set<ModelIdentifier>, std::set<ModelIdentifier>> RemoveNode(const ModelIdentifier& model_id);

    // Recursive version for internal use.
    Status CircularDependencyCheck(DependencyNode* current_node, const DependencyNode* start_node);

    std::unordered_map<std::string, std::set<ModelIdentifier>>& global_map_ref_;
    std::unordered_map<ModelIdentifier, std::unique_ptr<DependencyNode>> nodes_;
    // A list of model names that there are nodes depdending on but not present.
    // Note that the key is not ModelIdentifier to allow more flexible matching.
    std::unordered_map<std::string, std::set<ModelIdentifier>> missing_nodes_;
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
      const bool enable_model_namespacing,
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
  const std::set<std::tuple<ModelIdentifier, int64_t, size_t>> InflightStatus();

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

  /// Obtain the specified model.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model.
  /// \param model Return the model object.
  /// \return error status.
  Status GetModel(
      const std::string& model_name, const int64_t model_version,
      std::shared_ptr<Model>* model);

  /// Obtain the specified model.
  /// \param model_id The identifier of the model.
  /// \param model_version The version of the model.
  /// \param model Return the model object.
  /// \return error status.
  Status GetModel(
      const ModelIdentifier& model_id, const int64_t model_version,
      std::shared_ptr<Model>* model);

  Status FindModelIdentifier(const std::string& model_name, ModelIdentifier* model_id);

  /// Get the index of all models in all repositories.
  /// \param ready_only If true return only index of models that are ready.
  /// \param index Returns the index.
  /// \return error status.
  Status RepositoryIndex(const bool ready_only, std::vector<ModelIndex>* index);

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
      std::unordered_map<ModelIdentifier, std::unique_ptr<ModelInfo>>;

  // Set of DependencyNode
  using NodeSet = std::set<DependencyNode*>;

  ModelRepositoryManager(
      const std::set<std::string>& repository_paths, const bool autofill,
      const bool polling_enabled, const bool model_control_enabled,
      const double min_compute_capability,
      const bool enable_model_namespacing,
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
      std::set<ModelIdentifier>* added, std::set<ModelIdentifier>* deleted,
      std::set<ModelIdentifier>* modified, std::set<ModelIdentifier>* unmodified,
      ModelInfoMap* updated_infos, bool* all_models_polled);

  /// Helper function for Poll() to initialize ModelInfo for the model.
  /// \param model_id The identifier of the model.
  /// \param path The model path. Empty path means the model is provided via
  /// 'params'
  /// \param params The model parameters provided for polling model.
  /// \param info Return the updated ModelInfo. 'nullptr' will be returned if
  /// existing ModelInfo for the model should be reused.
  /// \return The error status.
  Status InitializeModelInfo(
      const ModelIdentifier& model_id, const std::string& path,
      const std::vector<const InferenceParameter*>& params,
      std::unique_ptr<ModelInfo>* info);

  /// Load models based on the dependency graph. The function will iteratively
  /// load models that all the models they depend on has been loaded, and unload
  /// models if their dependencies are no longer satisfied.
  /// \return The status of the model loads.
  std::map<ModelIdentifier, Status> LoadModelByDependency();

  /// Helper function to update the dependency graph based on the poll result
  /// \param added The names of the models added to the repository.
  /// \param deleted The names of the models removed from the repository.
  /// \param modified The names of the models remaining in the
  /// repository that have been changed.
  /// \param deleted_dependents The names of dependent models to be removed
  /// from the repository.
  /// \return The error status.
  Status UpdateDependencyGraph(
      const std::set<ModelIdentifier>& added, const std::set<ModelIdentifier>& deleted,
      const std::set<ModelIdentifier>& modified,
      std::set<ModelIdentifier>* deleted_dependents = nullptr);

  /// Get the model info for a named model.
  /// \param name The model name.
  /// \param model_info Returns the model information.
  /// \return OK if found, NOT_FOUND otherwise.
  Status GetModelInfo(const ModelIdentifier& model_id, ModelInfo** model_info) const;

  /// Get the models to be loaded / unloaded based on the model loaded in
  /// previous iteration.
  /// \param loaded_models The models loaded / unloaded in previous iteration.
  /// Unloaded models will be represented as models with no loaded versions.
  /// \return A pair of node set containing models to be loaded and models to be
  /// unloaded for the next iteration.
  std::pair<NodeSet, NodeSet> ModelsToLoadUnload(
      const NodeSet& loaded_models,
      const std::map<ModelIdentifier, Status>& model_load_status);

  /// Check if the node is ready for the next iteration. A node is ready if the
  /// node is invalid (containing invalid model config or its depdencies failed
  /// to load) or all of its dependencies are satisfied.
  /// \param node The node to be checked.
  /// \return True if the node is ready. False otherwise.
  bool CheckNode(
      DependencyNode* node,
      const std::map<ModelIdentifier, Status>& model_load_status);

  bool ModelDirectoryOverride(
      const std::vector<const InferenceParameter*>& model_params);

  const bool autofill_;
  const bool polling_enabled_;
  const bool model_control_enabled_;
  const double min_compute_capability_;

  std::mutex poll_mu_;

  // Dependency graph

  // WAR to avoid change in behavior (mainly error reporting) if model namespace
  // is not enable.
  std::function<Status(const std::string& model_name, ModelIdentifier* model_id)> find_identifier_fn_;
  // A map from model name to model identifiers that share the same model name
  std::unordered_map<std::string, std::set<ModelIdentifier>> global_map_;
  DependencyGraph dependency_graph_;

  // Repository specific..

  const bool enable_model_namespacing_;
  ModelInfoMap infos_;
  std::set<std::string> repository_paths_;
  // Mappings from (overridden) model names to a pair of their repository and
  // absolute path
  // [FIXME] key should be updated to contain namespace as well, need to work with
  // enable namespace. Need to revisit with repository side of things
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      model_mappings_;

  // Model lifecycle

  std::unique_ptr<ModelLifeCycle> model_life_cycle_;
};

}}  // namespace triton::core
