// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <memory>
#include <string>

#include "backend_manager.h"
#include "backend_model_instance.h"
#include "filesystem/api.h"
#include "infer_request.h"
#include "model.h"
#include "model_config.pb.h"
#include "status.h"

namespace triton { namespace core {

class InferenceServer;

//
// Represents a model.
//
// Inheriting from Model to implement backend APIs
//
class TritonModel : public Model {
 public:
  typedef TRITONSERVER_Error* (*TritonModelBatchInclFn_t)(
      TRITONBACKEND_Request* request, void* userp, bool* should_include);
  typedef TRITONSERVER_Error* (*TritonModelBatchInitFn_t)(
      TRITONBACKEND_Batcher* batcher, void** userp);
  typedef TRITONSERVER_Error* (*TritonModelBatchFiniFn_t)(void* userp);
  typedef TRITONSERVER_Error* (*TritonModelBatcherInitFn_t)(
      TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model);
  typedef TRITONSERVER_Error* (*TritonModelBatcherFiniFn_t)(
      TRITONBACKEND_Batcher* batcher);

  static Status Create(
      InferenceServer* server, const std::string& model_path,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const ModelIdentifier& model_id, const int64_t version,
      inference::ModelConfig model_config, const bool is_config_provided,
      std::unique_ptr<TritonModel>* model);
  ~TritonModel();

  // Return path to the localized model directory.
  const std::string& LocalizedModelPath() const
  {
    return localized_model_dir_->Path();
  }

  // Return pointer to the underlying server.
  InferenceServer* Server() { return server_; }
  // Return whether the backend should attempt to auto-complete the model config
  bool AutoCompleteConfig() const { return auto_complete_config_; }
  // Called by TRITONBACKEND_ModelSetConfig() C-API.
  Status UpdateModelConfig(
      const uint32_t config_version,
      TRITONSERVER_Message* updated_config_message);
  // Return the underlying backend.
  const std::shared_ptr<TritonBackend>& Backend() const { return backend_; }
  // Return the backend command line config map.
  const triton::common::BackendCmdlineConfigMap& BackendConfigMap() const
  {
    return backend_cmdline_config_map_;
  }
  // Return the host policy command line config map.
  const triton::common::HostPolicyCmdlineConfigMap& HostPolicyMap() const
  {
    return host_policy_map_;
  }

  // True if different instances should be grouped by device; false otherwise.
  bool DeviceBlocking() const { return device_blocking_; }
  // Get a vector of non-passive background instances that share the device id.
  std::vector<std::shared_ptr<TritonModelInstance>> GetInstancesByDevice(
      int32_t device_id) const;

  // Manipulate the opaque state associated with this model.
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  // Update instance group.
  Status UpdateInstanceGroup(const inference::ModelConfig& new_model_config);

  // Custom batching function getters.
  TritonModelBatchInclFn_t ModelBatchInclFn() const { return batch_incl_fn_; }
  TritonModelBatchInitFn_t ModelBatchInitFn() const { return batch_init_fn_; }
  TritonModelBatchFiniFn_t ModelBatchFiniFn() const { return batch_fini_fn_; }
  TRITONBACKEND_Batcher** Batcher() { return &batcher_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonModel);

  TritonModel(
      InferenceServer* server,
      const std::shared_ptr<LocalizedPath>& localized_model_dir,
      const std::shared_ptr<TritonBackend>& backend,
      const double min_compute_capability, const ModelIdentifier& model_id,
      const int64_t version, const inference::ModelConfig& config,
      const bool auto_complete_config,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map);

  // Prepare the next set of instances on the background. Returns the instances
  // that will be added and removed if the next set of instances is to be
  // committed.
  Status PrepareInstances(
      const inference::ModelConfig& model_config,
      std::vector<std::shared_ptr<TritonModelInstance>>* added_instances,
      std::vector<std::shared_ptr<TritonModelInstance>>* removed_instances);
  // Replace the foreground instances with background instances.
  void CommitInstances();

  // Return all foreground instances indexed by its respective signature.
  std::unordered_map<
      TritonModelInstance::Signature,
      std::vector<std::shared_ptr<TritonModelInstance>>>
  IndexInstances() const;

  // Add a new instance into the background.
  void RegisterBackgroundInstance(
      std::shared_ptr<TritonModelInstance>&& instance, const bool passive);
  // Clear all background instances.
  void ClearBackgroundInstances();

  // Gets the execution policy setting from the backend.
  Status GetExecutionPolicy(const inference::ModelConfig& model_config);

  std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>>
  AccumulatedInstanceMemoryUsage() const override
  {
    std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>> res;
    // [FIXME] thread-safety on encountering instance change
    for (const auto& instances : {&instances_, &passive_instances_}) {
      for (const auto& instance : *instances) {
        const auto& imu = instance->MemoryUsage();
        for (const auto& mem_type_map : imu) {
          const auto& mem_type = mem_type_map.first;
          for (const auto& mem_id_map : mem_type_map.second) {
            const auto& mem_id = mem_id_map.first;
            const auto& byte_size = mem_id_map.second;
            res[mem_type][mem_id] += byte_size;
          }
        }
      }
    }
    return res;
  }

  // Set the scheduler based on the model configuration and the provided
  // instances.
  Status SetConfiguredScheduler(
      const std::vector<std::shared_ptr<TritonModelInstance>>& new_instances);
  // Update the set scheduler to the new set of instances.
  Status UpdateConfiguredScheduler(
      const std::vector<std::shared_ptr<TritonModelInstance>>& added_instances,
      const std::vector<std::shared_ptr<TritonModelInstance>>&
          removed_instances);

  // Set the batching strategy, if custom functions provided by user.
  // This function should only be called with the dynamic batcher.
  Status SetBatchingStrategy(const std::string& batch_libpath);

  // Merges the global backend configs with the specific
  // backend configs.
  static Status ResolveBackendConfigs(
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const std::string& backend_name,
      triton::common::BackendCmdlineConfig& config);

  // Sets defaults for some backend configurations when none are specified on
  // the command line.
  static Status SetBackendConfigDefaults(
      triton::common::BackendCmdlineConfig& config);

  // Get the search paths to the backend shared library.
  static std::vector<std::string> GetBackendLibrarySearchPaths(
      const std::string& model_path, int64_t version,
      const std::string& backend_dir, const std::string& backend_name);

  // Get backend library directory and path, and search paths for the library
  // and whether the backend is based on Python backend. The model configuration
  // runtime field will be updated if left empty.
  static Status GetBackendLibraryProperties(
      const std::string& model_path, int64_t version,
      const std::string& backend_dir, const std::string& backend_name,
      inference::ModelConfig* model_config, bool* is_python_based_backend,
      std::vector<std::string>* search_paths, std::string* backend_libdir,
      std::string* backend_libpath);

  // Get 'backend_libname', 'backend_libdir', 'backend_libpath' and
  // 'is_python_based_backend' by searching for different possible backend
  // library names on 'search_paths'. Searching for Python based backend
  // runtime is limited to 'backend_dir'.
  static Status GetBackendRuntimeLibraryName(
      const std::string& backend_dir, const std::string& backend_name,
      const std::vector<std::string>& search_paths,
      std::string* backend_libname, std::string* backend_libdir,
      std::string* backend_libpath, bool* is_python_based_backend);

  // Search for 'backend_libname' on 'search_paths'. If found, the matching
  // search path will be stored in 'backend_libdir' and the backend library path
  // will be stored in 'backend_libpath'. If not found, 'backend_libpath' will
  // be set to empty.
  static Status FindBackendLibraryPath(
      const std::vector<std::string>& search_paths,
      const std::string& backend_libname, std::string* backend_libdir,
      std::string* backend_libpath);

  // Assemble the C++ runtime library name.
  static std::string AssembleCPPRuntimeLibraryName(
      const std::string& backend_name);

  // Clear library handles.
  void ClearHandles();

  // The server object that owns this model. The model holds this as a
  // raw pointer because the lifetime of the server is guaranteed to
  // be longer than the lifetime of a model owned by the server.
  InferenceServer* server_;

  // The minimum supported compute capability on device.
  const double min_compute_capability_;
  // Whether the backend should attempt to auto-complete the model config.
  const bool auto_complete_config_;
  // The backend cmdline config.
  const triton::common::BackendCmdlineConfigMap backend_cmdline_config_map_;
  // The host policy map.
  const triton::common::HostPolicyCmdlineConfigMap host_policy_map_;
  // The device blocking. It should not be changed after the model is created.
  bool device_blocking_;

  // The localized repo directory holding the model. If localization
  // required creation of a temporary local copy then that copy will
  // persist as along as this object is retained by this model.
  std::shared_ptr<LocalizedPath> localized_model_dir_;

  // Backend used by this model.
  std::shared_ptr<TritonBackend> backend_;

  // The model instances for this model. Passive instances are loaded but not
  // added to the scheduler.
  std::vector<std::shared_ptr<TritonModelInstance>> instances_;
  std::vector<std::shared_ptr<TritonModelInstance>> passive_instances_;
  // They are the background 'instances_' and 'passive_instances_', not yet
  // effective until committed.
  std::vector<std::shared_ptr<TritonModelInstance>> bg_instances_;
  std::vector<std::shared_ptr<TritonModelInstance>> bg_passive_instances_;

  // Opaque state associated with this model.
  void* state_;

  // Custom batching shared object handle, function pointers, and batcher
  // pointer.
  void* batch_dlhandle_ = nullptr;
  TritonModelBatchInclFn_t batch_incl_fn_ = nullptr;
  TritonModelBatchInitFn_t batch_init_fn_ = nullptr;
  TritonModelBatchFiniFn_t batch_fini_fn_ = nullptr;
  TritonModelBatcherInitFn_t batcher_init_fn_ = nullptr;
  TritonModelBatcherFiniFn_t batcher_fini_fn_ = nullptr;
  TRITONBACKEND_Batcher* batcher_ = nullptr;
};

}}  // namespace triton::core
