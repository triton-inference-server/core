// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "filesystem.h"
#include "infer_request.h"
#include "model.h"
#include "model_config.pb.h"
#include "status.h"

namespace triton { namespace core {

class InferenceServer;
class TritonModelInstance;

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
      const int64_t version, inference::ModelConfig model_config,
      const bool is_config_provided, std::unique_ptr<TritonModel>* model);
  ~TritonModel();

  const std::string& LocalizedModelPath() const
  {
    return localized_model_dir_->Path();
  }
  InferenceServer* Server() { return server_; }
  bool AutoCompleteConfig() const { return auto_complete_config_; }
  Status UpdateModelConfig(
      const uint32_t config_version,
      TRITONSERVER_Message* updated_config_message);
  const std::shared_ptr<TritonBackend>& Backend() const { return backend_; }
  std::vector<std::shared_ptr<TritonModelInstance>> Instances();
  bool DeviceBlocking() const { return device_blocking_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  // Override base functions with its thread safe version.
  Status Enqueue(std::unique_ptr<InferenceRequest>& request);
  size_t InflightInferenceCount();
  void Stop();

  // Called by 'backend_model_instance' to move new instances to this object.
  // The new instances are registered into the background.
  Status RegisterInstance(
      std::shared_ptr<TritonModelInstance>&& instance, const bool passive);

  // Update instance group. 'caller_lock' will be released when creating new
  // instances and re-held when returning, to allow atomic switch over to the
  // new instances.
  Status UpdateInstanceGroup(
      const inference::ModelConfig& new_model_config,
      std::unique_lock<std::mutex>* caller_lock);

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
      const double min_compute_capability, const int64_t version,
      const inference::ModelConfig& config, const bool auto_complete_config,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map);

  // Replace the foreground instances with background instances. It is the
  // caller responsibility to hold the 'update_mu_' if the scheduler is to be
  // updated.
  Status CommitInstances();

  // Return the current scheduler thread safe.
  std::shared_ptr<Scheduler> GetScheduler();

  // Set or update the scheduler. It is the caller responsibility to hold the
  // 'update_mu_' if the scheduler is to be updated.
  Status SetSchedulerMutable(std::unique_ptr<Scheduler> scheduler);

  // Set the scheduler based on the model configuration and foreground
  // 'instances'. It is the caller responsibility to hold the 'update_mu_' if
  // the scheduler is to be updated.
  Status SetConfiguredScheduler();

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

  Status Initialize();
  Status WarmUp();

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
  // The backend cmdline config when this object is created.
  const triton::common::BackendCmdlineConfigMap backend_cmdline_config_map_;
  // The host policy map when this object is created.
  const triton::common::HostPolicyCmdlineConfigMap host_policy_map_;
  // The device blocking when this object is created.
  bool device_blocking_;

  // The localized repo directory holding the model. If localization
  // required creation of a temporary local copy then that copy will
  // persist as along as this object is retained by this model.
  std::shared_ptr<LocalizedPath> localized_model_dir_;

  // Backend used by this model.
  std::shared_ptr<TritonBackend> backend_;

  // Protect concurrent access while updating. Currently 'scheduler_' and
  // variation of 'instances_' vectors need to be protected.
  std::mutex update_mu_;

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
