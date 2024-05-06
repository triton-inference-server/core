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
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "backend_manager.h"
#include "cache_manager.h"
#include "infer_parameter.h"
#include "model_config.pb.h"
#include "model_repository_manager/model_repository_manager.h"
#include "rate_limiter.h"
#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

// Maps cache name -> json config string
using CacheConfigMap = std::unordered_map<std::string, std::string>;

class Model;
class InferenceRequest;

enum class ModelControlMode { MODE_NONE, MODE_POLL, MODE_EXPLICIT };

enum class RateLimitMode { RL_EXEC_COUNT, RL_OFF };

// Readiness status for the inference server.
enum class ServerReadyState {
  // The server is in an invalid state and will likely not response
  // correctly to any requests.
  SERVER_INVALID,

  // The server is initializing.
  SERVER_INITIALIZING,

  // The server is ready and accepting requests.
  SERVER_READY,

  // The server is exiting and will not respond to requests.
  SERVER_EXITING,

  // The server did not initialize correctly.
  SERVER_FAILED_TO_INITIALIZE
};

// Inference server information.
class InferenceServer {
 public:
  // Construct an inference server.
  InferenceServer();

  ~InferenceServer();

  // Initialize the server. Return true on success, false otherwise.
  Status Init();

  // Stop the server.  Return true if all models are unloaded, false
  // if exit timeout occurs. If 'force' is true attempt to stop the
  // server even if it is not in a ready state.
  Status Stop(const bool force = false);

  // Check the model repository for changes and update server state
  // based on those changes.
  Status PollModelRepository();

  // Server health
  Status IsLive(bool* live);
  Status IsReady(bool* ready);

  // Model health
  Status ModelIsReady(
      const std::string& model_name, const int64_t model_version, bool* ready);

  // Return the ready versions of specific model
  Status ModelReadyVersions(
      const std::string& model_name, std::vector<int64_t>* versions);

  // Return the ready versions of all models
  Status ModelReadyVersions(
      std::map<std::string, std::vector<int64_t>>* model_versions);

  /// Get the index of all models in all repositories.
  /// \param ready_only If true return only index of models that are ready.
  /// \param index Returns the index.
  /// \return error status.
  Status RepositoryIndex(
      const bool ready_only,
      std::vector<ModelRepositoryManager::ModelIndex>* index);

  // Inference. If Status::Success is returned then this function has
  // taken ownership of the request object and so 'request' will be
  // nullptr. If non-success is returned then the caller still retains
  // ownership of 'request'.
  Status InferAsync(std::unique_ptr<InferenceRequest>& request);

  // Load the corresponding model. Reload the model if it has been loaded.
  Status LoadModel(
      const std::unordered_map<
          std::string, std::vector<const InferenceParameter*>>& models);

  // Unload the corresponding model.
  Status UnloadModel(
      const std::string& model_name, const bool unload_dependents);

  // Print backends and models summary
  Status PrintBackendAndModelSummary();

  // Register model repository path and associated mappings
  Status RegisterModelRepository(
      const std::string& repository,
      const std::unordered_map<std::string, std::string>& model_mapping);

  // Unregister model repository path.
  Status UnregisterModelRepository(const std::string& repository);

  // Return the server version.
  const std::string& Version() const { return version_; }

  // Return the server extensions.
  const std::vector<const char*>& Extensions() const { return extensions_; }

  // Get / set the ID of the server.
  const std::string& Id() const { return id_; }
  void SetId(const std::string& id) { id_ = id; }

  // Get / set the model repository path
  const std::set<std::string>& ModelRepositoryPaths() const
  {
    return model_repository_paths_;
  }

  void SetModelRepositoryPaths(const std::set<std::string>& p)
  {
    model_repository_paths_ = p;
  }

  // Get / set model control mode.
  ModelControlMode GetModelControlMode() const { return model_control_mode_; }
  void SetModelControlMode(ModelControlMode m) { model_control_mode_ = m; }

  // Get / set the startup models
  const std::set<std::string>& StartupModels() const { return startup_models_; }
  void SetStartupModels(const std::set<std::string>& m) { startup_models_ = m; }

  // Get / set strict model configuration enable.
  bool StrictModelConfigEnabled() const { return strict_model_config_; }
  void SetStrictModelConfigEnabled(bool e) { strict_model_config_ = e; }

  // Get / set custom model configuration file name.
  std::string ModelConfigName() const { return model_config_name_; }
  void SetModelConfigName(const std::string& name)
  {
    model_config_name_ = name;
  }

  // Get / set rate limiter mode.
  RateLimitMode RateLimiterMode() const { return rate_limit_mode_; }
  void SetRateLimiterMode(RateLimitMode m) { rate_limit_mode_ = m; }

  // Get / set rate limit resource counts
  const RateLimiter::ResourceMap& RateLimiterResources() const
  {
    return rate_limit_resource_map_;
  }
  void SetRateLimiterResources(const RateLimiter::ResourceMap& rm)
  {
    rate_limit_resource_map_ = rm;
  }

  // Get / set the pinned memory pool byte size.
  int64_t PinnedMemoryPoolByteSize() const { return pinned_memory_pool_size_; }
  void SetPinnedMemoryPoolByteSize(int64_t s)
  {
    pinned_memory_pool_size_ = std::max((int64_t)0, s);
  }

  // Get / set whether response cache will be enabled server-wide.
  // NOTE: Models still need caching enabled in individual model configs.
  bool ResponseCacheEnabled()
  {
    // Only return true if cache was enabled, and has been initialized
    return response_cache_enabled_ && CacheManager() && CacheManager()->Cache();
  }
  void SetResponseCacheEnabled(bool e) { response_cache_enabled_ = e; }
  void SetCacheConfig(CacheConfigMap cfg) { cache_config_map_ = cfg; }
  std::string CacheDir() const { return cache_dir_; }
  void SetCacheDir(std::string dir) { cache_dir_ = dir; }

  // Get / set CUDA memory pool size
  const std::map<int, uint64_t>& CudaMemoryPoolByteSize() const
  {
    return cuda_memory_pool_size_;
  }

  void SetCudaMemoryPoolByteSize(const std::map<int, uint64_t>& s)
  {
    cuda_memory_pool_size_ = s;
  }

  // Get / set CUDA virtual address space size
  const std::map<int, size_t>& CudaVirtualAddressSpaceSize() const
  {
    return cuda_virtual_address_space_size_;
  }

  void SetCudaVirtualAddressSpaceSize(const std::map<int, size_t>& s)
  {
    cuda_virtual_address_space_size_ = s;
  }

  // Get / set the minimum support CUDA compute capability.
  double MinSupportedComputeCapability() const
  {
    return min_supported_compute_capability_;
  }
  void SetMinSupportedComputeCapability(double c)
  {
    min_supported_compute_capability_ = c;
  }

  // Get / set strict readiness enable.
  bool StrictReadinessEnabled() const { return strict_readiness_; }
  void SetStrictReadinessEnabled(bool e) { strict_readiness_ = e; }

  // Get / set the server exit timeout, in seconds.
  int32_t ExitTimeoutSeconds() const { return exit_timeout_secs_; }
  void SetExitTimeoutSeconds(int32_t s) { exit_timeout_secs_ = std::max(0, s); }

  void SetBufferManagerThreadCount(unsigned int c)
  {
    buffer_manager_thread_count_ = c;
  }

  void SetModelLoadThreadCount(unsigned int c) { model_load_thread_count_ = c; }

  void SetModelLoadRetryCount(unsigned int c) { model_load_retry_count_ = c; }

  void SetModelNamespacingEnabled(const bool e)
  {
    enable_model_namespacing_ = e;
  }

  // Set a backend command-line configuration
  void SetBackendCmdlineConfig(
      const triton::common::BackendCmdlineConfigMap& bc)
  {
    backend_cmdline_config_map_ = bc;
  }

  void SetHostPolicyCmdlineConfig(
      const triton::common::HostPolicyCmdlineConfigMap& hp)
  {
    host_policy_map_ = hp;
  }

  void SetRepoAgentDir(const std::string& d) { repoagent_dir_ = d; }

  // Return the requested model object.
  Status GetModel(
      const std::string& model_name, const int64_t model_version,
      std::shared_ptr<Model>* model)
  {
    // Allow model retrieval while server exiting to provide graceful
    // completion of inference sequence that spans multiple requests.
    if ((ready_state_ != ServerReadyState::SERVER_READY) &&
        (ready_state_ != ServerReadyState::SERVER_EXITING)) {
      return Status(Status::Code::UNAVAILABLE, "Server not ready");
    }
    return model_repository_manager_->GetModel(
        model_name, model_version, model);
  }

  // Return the requested model object.
  Status GetModel(
      const ModelIdentifier& model_id, const int64_t model_version,
      std::shared_ptr<Model>* model)
  {
    // Allow model retrieval while server exiting to provide graceful
    // completion of inference sequence that spans multiple requests.
    if ((ready_state_ != ServerReadyState::SERVER_READY) &&
        (ready_state_ != ServerReadyState::SERVER_EXITING)) {
      return Status(Status::Code::UNAVAILABLE, "Server not ready");
    }
    return model_repository_manager_->GetModel(model_id, model_version, model);
  }

  // Get the Backend Manager
  const std::shared_ptr<TritonBackendManager>& BackendManager()
  {
    return backend_manager_;
  }

  // Return the pointer to RateLimiter object.
  std::shared_ptr<RateLimiter> GetRateLimiter() { return rate_limiter_; }

  // Get the Cache Manager
  const std::shared_ptr<TritonCacheManager>& CacheManager()
  {
    return cache_manager_;
  }

 private:
  const std::string version_;
  std::string id_;
  std::vector<const char*> extensions_;

  std::set<std::string> model_repository_paths_;
  std::set<std::string> startup_models_;
  ModelControlMode model_control_mode_;
  bool strict_model_config_;
  bool strict_readiness_;
  std::string model_config_name_;
  uint32_t exit_timeout_secs_;
  uint32_t buffer_manager_thread_count_;
  uint32_t model_load_thread_count_;
  uint32_t model_load_retry_count_;
  bool enable_model_namespacing_;
  uint64_t pinned_memory_pool_size_;
  bool response_cache_enabled_;
  CacheConfigMap cache_config_map_;
  std::string cache_dir_;
  std::map<int, uint64_t> cuda_memory_pool_size_;
  std::map<int, size_t> cuda_virtual_address_space_size_;
  double min_supported_compute_capability_;
  triton::common::BackendCmdlineConfigMap backend_cmdline_config_map_;
  triton::common::HostPolicyCmdlineConfigMap host_policy_map_;
  std::string repoagent_dir_;
  RateLimitMode rate_limit_mode_;
  RateLimiter::ResourceMap rate_limit_resource_map_;

  // Current state of the inference server.
  ServerReadyState ready_state_;

  // Number of in-flight, non-inference requests. During shutdown we
  // attempt to wait for all in-flight non-inference requests to
  // complete before exiting (also wait for in-flight inference
  // requests but that is determined by model shared_ptr).
  std::atomic<uint64_t> inflight_request_counter_;

  std::shared_ptr<RateLimiter> rate_limiter_;
  std::unique_ptr<ModelRepositoryManager> model_repository_manager_;
  std::shared_ptr<TritonBackendManager> backend_manager_;
  std::shared_ptr<TritonCacheManager> cache_manager_;
};

}}  // namespace triton::core
