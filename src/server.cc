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

#include "server.h"

#include <stdint.h>
#include <time.h>

#include <algorithm>
#include <csignal>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "backend_manager.h"
#include "constants.h"
#include "cuda_utils.h"
#include "model.h"
#include "model_config.pb.h"
#include "model_config_utils.h"
#include "pinned_memory_manager.h"
#include "repo_agent.h"
#include "triton/common/async_work_queue.h"
#include "triton/common/logging.h"
#include "triton/common/model_config.h"
#include "triton/common/table_printer.h"

#ifdef TRITON_ENABLE_GPU
#include "cuda_block_manager.h"
#include "cuda_memory_manager.h"
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace core {

namespace {

// Scoped increment / decrement of atomic
class ScopedAtomicIncrement {
 public:
  explicit ScopedAtomicIncrement(std::atomic<uint64_t>& counter)
      : counter_(counter)
  {
    counter_++;
  }

  ~ScopedAtomicIncrement() { counter_--; }

 private:
  std::atomic<uint64_t>& counter_;
};

}  // namespace

//
// InferenceServer
//
InferenceServer::InferenceServer()
    : version_(TRITON_VERSION), ready_state_(ServerReadyState::SERVER_INVALID)
{
  id_ = "triton";
  extensions_.push_back("classification");
  extensions_.push_back("sequence");
  extensions_.push_back("model_repository");
  extensions_.push_back("model_repository(unload_dependents)");
  extensions_.push_back("schedule_policy");
  extensions_.push_back("model_configuration");
  extensions_.push_back("system_shared_memory");
  extensions_.push_back("cuda_shared_memory");
  extensions_.push_back("binary_tensor_data");
  extensions_.push_back("parameters");
#ifdef TRITON_ENABLE_STATS
  extensions_.push_back("statistics");
#endif  // TRITON_ENABLE_STATS
#ifdef TRITON_ENABLE_TRACING
  extensions_.push_back("trace");
#endif  // TRITON_ENABLE_TRACING
#ifdef TRITON_ENABLE_LOGGING
  extensions_.push_back("logging");
#endif  // TRITON_ENABLE_LOGGING
  strict_model_config_ = true;
  strict_readiness_ = true;
  exit_timeout_secs_ = 30;
  pinned_memory_pool_size_ = 1 << 28;
  buffer_manager_thread_count_ = 0;
  model_load_thread_count_ = 4;
  model_load_retry_count_ = 0;
  enable_model_namespacing_ = false;

#ifdef TRITON_ENABLE_GPU
  min_supported_compute_capability_ = TRITON_MIN_COMPUTE_CAPABILITY;
#else
  min_supported_compute_capability_ = 0.0;
#endif  // TRITON_ENABLE_GPU

  inflight_request_counter_ = 0;
}

Status
InferenceServer::Init()
{
  Status status;

  ready_state_ = ServerReadyState::SERVER_INITIALIZING;

  if (model_repository_paths_.empty()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return Status(
        Status::Code::INVALID_ARG, "--model-repository must be specified");
  }

  // RepoAgentManager
  if (repoagent_dir_.empty()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return Status(
        Status::Code::INVALID_ARG, "--repoagent-directory can not be empty");
  }

  status = TritonRepoAgentManager::SetGlobalSearchPath(repoagent_dir_);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }

  // BackendManager
  status = TritonBackendManager::Create(&backend_manager_);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }

  // CacheManager
  status = TritonCacheManager::Create(&cache_manager_, cache_dir_);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }

  // Only a single global cache is supported at this time.
  if (cache_config_map_.size() > 1) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return Status(
        Status::Code::INVALID_ARG,
        "found multiple cache configurations, but only a single cache is "
        "currently supported");
  }

  // Initialize each cache with its respective config
  for (const auto& iter : cache_config_map_) {
    const auto& name = iter.first;
    const auto& config = iter.second;
    std::shared_ptr<TritonCache> cache;
    status = cache_manager_->CreateCache(name, config, &cache);
    if (!status.IsOk()) {
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
      return status;
    }
  }

  if (buffer_manager_thread_count_ > 0) {
    status = CommonErrorToStatus(triton::common::AsyncWorkQueue::Initialize(
        buffer_manager_thread_count_));
    if (!status.IsOk()) {
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
      return status;
    }
  }

  std::unique_ptr<RateLimiter> local_rate_limiter;
  bool ignore_resources_and_priority =
      (rate_limit_mode_ == RateLimitMode::RL_OFF);

  status = RateLimiter::Create(
      ignore_resources_and_priority, rate_limit_resource_map_,
      &local_rate_limiter);
  rate_limiter_ = std::move(local_rate_limiter);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }

  PinnedMemoryManager::Options options(pinned_memory_pool_size_);
  status = PinnedMemoryManager::Create(options);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }


#ifdef TRITON_ENABLE_GPU
  // Set the default CUDA memory pool size for GPUs where it is not
  // set explicitly.
  std::set<int> supported_gpus;
  if (GetSupportedGPUs(&supported_gpus, min_supported_compute_capability_)
          .IsOk()) {
    for (const auto gpu : supported_gpus) {
      if (cuda_memory_pool_size_.find(gpu) == cuda_memory_pool_size_.end()) {
        cuda_memory_pool_size_[gpu] = 1 << 26;
      }
      if (cuda_virtual_address_space_size_.find(gpu) ==
          cuda_virtual_address_space_size_.end()) {
        cuda_virtual_address_space_size_[gpu] = 1 << 30;
      }
    }
  }

  CudaMemoryManager::Options cuda_options(
      min_supported_compute_capability_, cuda_memory_pool_size_);
  status = CudaMemoryManager::Create(cuda_options);
  // If CUDA memory manager can't be created, just log error as the
  // server can still function properly
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }

  status = CudaBlockManager::Create(min_supported_compute_capability_);
  // If CUDA memory manager can't be created, just log error as the
  // server can still function properly
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }

#endif  // TRITON_ENABLE_GPU

  status = EnablePeerAccess(min_supported_compute_capability_);
  if (!status.IsOk()) {
    // failed to enable peer access is not critical, just inefficient.
    LOG_WARNING << status.Message();
  }

  // Create the model manager for the repository. Unless model control
  // is disabled, all models are eagerly loaded when the manager is created.
  bool polling_enabled = (model_control_mode_ == ModelControlMode::MODE_POLL);
  bool model_control_enabled =
      (model_control_mode_ == ModelControlMode::MODE_EXPLICIT);
  const ModelLifeCycleOptions life_cycle_options(
      min_supported_compute_capability_, backend_cmdline_config_map_,
      host_policy_map_, model_load_thread_count_, model_load_retry_count_);
  status = ModelRepositoryManager::Create(
      this, version_, model_repository_paths_, startup_models_,
      strict_model_config_, model_config_name_, polling_enabled,
      model_control_enabled, life_cycle_options, enable_model_namespacing_,
      &model_repository_manager_);
  if (!status.IsOk()) {
    if (model_repository_manager_ == nullptr) {
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    } else {
      // If error is returned while the manager is set, we assume the
      // failure is due to a model not loading correctly so we just
      // continue if not exiting on error.
      ready_state_ = ServerReadyState::SERVER_READY;
      PrintBackendAndModelSummary();
    }
  } else {
    ready_state_ = ServerReadyState::SERVER_READY;
    PrintBackendAndModelSummary();
  }

  return status;
}

InferenceServer::~InferenceServer()
{
  PinnedMemoryManager::Reset();
#ifdef TRITON_ENABLE_GPU
  CudaMemoryManager::Reset();
  CudaBlockManager::Reset();
#endif  // TRITON_ENABLE_GPU
}

Status
InferenceServer::Stop(const bool force)
{
  if (!force && (ready_state_ != ServerReadyState::SERVER_READY)) {
    return Status::Success;
  }

  ready_state_ = ServerReadyState::SERVER_EXITING;

  if (model_repository_manager_ == nullptr) {
    LOG_INFO << "No server context available. Exiting immediately.";
    return Status::Success;
  } else {
    LOG_INFO << "Waiting for in-flight requests to complete.";
  }

  Status status = model_repository_manager_->StopAllModels();
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }

  // Wait for all in-flight non-inference requests to complete and all
  // loaded models to unload, or for the exit timeout to expire.
  uint32_t exit_timeout_iters = exit_timeout_secs_;
  bool unloading_model = false;
  while (true) {
    if (!unloading_model) {
      // Check if all in-flight inference requests / sequences are completed
      const auto& inflight_status = model_repository_manager_->InflightStatus();
      LOG_INFO << "Timeout " << exit_timeout_iters << ": Found "
               << inflight_status.size()
               << " model versions that have in-flight inferences";
      for (const auto& inflight : inflight_status) {
        LOG_INFO << "Model '" << std::get<0>(inflight) << "' "
                 << "(version " << std::get<1>(inflight) << ") has "
                 << std::get<2>(inflight) << " in-flight inferences";
      }

      if (inflight_status.size() == 0) {
        status = model_repository_manager_->UnloadAllModels();
        if (!status.IsOk()) {
          LOG_WARNING << status.Message();
        } else {
          unloading_model = true;
          LOG_INFO << "All models are stopped, unloading models";
          continue;
        }
      }
    } else {
      const auto& live_models = model_repository_manager_->LiveModelStates();
      size_t bg_models_size = model_repository_manager_->BackgroundModelsSize();
      size_t num_models = live_models.size() + bg_models_size;

      LOG_INFO << "Timeout " << exit_timeout_iters << ": Found " << num_models
               << " live models and " << inflight_request_counter_
               << " in-flight non-inference requests";
      if (LOG_VERBOSE_IS_ON(1)) {
        for (const auto& m : live_models) {
          for (const auto& v : m.second) {
            LOG_VERBOSE(1) << m.first << " v" << v.first << ": "
                           << ModelReadyStateString(v.second.first);
          }
        }
      }

      if ((num_models == 0) && (inflight_request_counter_ == 0)) {
        return Status::Success;
      }
    }
    if (exit_timeout_iters <= 0) {
      break;
    }

    exit_timeout_iters--;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status(
      Status::Code::INTERNAL, "Exit timeout expired. Exiting immediately.");
}

Status
InferenceServer::PollModelRepository()
{
  LOG_VERBOSE(1) << "Polling model repository";

  // Look for changes and update the loaded model configurations
  // appropriately.
  if (ready_state_ == ServerReadyState::SERVER_READY) {
    ScopedAtomicIncrement inflight(inflight_request_counter_);
    RETURN_IF_ERROR(model_repository_manager_->PollAndUpdate());
  }

  return Status::Success;
}

Status
InferenceServer::IsLive(bool* live)
{
  *live = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(Status::Code::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered live if it can respond to this health
  // request and it was able to initialize.
  *live =
      ((ready_state_ != ServerReadyState::SERVER_INVALID) &&
       (ready_state_ != ServerReadyState::SERVER_INITIALIZING) &&
       (ready_state_ != ServerReadyState::SERVER_FAILED_TO_INITIALIZE));
  return Status::Success;
}

Status
InferenceServer::IsReady(bool* ready)
{
  *ready = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(Status::Code::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered ready if it is in the ready state.
  // Additionally can report ready only when all models are ready.
  *ready = (ready_state_ == ServerReadyState::SERVER_READY);
  if (*ready && strict_readiness_) {
    // Strict readiness... get the model status and make sure all
    // models are ready.
    const auto model_versions = model_repository_manager_->ModelStates();

    for (const auto& mv : model_versions) {
      // If a model status is present but no version status,
      // the model is not ready as there is no proper version to be served
      if (mv.second.size() == 0) {
        *ready = false;
        goto strict_done;
      }
      for (const auto& vs : mv.second) {
        // Okay if model is not ready due to unload
        if ((vs.second.first != ModelReadyState::READY) &&
            (vs.second.second != "unloaded")) {
          *ready = false;
          goto strict_done;
        }
      }
    }
  strict_done:;
  }

  return Status::Success;
}

Status
InferenceServer::ModelIsReady(
    const std::string& model_name, const int64_t model_version, bool* ready)
{
  *ready = false;

  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  std::shared_ptr<Model> model;
  if (GetModel(model_name, model_version, &model).IsOk()) {
    ModelReadyState state;
    if (model_repository_manager_
            ->ModelState(model_name, model->Version(), &state)
            .IsOk()) {
      *ready = (state == ModelReadyState::READY);
    }
  }

  return Status::Success;
}

Status
InferenceServer::ModelReadyVersions(
    const std::string& model_name, std::vector<int64_t>* versions)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  const auto version_states =
      model_repository_manager_->VersionStates(model_name);
  for (const auto& pr : version_states) {
    if (pr.second.first == ModelReadyState::READY) {
      versions->push_back(pr.first);
    }
  }

  return Status::Success;
}

Status
InferenceServer::ModelReadyVersions(
    std::map<std::string, std::vector<int64_t>>* ready_model_versions)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  const auto model_versions =
      model_repository_manager_->LiveModelStates(true /* strict_readiness */);

  ready_model_versions->clear();
  std::vector<int64_t> versions;
  for (const auto& mv_pair : model_versions) {
    for (const auto& vs_pair : mv_pair.second) {
      versions.emplace_back(vs_pair.first);
    }
    ready_model_versions->emplace(mv_pair.first.str(), std::move(versions));
  }

  return Status::Success;
}

Status
InferenceServer::RepositoryIndex(
    const bool ready_only,
    std::vector<ModelRepositoryManager::ModelIndex>* index)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  return model_repository_manager_->RepositoryIndex(ready_only, index);
}

Status
InferenceServer::InferAsync(std::unique_ptr<InferenceRequest>& request)
{
  // Allow inference request while server exiting to provide graceful
  // completion of inference sequence that spans multiple requests.
  if ((ready_state_ != ServerReadyState::SERVER_READY) &&
      (ready_state_ != ServerReadyState::SERVER_EXITING)) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

#ifdef TRITON_ENABLE_STATS
  request->CaptureRequestStartNs();
  INFER_TRACE_ACTIVITY(
      request->TraceProxy(), TRITONSERVER_TRACE_REQUEST_START,
      request->RequestStartNs());
#endif  // TRITON_ENABLE_STATS

  return InferenceRequest::Run(request);
}

Status
InferenceServer::LoadModel(
    const std::unordered_map<
        std::string, std::vector<const InferenceParameter*>>& models)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  auto action_type = ActionType::LOAD;
  return model_repository_manager_->LoadUnloadModel(
      models, action_type, false /* unload_dependents */);
}

Status
InferenceServer::UnloadModel(
    const std::string& model_name, const bool unload_dependents)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  auto action_type = ActionType::UNLOAD;
  return model_repository_manager_->LoadUnloadModel(
      {{model_name, {}}}, action_type, unload_dependents);
}

Status
InferenceServer::PrintBackendAndModelSummary()
{
  // Repository Agents Summary
  std::vector<std::string> repoagent_headers;
  repoagent_headers.emplace_back("Repository Agent");
  repoagent_headers.emplace_back("Path");

  triton::common::TablePrinter repoagents_table(repoagent_headers);

  std::unique_ptr<std::unordered_map<std::string, std::string>> repoagent_state;
  RETURN_IF_ERROR(TritonRepoAgentManager::AgentState(&repoagent_state));

  for (const auto& repoagent_pair : *repoagent_state) {
    std::vector<std::string> repoagent_record;
    repoagent_record.emplace_back(repoagent_pair.first);
    repoagent_record.emplace_back(repoagent_pair.second);
    repoagents_table.InsertRow(repoagent_record);
  }
  std::string repoagents_table_string = repoagents_table.PrintTable();
  LOG_INFO << repoagents_table_string;

  // Backends Summary
  std::vector<std::string> backend_headers;
  backend_headers.emplace_back("Backend");
  backend_headers.emplace_back("Path");
  backend_headers.emplace_back("Config");

  triton::common::TablePrinter backends_table(backend_headers);

  std::unique_ptr<std::unordered_map<std::string, std::vector<std::string>>>
      backend_state;
  RETURN_IF_ERROR(backend_manager_->BackendState(&backend_state));

  for (const auto& backend_pair : *backend_state) {
    std::vector<std::string> backend_record;

    // Backend Name
    backend_record.emplace_back(backend_pair.first);

    // Backend config and lib path
    for (const auto& backend_field : backend_pair.second) {
      backend_record.emplace_back(backend_field);
    }
    backends_table.InsertRow(backend_record);
  }
  std::string backends_table_string = backends_table.PrintTable();
  LOG_INFO << backends_table_string;

  // Models Summary
  auto model_states = model_repository_manager_->ModelStates();

  std::vector<std::string> model_headers;
  model_headers.emplace_back("Model");
  model_headers.emplace_back("Version");
  model_headers.emplace_back("Status");

  triton::common::TablePrinter models_table(model_headers);

  for (const auto& model_state : model_states) {
    auto model_version_map = model_state.second;
    ModelIdentifier model_id = model_state.first;

    // If model_version_map size is zero, no version is found for this model
    if (model_version_map.size() == 0) {
      std::vector<std::string> model_record;
      model_record.emplace_back(model_id.str());
      model_record.emplace_back("-");
      model_record.emplace_back("Not loaded: No model version was found");
      models_table.InsertRow(model_record);
    } else {
      for (const auto& model_map : model_version_map) {
        std::vector<std::string> model_record;
        std::string model_version = std::to_string(model_map.first);
        auto model_status_pair = model_map.second;
        std::string model_status =
            ModelReadyStateString(model_status_pair.first);

        if (model_status_pair.second != "") {
          model_status += ": " + model_status_pair.second;
        }

        model_record.emplace_back(model_id.str());
        model_record.emplace_back(model_version);
        model_record.emplace_back(model_status);
        models_table.InsertRow(model_record);
      }
    }
  }
  std::string models_table_string = models_table.PrintTable();
  LOG_INFO << models_table_string;

  return Status::Success;
}

Status
InferenceServer::RegisterModelRepository(
    const std::string& repository,
    const std::unordered_map<std::string, std::string>& model_mapping)
{
  return model_repository_manager_->RegisterModelRepository(
      repository, model_mapping);
}

Status
InferenceServer::UnregisterModelRepository(const std::string& repository)
{
  return model_repository_manager_->UnregisterModelRepository(repository);
}

}}  // namespace triton::core
