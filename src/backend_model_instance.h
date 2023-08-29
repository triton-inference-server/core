// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/core/span.hpp>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "constants.h"
#include "memory.h"
#include "metric_model_reporter.h"
#include "model_config.pb.h"
#include "model_config_utils.h"
#include "server_message.h"
#include "status.h"
#include "triton/common/sync_queue.h"

namespace triton { namespace core {

class TritonModel;
class InferenceRequest;

//
// Represents a model instance.
//
class TritonModelInstance {
 public:
  struct SecondaryDevice {
    SecondaryDevice(const std::string kind, const int64_t id)
        : kind_(kind), id_(id)
    {
    }
    const std::string kind_;
    const int64_t id_;
  };

  class Signature {
   public:
    Signature(
        const inference::ModelInstanceGroup& group_config, int32_t device_id)
        : group_config_(group_config), device_id_(device_id),
          hash_(std::hash<std::string>{}(
              std::to_string(device_id_) +
              InstanceConfigSignature(group_config_)))
    {
    }
    bool operator==(const Signature& rhs) const
    {
      return device_id_ == rhs.device_id_ &&
             EquivalentInInstanceConfig(group_config_, rhs.group_config_);
    }
    bool operator!=(const Signature& rhs) const { return !(*this == rhs); }
    std::size_t Hash() const { return hash_; }

   private:
    const inference::ModelInstanceGroup group_config_;
    const int32_t device_id_;
    const std::size_t hash_;
  };

  static Status CreateInstance(
      TritonModel* model, const std::string& name, const Signature& signature,
      TRITONSERVER_InstanceGroupKind kind, int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const std::string& host_policy_name,
      const inference::ModelRateLimiter& rate_limiter_config,
      const std::vector<SecondaryDevice>& secondary_devices,
      std::shared_ptr<TritonModelInstance>* triton_model_instance);
  ~TritonModelInstance();

  const std::string& Name() const { return name_; }
  Signature& GetSignature() { return signature_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }
  const triton::common::HostPolicyCmdlineConfig& HostPolicy() const
  {
    return host_policy_;
  }
  const TritonServerMessage& HostPolicyMessage() const
  {
    return host_policy_message_;
  }
  bool IsPassive() const { return passive_; }
  const std::vector<std::string>& Profiles() const { return profile_names_; }

  const std::vector<SecondaryDevice>& SecondaryDevices() const
  {
    return secondary_devices_;
  }

  Status Initialize();
  Status WarmUp();
  Status Schedule(std::vector<std::unique_ptr<InferenceRequest>>&& requests);

  TritonModel* Model() const { return model_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  MetricModelReporter* MetricReporter() const { return reporter_.get(); }

  // Directly call from C API, so arguments are in the same style
  void SetMemoryUsage(boost::span<BufferAttributes*> memory_usage)
  {
    std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>> lusage;
    for (const auto& mu : memory_usage) {
      lusage[mu->MemoryType()][mu->MemoryTypeId()] = mu->ByteSize();
    }
    std::lock_guard<std::mutex> lk(usage_mtx_);
    memory_usage_.swap(lusage);
  }

  std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>> MemoryUsage()
      const
  {
    std::lock_guard<std::mutex> lk(usage_mtx_);
    return memory_usage_;
  }

  static bool ShareBackendThread(
      const bool device_blocking, const TRITONSERVER_InstanceGroupKind kind)
  {
    return device_blocking && (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU);
  }

 private:
  class TritonBackendThread {
   public:
    static Status CreateBackendThread(
        const std::string name, TritonModelInstance* model, const int nice,
        const int32_t device_id,
        std::unique_ptr<TritonBackendThread>* triton_backend_thread);
    void AddModelInstance(TritonModelInstance* model_instance);
    Status InitAndWarmUpModelInstance(TritonModelInstance* model_instance);
    void StopBackendThread();
    ~TritonBackendThread();

   private:
    TritonBackendThread(
        const std::string& name, TritonModel* model, const int nice,
        const int32_t device_id);
    void BackendThread();

    const std::string name_;
    const int nice_;
    const int32_t device_id_;

    TritonModel* model_;
    std::deque<TritonModelInstance*> model_instances_;

    std::thread backend_thread_;
    std::atomic<bool> backend_thread_exit_;
  };

  struct WarmupData {
    WarmupData(const std::string& sample_name, const size_t count)
        : sample_name_(sample_name), count_(std::max(count, size_t{1}))
    {
    }

    std::string sample_name_;
    size_t count_;
    // Using a batch of requests to satisfy batch size, this provides better
    // alignment on the batch expected by the model, especially for sequence
    // model.
    std::vector<std::unique_ptr<InferenceRequest>> requests_;

    // Placeholder for input data
    std::unique_ptr<AllocatedMemory> zero_data_;
    std::unique_ptr<AllocatedMemory> random_data_;
    std::vector<std::unique_ptr<std::string>> provided_data_;
  };

  DISALLOW_COPY_AND_ASSIGN(TritonModelInstance);
  TritonModelInstance(
      TritonModel* model, const std::string& name, const Signature& signature,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const triton::common::HostPolicyCmdlineConfig& host_policy,
      const TritonServerMessage& host_policy_message,
      const std::vector<SecondaryDevice>& secondary_devices);
  static Status ConstructAndInitializeInstance(
      TritonModel* model, const std::string& name, const Signature& signature,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const std::string& host_policy_name,
      const triton::common::HostPolicyCmdlineConfig& host_policy,
      const inference::ModelRateLimiter& rate_limiter_config,
      const std::vector<SecondaryDevice>& secondary_devices,
      std::shared_ptr<TritonModelInstance>* triton_model_instance);
  Status SetBackendThread(
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const bool device_blocking);
  Status GenerateWarmupData();

  Status PrepareRequestsForExecution(
      std::vector<std::unique_ptr<InferenceRequest>>& requests);
  Status PrepareRequestsOrRespond(
      std::vector<std::unique_ptr<InferenceRequest>>& requests);
  void Execute(std::vector<TRITONBACKEND_Request*>& triton_requests);

  std::shared_ptr<TritonBackendThread> triton_backend_thread_;

  std::vector<WarmupData> warmup_samples_;

  // The TritonModel object that owns this instance. The instance
  // holds this as a raw pointer because the lifetime of the model is
  // guaranteed to be longer than the lifetime of an instance owned by the
  // model.
  TritonModel* model_;

  std::string name_;
  Signature signature_;

  // For CPU device_id_ is always 0. For GPU device_id_ indicates the
  // GPU device to be used by the instance.
  TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;
  const triton::common::HostPolicyCmdlineConfig host_policy_;
  TritonServerMessage host_policy_message_;
  std::vector<std::string> profile_names_;
  bool passive_;

  std::vector<SecondaryDevice> secondary_devices_;

  // Reporter for metrics, or nullptr if no metrics should be reported
  std::shared_ptr<MetricModelReporter> reporter_;

  // Records of memory used for the model instance
  std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>> memory_usage_;
  mutable std::mutex usage_mtx_;

  // Opaque state associated with this model instance.
  void* state_;
};

}}  // namespace triton::core

// Interface for triton::core::TritonModelInstance::Signature hash function
namespace std {
template <>
struct hash<triton::core::TritonModelInstance::Signature> {
  std::size_t operator()(
      const triton::core::TritonModelInstance::Signature& s) const
  {
    return s.Hash();
  }
};
}  // namespace std
