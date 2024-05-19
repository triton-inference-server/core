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

#include <boost/core/span.hpp>

#include "infer_stats.h"
#include "label_provider.h"
#include "metric_model_reporter.h"
#include "model_config.pb.h"
#include "scheduler.h"
#include "status.h"

namespace triton { namespace core {
struct ModelIdentifier {
  ModelIdentifier(const std::string& model_namespace, const std::string& name)
      : namespace_(model_namespace), name_(name)
  {
  }

  ModelIdentifier(const ModelIdentifier&) = default;

  ModelIdentifier& operator=(const ModelIdentifier&) = default;

  bool operator<(const ModelIdentifier& rhs) const
  {
    if (namespace_ == rhs.namespace_) {
      return (name_ < rhs.name_);
    } else {
      return (namespace_ < rhs.namespace_);
    }
  }

  bool operator==(const ModelIdentifier& rhs) const
  {
    return ((namespace_ == rhs.namespace_) && (name_ == rhs.name_));
  }

  bool operator!=(const ModelIdentifier& rhs) const
  {
    return ((namespace_ != rhs.namespace_) || (name_ != rhs.name_));
  }

  bool NamespaceDisabled() const { return namespace_.empty(); }

  friend std::ostream& operator<<(std::ostream& os, const ModelIdentifier& rhs)
  {
    // Avoid log differences if namespace is disabled
    if (rhs.NamespaceDisabled()) {
      os << rhs.name_;
      return os;
    }
    os << rhs.namespace_ << "::" << rhs.name_;
    return os;
  }

  std::string str() const
  {
    // Avoid log differences if namespace is disabled
    if (NamespaceDisabled()) {
      return name_;
    }
    return (namespace_ + "::" + name_);
  }

  // namespace is not a reflection of the model repository although it is
  // currently implemented to be the same as the repository of the model.
  std::string namespace_;
  // name is the name registered to Triton, it is the model directory name
  // by default and may be overwritten.
  std::string name_;
};
}}  // namespace triton::core

// define hash function for struct to be used as unordered_map key
namespace std {
template <>
class hash<triton::core::ModelIdentifier> {
 public:
  size_t operator()(const triton::core::ModelIdentifier& model_id) const
  {
    // trivial hash for multiple entries
    // https://en.cppreference.com/w/cpp/utility/hash
    return (
        hash<std::string>()(model_id.namespace_) ^
        (hash<std::string>()(model_id.name_) << 1));
  }
};
}  // namespace std
namespace triton { namespace core {

class InferenceRequest;

//
// Interface for models that handle inference requests.
//
class Model {
 public:
  explicit Model(
      const double min_compute_capability, const std::string& model_dir,
      const ModelIdentifier& model_id, const int64_t version,
      const inference::ModelConfig& config)
      : config_(config), min_compute_capability_(min_compute_capability),
        model_id_(model_id), version_(version), required_input_count_(0),
        model_dir_(model_dir), set_model_config_(false)
  {
  }
  virtual ~Model() {}

  // Get the name of model being served.
  const std::string& Name() const { return config_.name(); }

  // Get the identifier of model being served.
  const ModelIdentifier& ModelId() const { return model_id_; }

  // Get the version of model being served.
  int64_t Version() const { return version_; }

  // Get the configuration of model being served.
  const inference::ModelConfig& Config() const { return config_; }

  // Get whether response cache is enabled for this model.
  bool ResponseCacheEnabled() const
  {
    return config_.response_cache().enable();
  }

  // Get the number of required inputs
  size_t RequiredInputCount() const { return required_input_count_; }

  // Get the stats collector for the model being served.
  InferenceStatsAggregator* MutableStatsAggregator()
  {
    return &stats_aggregator_;
  }
  const InferenceStatsAggregator& StatsAggregator() const
  {
    return stats_aggregator_;
  }

  void SetMemoryUsage(boost::span<BufferAttributes*> memory_usage)
  {
    std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>> lusage;
    for (const auto& mu : memory_usage) {
      lusage[mu->MemoryType()][mu->MemoryTypeId()] = mu->ByteSize();
    }
    std::lock_guard<std::mutex> lk(usage_mtx_);
    memory_usage_.swap(lusage);
  }

  std::vector<BufferAttributes> AccumulatedMemoryUsage() const
  {
    auto lusage = AccumulatedInstanceMemoryUsage();
    // accumulate model memory usage
    {
      std::lock_guard<std::mutex> lk(usage_mtx_);
      for (const auto& mem_type_map : memory_usage_) {
        const auto& mem_type = mem_type_map.first;
        for (const auto& mem_id_map : mem_type_map.second) {
          const auto& mem_id = mem_id_map.first;
          const auto& byte_size = mem_id_map.second;
          lusage[mem_type][mem_id] += byte_size;
        }
      }
    }
    // Convert to buffer attribute
    std::vector<BufferAttributes> res;
    for (const auto& mem_type_map : lusage) {
      const auto& mem_type = mem_type_map.first;
      for (const auto& mem_id_map : mem_type_map.second) {
        const auto& mem_id = mem_id_map.first;
        const auto& byte_size = mem_id_map.second;
        res.emplace_back(byte_size, mem_type, mem_id, nullptr);
      }
    }
    return res;
  }

  // Get the model configuration for a named input.
  Status GetInput(
      const std::string& name, const inference::ModelInput** input) const;

  // Get the model configuration for a named output.
  Status GetOutput(
      const std::string& name, const inference::ModelOutput** output) const;

  // Get a label provider for the model.
  const std::shared_ptr<LabelProvider>& GetLabelProvider() const
  {
    return label_provider_;
  }

  // Initialize the instance for Triton core usage
  Status Init(const bool is_config_provided);

  // Enqueue a request for execution. If Status::Success is returned
  // then the model has taken ownership of the request object and so
  // 'request' will be nullptr. If non-success is returned then the
  // caller still retains ownership of 'request'.
  Status Enqueue(std::unique_ptr<InferenceRequest>& request)
  {
    return scheduler_->Enqueue(request);
  }

  // Return the number of in-flight inferences.
  size_t InflightInferenceCount()
  {
    return scheduler_->InflightInferenceCount();
  }

  // Stop processing future requests unless they are considered as in-flight.
  void Stop() { scheduler_->Stop(); }

  uint64_t DefaultPriorityLevel() const { return default_priority_level_; }

  uint64_t MaxPriorityLevel() const { return max_priority_level_; }

  // Returns the model's metric reporter
  std::shared_ptr<MetricModelReporter> MetricReporter() const
  {
    return reporter_;
  }

 protected:
  virtual std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>>
  AccumulatedInstanceMemoryUsage() const
  {
    return {};
  }

  // Set the configuration of the model being served.
  Status SetModelConfig(const inference::ModelConfig& config);

  // Explicitly set the scheduler to use for inference requests to the
  // model. The scheduler can only be set once for a model.
  Status SetScheduler(std::unique_ptr<Scheduler> scheduler);

  // The scheduler to use for this model.
  std::unique_ptr<Scheduler> scheduler_;

  // Configuration of the model.
  inference::ModelConfig config_;

 private:
  // The minimum supported CUDA compute capability.
  const double min_compute_capability_;

  // Identifier for the model.
  ModelIdentifier model_id_;

  // Version of the model.
  int64_t version_;

  // The stats collector for the model.
  InferenceStatsAggregator stats_aggregator_;

  // Records of memory used for loading the model
  std::map<TRITONSERVER_MemoryType, std::map<int64_t, size_t>> memory_usage_;
  mutable std::mutex usage_mtx_;

  // Label provider for this model.
  std::shared_ptr<LabelProvider> label_provider_;

  size_t required_input_count_;

  // Map from input name to the model configuration for that input.
  std::unordered_map<std::string, inference::ModelInput> input_map_;

  // Map from output name to the model configuration for that output.
  std::unordered_map<std::string, inference::ModelOutput> output_map_;

  // Path to model
  std::string model_dir_;

  // The default priority level for the model.
  uint64_t default_priority_level_;

  // The largest priority value for the model.
  uint64_t max_priority_level_;

  // Whether or not model config has been set.
  bool set_model_config_;

  // Reporter for metrics, or nullptr if no metrics should be reported
  std::shared_ptr<MetricModelReporter> reporter_;
};

}}  // namespace triton::core
