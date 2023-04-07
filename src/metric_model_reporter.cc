// Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "metric_model_reporter.h"

#ifdef TRITON_ENABLE_METRICS

#include "constants.h"
#include "metrics.h"

// Global config group has 'name' of empty string.
constexpr char GLOBAL_CONFIG_GROUP[] = "";

namespace triton { namespace core {

//
// MetricReporterConfig
//
void
MetricReporterConfig::ParseConfig()
{
  // Global config only for now in config map
  auto metrics_config_map = Metrics::ConfigMap();
  const auto& metrics_config = metrics_config_map[GLOBAL_CONFIG_GROUP];

  // Default behavior is counters for most latency metrics if no types specified
  for (const auto& pair : metrics_config) {
    if (pair.first == "latency_counters" && pair.second == "false") {
      enable_latency_counters_ = false;
    }

    if (pair.first == "latency_summaries" && pair.second == "true") {
      enable_latency_summaries_ = true;
    }
  }
}

//
// MetricModelReporter
//
Status
MetricModelReporter::Create(
    const std::string& model_name, const int64_t model_version,
    const int device, const triton::common::MetricTagsMap& model_tags,
    std::shared_ptr<MetricModelReporter>* metric_model_reporter)
{
  static std::mutex mtx;
  static std::unordered_map<size_t, std::weak_ptr<MetricModelReporter>>
      reporter_map;

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_name, model_version, device, model_tags);
  auto hash_labels = Metrics::HashLabels(labels);

  std::lock_guard<std::mutex> lock(mtx);

  const auto& itr = reporter_map.find(hash_labels);
  if (itr != reporter_map.end()) {
    // Found in map. If the weak_ptr is still valid that means that
    // there are other models using the reporter and we just reuse that
    // same reporter. If the weak_ptr is not valid then we need to remove
    // the weak_ptr from the map and create the reporter again.
    *metric_model_reporter = itr->second.lock();
    if (*metric_model_reporter != nullptr) {
      return Status::Success;
    }

    reporter_map.erase(itr);
  }

  metric_model_reporter->reset(
      new MetricModelReporter(model_name, model_version, device, model_tags));
  reporter_map.insert({hash_labels, *metric_model_reporter});
  return Status::Success;
}

MetricModelReporter::MetricModelReporter(
    const std::string& model_name, const int64_t model_version,
    const int device, const triton::common::MetricTagsMap& model_tags)
{
  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_name, model_version, device, model_tags);

  // Parse metrics config to control metric setup and behavior
  config_.ParseConfig();
  // Initialize families and metrics
  InitializeCounters(labels);
  InitializeSummaries(labels);
}

MetricModelReporter::~MetricModelReporter()
{
  // Cleanup metrics for each family
  for (auto& iter : counter_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      family_ptr->Remove(counters_[name]);
    }
  }

  for (auto& iter : summary_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      family_ptr->Remove(summaries_[name]);
    }
  }
}

void
MetricModelReporter::InitializeCounters(
    const std::map<std::string, std::string>& labels)
{
  // Always setup these counters, regardless of config
  counter_families_["inf_success"] = &Metrics::FamilyInferenceSuccess();
  counter_families_["inf_failure"] = &Metrics::FamilyInferenceFailure();
  counter_families_["inf_count"] = &Metrics::FamilyInferenceCount();
  counter_families_["inf_exec_count"] =
      &Metrics::FamilyInferenceExecutionCount();

  // Latency metrics will be initialized based on config
  if (config_.enable_latency_counters_) {
    // Request
    counter_families_["request_duration"] =
        &Metrics::FamilyInferenceRequestDuration();
    counter_families_["queue_duration"] =
        &Metrics::FamilyInferenceQueueDuration();
    // Compute
    counter_families_["compute_input_duration"] =
        &Metrics::FamilyInferenceComputeInputDuration();
    counter_families_["compute_infer_duration"] =
        &Metrics::FamilyInferenceComputeInferDuration();
    counter_families_["compute_output_duration"] =
        &Metrics::FamilyInferenceComputeOutputDuration();
    // Cache
    counter_families_["cache_hit_count"] = &Metrics::FamilyCacheHitCount();
    counter_families_["cache_miss_count"] = &Metrics::FamilyCacheMissCount();
    counter_families_["cache_hit_duration"] =
        &Metrics::FamilyCacheHitDuration();
    counter_families_["cache_miss_duration"] =
        &Metrics::FamilyCacheMissDuration();
  }

  // Create metrics for each family
  for (auto& iter : counter_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      counters_[name] = CreateMetric<prometheus::Counter>(*family_ptr, labels);
    }
  }
}

void
MetricModelReporter::InitializeSummaries(
    const std::map<std::string, std::string>& labels)
{
  // Latency metrics will be initialized based on config
  if (config_.enable_latency_summaries_) {
    // Request
    summary_families_["request_duration"] =
        &Metrics::FamilyInferenceRequestSummary();
    summary_families_["queue_duration"] =
        &Metrics::FamilyInferenceQueueSummary();
    // Compute
    summary_families_["compute_input_duration"] =
        &Metrics::FamilyInferenceComputeInputSummary();
    summary_families_["compute_infer_duration"] =
        &Metrics::FamilyInferenceComputeInferSummary();
    summary_families_["compute_output_duration"] =
        &Metrics::FamilyInferenceComputeOutputSummary();
    // Cache: Note that counts are included in summary
    summary_families_["cache_hit_duration"] = &Metrics::FamilyCacheHitSummary();
    summary_families_["cache_miss_duration"] =
        &Metrics::FamilyCacheMissSummary();
  }

  // FIXME: Expose quantiles in --metrics-config
  const auto quantiles = prometheus::Summary::Quantiles{
      {0.5, 0.05}, {0.9, 0.01}, {0.95, 0.001}, {0.99, 0.001}, {0.999, 0.001}};

  // Create metrics for each family
  for (auto& iter : summary_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      summaries_[name] =
          CreateMetric<prometheus::Summary>(*family_ptr, labels, quantiles);
    }
  }
}

void
MetricModelReporter::GetMetricLabels(
    std::map<std::string, std::string>* labels, const std::string& model_name,
    const int64_t model_version, const int device,
    const triton::common::MetricTagsMap& model_tags)
{
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), model_name));
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelVersion), std::to_string(model_version)));
  for (const auto& tag : model_tags) {
    labels->insert(std::map<std::string, std::string>::value_type(
        "_" + tag.first, tag.second));
  }

  // 'device' can be < 0 to indicate that the GPU is not known. In
  // that case use a metric that doesn't have the gpu_uuid label.
  if (device >= 0) {
    std::string uuid;
    if (Metrics::UUIDForCudaDevice(device, &uuid)) {
      labels->insert(std::map<std::string, std::string>::value_type(
          std::string(kMetricsLabelGpuUuid), uuid));
    }
  }
}

template <typename T, typename... Args>
T*
MetricModelReporter::CreateMetric(
    prometheus::Family<T>& family,
    const std::map<std::string, std::string>& labels, Args&&... args)
{
  return &family.Add(labels, args...);
}

void
MetricModelReporter::IncrementCounter(const std::string& name, double value)
{
  auto iter = counters_.find(name);
  if (iter == counters_.end()) {
    // No counter metric exists with this name
    return;
  }

  auto counter = iter->second;
  if (!counter) {
    // Counter is uninitialized/nullptr
    return;
  }
  counter->Increment(value);
}

void
MetricModelReporter::ObserveSummary(const std::string& name, double value)
{
  auto iter = summaries_.find(name);
  if (iter == summaries_.end()) {
    // No summary metric exists with this name
    return;
  }

  auto summary = iter->second;
  if (!summary) {
    // Summary is uninitialized/nullptr
    return;
  }
  summary->Observe(value);
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METRICS
