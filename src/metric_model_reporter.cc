// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace triton { namespace core {

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

  metric_inf_success_ =
      CreateCounterMetric(Metrics::FamilyInferenceSuccess(), labels);
  metric_inf_failure_ =
      CreateCounterMetric(Metrics::FamilyInferenceFailure(), labels);
  metric_inf_count_ =
      CreateCounterMetric(Metrics::FamilyInferenceCount(), labels);
  metric_inf_exec_count_ =
      CreateCounterMetric(Metrics::FamilyInferenceExecutionCount(), labels);
  metric_inf_request_duration_us_ =
      CreateCounterMetric(Metrics::FamilyInferenceRequestDuration(), labels);
  metric_inf_queue_duration_us_ =
      CreateCounterMetric(Metrics::FamilyInferenceQueueDuration(), labels);
  metric_inf_compute_input_duration_us_ = CreateCounterMetric(
      Metrics::FamilyInferenceComputeInputDuration(), labels);
  metric_inf_compute_infer_duration_us_ = CreateCounterMetric(
      Metrics::FamilyInferenceComputeInferDuration(), labels);
  metric_inf_compute_output_duration_us_ = CreateCounterMetric(
      Metrics::FamilyInferenceComputeOutputDuration(), labels);
  metric_cache_hit_count_ =
      CreateCounterMetric(Metrics::FamilyCacheHitCount(), labels);
  metric_cache_hit_lookup_duration_us_ =
      CreateCounterMetric(Metrics::FamilyCacheHitLookupDuration(), labels);
  metric_cache_miss_count_ =
      CreateCounterMetric(Metrics::FamilyCacheMissCount(), labels);
  metric_cache_miss_lookup_duration_us_ =
      CreateCounterMetric(Metrics::FamilyCacheMissLookupDuration(), labels);
  metric_cache_miss_insertion_duration_us_ =
      CreateCounterMetric(Metrics::FamilyCacheMissInsertionDuration(), labels);
}

MetricModelReporter::~MetricModelReporter()
{
  Metrics::FamilyInferenceSuccess().Remove(metric_inf_success_);
  Metrics::FamilyInferenceFailure().Remove(metric_inf_failure_);
  Metrics::FamilyInferenceCount().Remove(metric_inf_count_);
  Metrics::FamilyInferenceExecutionCount().Remove(metric_inf_exec_count_);
  Metrics::FamilyInferenceRequestDuration().Remove(
      metric_inf_request_duration_us_);
  Metrics::FamilyInferenceQueueDuration().Remove(metric_inf_queue_duration_us_);
  Metrics::FamilyInferenceComputeInputDuration().Remove(
      metric_inf_compute_input_duration_us_);
  Metrics::FamilyInferenceComputeInferDuration().Remove(
      metric_inf_compute_infer_duration_us_);
  Metrics::FamilyInferenceComputeOutputDuration().Remove(
      metric_inf_compute_output_duration_us_);
  Metrics::FamilyCacheHitCount().Remove(metric_cache_hit_count_);
  Metrics::FamilyCacheHitLookupDuration().Remove(
      metric_cache_hit_lookup_duration_us_);
  Metrics::FamilyCacheMissCount().Remove(metric_cache_miss_count_);
  Metrics::FamilyCacheMissInsertionDuration().Remove(
      metric_cache_miss_insertion_duration_us_);
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

prometheus::Counter*
MetricModelReporter::CreateCounterMetric(
    prometheus::Family<prometheus::Counter>& family,
    const std::map<std::string, std::string>& labels)
{
  return &family.Add(labels);
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METRICS
