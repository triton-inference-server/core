// Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "infer_stats.h"
#include "triton/common/logging.h"

// Global config group has 'name' of empty string.
constexpr char GLOBAL_CONFIG_GROUP[] = "";

namespace triton { namespace core {

//
// MetricReporterConfig
//
void
MetricReporterConfig::ParseConfig(
    bool response_cache_enabled, bool is_decoupled,
    const inference::ModelMetrics& model_metrics)
{
  // Global config only for now in config map
  auto metrics_config_map = Metrics::ConfigMap();
  const auto& metrics_config = metrics_config_map[GLOBAL_CONFIG_GROUP];

  // Default behavior is counters for most latency metrics if no types specified
  for (const auto& pair : metrics_config) {
    if (pair.first == "counter_latencies" && pair.second == "false") {
      latency_counters_enabled_ = false;
    }

    if (pair.first == "histogram_latencies" && pair.second == "true") {
      latency_histograms_enabled_ = true;
    }

    if (pair.first == "summary_latencies" && pair.second == "true") {
      latency_summaries_enabled_ = true;
    }

    // ex: summary_quantiles="0.5:0.05 0.9:0.01 0.99:0.001"
    if (pair.first == "summary_quantiles") {
      const auto& quantiles = ParseQuantiles(pair.second);
      if (!quantiles.empty()) {
        quantiles_ = quantiles;
      }
    }
  }

  // Set flag to signal to stats aggregator if caching is enabled or not
  cache_enabled_ = response_cache_enabled;
  is_decoupled_ = is_decoupled;

  // Override default histogram options if set in model_metrics.
  for (const auto& metric_control : model_metrics.metric_control()) {
    const std::string& family_name =
        metric_control.metric_identifier().family();

    // If family name exists, override with new options.
    if (metric_map_.find(family_name) != metric_map_.end()) {
      // Copy protobuf RepeatedField to std::vector
      const auto& buckets_proto = metric_control.histogram_options().buckets();
      const prometheus::Histogram::BucketBoundaries buckets(
          buckets_proto.begin(), buckets_proto.end());
      histogram_options_[metric_map_.at(family_name)] = buckets;
    } else {
      // metric_control config may be extended to support backend metrics.
      LOG_WARNING << "Metric family '" << family_name
                  << "' in 'metric_identifier' is not a customizable metric in "
                     "Triton core.";
    }
  }
}

prometheus::Summary::Quantiles
MetricReporterConfig::ParseQuantiles(std::string options)
{
  prometheus::Summary::Quantiles qpairs;
  std::stringstream ss(options);
  std::string pairStr;
  while (std::getline(ss, pairStr, ',')) {
    size_t colonPos = pairStr.find(':');
    if (colonPos == std::string::npos) {
      LOG_ERROR
          << "Invalid option: [" << pairStr
          << "]. No ':' delimiter found. Expected format is <quantile>:<error>";
      continue;
    }

    try {
      double quantile = std::stod(pairStr.substr(0, colonPos));
      double error = std::stod(pairStr.substr(colonPos + 1));
      qpairs.push_back({quantile, error});
    }
    catch (const std::invalid_argument& e) {
      LOG_ERROR << "Invalid option: [" << pairStr << "]. Error: " << e.what();
      continue;
    }
  }

  return qpairs;
}

//
// MetricModelReporter
//
const std::map<FailureReason, std::string>
    MetricModelReporter::failure_reasons_map = {
        {FailureReason::REJECTED, "REJECTED"},
        {FailureReason::CANCELED, "CANCELED"},
        {FailureReason::BACKEND, "BACKEND"},
        {FailureReason::OTHER, "OTHER"}};

Status
MetricModelReporter::Create(
    const ModelIdentifier& model_id, const int64_t model_version,
    const int device, bool response_cache_enabled, bool is_decoupled,
    const triton::common::MetricTagsMap& model_tags,
    const inference::ModelMetrics& model_metrics,
    std::shared_ptr<MetricModelReporter>* metric_model_reporter)
{
  static std::mutex mtx;
  static std::unordered_map<size_t, std::weak_ptr<MetricModelReporter>>
      reporter_map;

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_id, model_version, device, model_tags);
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

  metric_model_reporter->reset(new MetricModelReporter(
      model_id, model_version, device, response_cache_enabled, is_decoupled,
      model_tags, model_metrics));
  reporter_map.insert({hash_labels, *metric_model_reporter});
  return Status::Success;
}

MetricModelReporter::MetricModelReporter(
    const ModelIdentifier& model_id, const int64_t model_version,
    const int device, bool response_cache_enabled, bool is_decoupled,
    const triton::common::MetricTagsMap& model_tags,
    const inference::ModelMetrics& model_metrics)
{
  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_id, model_version, device, model_tags);

  // Parse metrics config to control metric setup and behavior
  config_.ParseConfig(response_cache_enabled, is_decoupled, model_metrics);

  // Initialize families and metrics
  InitializeCounters(labels);
  InitializeGauges(labels);
  InitializeHistograms(labels);
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

  for (auto& iter : gauge_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      family_ptr->Remove(gauges_[name]);
    }
  }

  for (auto& iter : histogram_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      family_ptr->Remove(histograms_[name]);
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
  counter_families_["inf_count"] = &Metrics::FamilyInferenceCount();
  counter_families_["inf_exec_count"] =
      &Metrics::FamilyInferenceExecutionCount();

  // Latency metrics will be initialized based on config
  if (config_.latency_counters_enabled_) {
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
    // Only create cache metrics if cache is enabled to reduce metric output
    if (config_.cache_enabled_) {
      counter_families_["cache_hit_count"] = &Metrics::FamilyCacheHitCount();
      counter_families_["cache_miss_count"] = &Metrics::FamilyCacheMissCount();
      counter_families_["cache_hit_duration"] =
          &Metrics::FamilyCacheHitDuration();
      counter_families_["cache_miss_duration"] =
          &Metrics::FamilyCacheMissDuration();
    }
  }

  // Create metrics for each family
  for (auto& iter : counter_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      counters_[name] = CreateMetric<prometheus::Counter>(*family_ptr, labels);
    }
  }

  // Initialize failure metrics with reasons
  for (const auto& reason_pair : failure_reasons_map) {
    std::map<std::string, std::string> extended_labels = labels;
    extended_labels["reason"] = reason_pair.second;
    counters_["inf_failure_" + reason_pair.second] =
        CreateMetric<prometheus::Counter>(
            Metrics::FamilyInferenceFailure(), extended_labels);
  }
}

void
MetricModelReporter::InitializeGauges(
    const std::map<std::string, std::string>& labels)
{
  // Always setup these inference request metrics, regardless of config
  gauge_families_[kPendingRequestMetric] = &Metrics::FamilyInferenceQueueSize();
  gauge_families_[kModelLoadTimeMetric] = &Metrics::FamilyModelLoadTime();

  for (auto& iter : gauge_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      gauges_[name] = CreateMetric<prometheus::Gauge>(*family_ptr, labels);
    }
  }
}

void
MetricModelReporter::InitializeHistograms(
    const std::map<std::string, std::string>& labels)
{
  // Update MetricReporterConfig::metric_map_ for new histograms.
  // Only create response metrics if decoupled model to reduce metric output
  if (config_.latency_histograms_enabled_) {
    if (config_.is_decoupled_) {
      histogram_families_[kFirstResponseHistogram] =
          &Metrics::FamilyFirstResponseDuration();
    }
  }

  for (auto& iter : histogram_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      const auto& buckets = config_.histogram_options_[name];
      histograms_[name] =
          CreateMetric<prometheus::Histogram>(*family_ptr, labels, buckets);
    }
  }
}

void
MetricModelReporter::InitializeSummaries(
    const std::map<std::string, std::string>& labels)
{
  // Latency metrics will be initialized based on config
  if (config_.latency_summaries_enabled_) {
    // Request
    if (!config_.cache_enabled_) {
      // FIXME: request_duration summary is currently disabled when cache is
      // enabled to avoid publishing misleading metrics.
      summary_families_["request_duration"] =
          &Metrics::FamilyInferenceRequestSummary();
    }
    summary_families_["queue_duration"] =
        &Metrics::FamilyInferenceQueueSummary();
    // Compute
    summary_families_["compute_input_duration"] =
        &Metrics::FamilyInferenceComputeInputSummary();
    summary_families_["compute_infer_duration"] =
        &Metrics::FamilyInferenceComputeInferSummary();
    summary_families_["compute_output_duration"] =
        &Metrics::FamilyInferenceComputeOutputSummary();
    // Only create cache metrics if cache is enabled to reduce metric output
    if (config_.cache_enabled_) {
      // Note that counts and sums are included in summaries
      summary_families_["cache_hit_duration"] =
          &Metrics::FamilyCacheHitSummary();
      summary_families_["cache_miss_duration"] =
          &Metrics::FamilyCacheMissSummary();
    }
  }

  // Create metrics for each family
  for (auto& iter : summary_families_) {
    const auto& name = iter.first;
    auto family_ptr = iter.second;
    if (family_ptr) {
      summaries_[name] = CreateMetric<prometheus::Summary>(
          *family_ptr, labels, config_.quantiles_);
    }
  }
}

void
MetricModelReporter::GetMetricLabels(
    std::map<std::string, std::string>* labels, const ModelIdentifier& model_id,
    const int64_t model_version, const int device,
    const triton::common::MetricTagsMap& model_tags)
{
  if (!model_id.NamespaceDisabled()) {
    labels->insert(std::map<std::string, std::string>::value_type(
        std::string(kMetricsLabelModelNamespace), model_id.namespace_));
  }
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), model_id.name_));
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

const MetricReporterConfig&
MetricModelReporter::Config()
{
  return config_;
}

void
MetricModelReporter::IncrementCounter(const std::string& name, double value)
{
  if (!config_.latency_counters_enabled_) {
    return;
  }

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

prometheus::Gauge*
MetricModelReporter::GetGauge(const std::string& name)
{
  auto iter = gauges_.find(name);
  if (iter == gauges_.end()) {
    // No gauge metric exists with this name
    return nullptr;
  }

  auto gauge = iter->second;
  return gauge;
}

void
MetricModelReporter::IncrementGauge(const std::string& name, double value)
{
  auto gauge = GetGauge(name);
  if (gauge) {
    gauge->Increment(value);
  }
}

void
MetricModelReporter::SetGauge(const std::string& name, double value)
{
  auto gauge = GetGauge(name);
  if (gauge) {
    gauge->Set(value);
  }
}

void
MetricModelReporter::DecrementGauge(const std::string& name, double value)
{
  IncrementGauge(name, -1 * value);
}

void
MetricModelReporter::ObserveHistogram(const std::string& name, double value)
{
  auto iter = histograms_.find(name);
  if (iter == histograms_.end()) {
    // No histogram metric exists with this name
    return;
  }

  auto histogram = iter->second;
  if (!histogram) {
    // histogram is uninitialized/nullptr
    return;
  }
  histogram->Observe(value);
}

void
MetricModelReporter::ObserveSummary(const std::string& name, double value)
{
  if (!config_.latency_summaries_enabled_) {
    return;
  }

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
