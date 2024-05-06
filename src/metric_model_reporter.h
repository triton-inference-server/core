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
#pragma once

#include "status.h"
#include "triton/common/model_config.h"

#ifdef TRITON_ENABLE_METRICS
#include "metrics.h"
#include "model.h"
#include "prometheus/registry.h"
#endif  // TRITON_ENABLE_METRICS

namespace triton { namespace core {

#ifdef TRITON_ENABLE_METRICS
struct ModelIdentifier;
#endif  // TRITON_ENABLE_METRICS

//
// MetricReporterConfig
//
struct MetricReporterConfig {
#ifdef TRITON_ENABLE_METRICS
  // Parses Metrics::ConfigMap and sets fields if specified
  void ParseConfig(bool response_cache_enabled);
  // Parses pairs of quantiles "quantile1:error1, quantile2:error2, ..."
  // and overwrites quantiles_ field if successful.
  prometheus::Summary::Quantiles ParseQuantiles(std::string options);

  // Create and use Counters for per-model latency related metrics
  bool latency_counters_enabled_ = true;
  // Create and use Summaries for per-model latency related metrics
  bool latency_summaries_enabled_ = false;
  // Quantiles used for any summary metrics. Each pair of values represents
  // { quantile, error }. For example, {0.90, 0.01} means to compute the
  // 90th percentile with 1% error on either side, so the approximate 90th
  // percentile value will be between the 89th and 91st percentiles.
  prometheus::Summary::Quantiles quantiles_ = {
      {0.5, 0.05}, {0.9, 0.01}, {0.95, 0.001}, {0.99, 0.001}, {0.999, 0.001}};

  // Whether this reporter's model has caching enabled or not.
  // This helps handle infer_stats aggregation for summaries on cache misses.
  bool cache_enabled_ = false;
#endif  // TRITON_ENABLE_METRICS
};

//
// Interface for a metric reporter for a given version of a model.
//
class MetricModelReporter {
 public:
#ifdef TRITON_ENABLE_METRICS
  static Status Create(
      const triton::core::ModelIdentifier& model_id,
      const int64_t model_version, const int device,
      bool response_cache_enabled,
      const triton::common::MetricTagsMap& model_tags,
      std::shared_ptr<MetricModelReporter>* metric_model_reporter);

  ~MetricModelReporter();

  // Get this reporter's config
  const MetricReporterConfig& Config();
  // Lookup counter metric by name, and increment it by value if it exists.
  void IncrementCounter(const std::string& name, double value);
  // Increase gauge by value.
  void IncrementGauge(const std::string& name, double value);
  // Decrease gauge by value.
  void DecrementGauge(const std::string& name, double value);
  // Lookup summary metric by name, and observe the value if it exists.
  void ObserveSummary(const std::string& name, double value);

 private:
  MetricModelReporter(
      const ModelIdentifier& model_id, const int64_t model_version,
      const int device, bool response_cache_enabled,
      const triton::common::MetricTagsMap& model_tags);

  static void GetMetricLabels(
      std::map<std::string, std::string>* labels,
      const ModelIdentifier& model_id, const int64_t model_version,
      const int device, const triton::common::MetricTagsMap& model_tags);

  template <typename T, typename... Args>
  T* CreateMetric(
      prometheus::Family<T>& family,
      const std::map<std::string, std::string>& labels, Args&&... args);

  void InitializeCounters(const std::map<std::string, std::string>& labels);
  void InitializeGauges(const std::map<std::string, std::string>& labels);
  void InitializeSummaries(const std::map<std::string, std::string>& labels);

  // Lookup gauge metric by name. Return gauge if found, nullptr otherwise.
  prometheus::Gauge* GetGauge(const std::string& name);


  // Metric Families
  std::unordered_map<std::string, prometheus::Family<prometheus::Counter>*>
      counter_families_;
  std::unordered_map<std::string, prometheus::Family<prometheus::Gauge>*>
      gauge_families_;
  std::unordered_map<std::string, prometheus::Family<prometheus::Summary>*>
      summary_families_;

  // Metrics
  std::unordered_map<std::string, prometheus::Counter*> counters_;
  std::unordered_map<std::string, prometheus::Gauge*> gauges_;
  std::unordered_map<std::string, prometheus::Summary*> summaries_;

  // Config
  MetricReporterConfig config_;
#endif  // TRITON_ENABLE_METRICS
};

}}  // namespace triton::core
