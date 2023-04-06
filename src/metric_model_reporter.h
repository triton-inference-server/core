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
#pragma once

#include "status.h"
#include "triton/common/model_config.h"

#ifdef TRITON_ENABLE_METRICS
#include "prometheus/registry.h"
#endif  // TRITON_ENABLE_METRICS

namespace triton { namespace core {

//
// Interface for a metric reporter for a given version of a model.
//
class MetricModelReporter {
 public:
#ifdef TRITON_ENABLE_METRICS
  static Status Create(
      const std::string& model_name, const int64_t model_version,
      const int device, const triton::common::MetricTagsMap& model_tags,
      std::shared_ptr<MetricModelReporter>* metric_model_reporter);

  ~MetricModelReporter();
  void IncrementCounter(const std::string& name, double value);
  void ObserveSummary(const std::string& name, double value);

 private:
  MetricModelReporter(
      const std::string& model_name, const int64_t model_version,
      const int device, const triton::common::MetricTagsMap& model_tags);

  static void GetMetricLabels(
      std::map<std::string, std::string>* labels, const std::string& model_name,
      const int64_t model_version, const int device,
      const triton::common::MetricTagsMap& model_tags);

  template <typename T, typename... Args>
  T* CreateMetric(
      prometheus::Family<T>& family,
      const std::map<std::string, std::string>& labels, Args&&... args);

  void InitializeCounters(const std::map<std::string, std::string>& labels);
  void InitializeSummaries(const std::map<std::string, std::string>& labels);

  // Metric Families
  std::unordered_map<std::string, prometheus::Family<prometheus::Counter>*>
      counter_families_;
  std::unordered_map<std::string, prometheus::Family<prometheus::Summary>*>
      summary_families_;

  // Metrics
  std::unordered_map<std::string, prometheus::Counter*> counters_;
  std::unordered_map<std::string, prometheus::Summary*> summaries_;
#endif  // TRITON_ENABLE_METRICS
};

}}  // namespace triton::core
