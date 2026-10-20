// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace triton { namespace core {

// Geometric bucket boundaries from 1us to 60s with a ratio of 1.2.
// histogram_quantile linear interpolation is then off by at most about half a
// bucket, roughly 10% relative error.
std::vector<double> DefaultLatencyHistogramBoundaries();

// How --metrics-config selects latency distribution metrics.
// summary_latencies used to build Prometheus Summaries (CKMS). Those summaries
// are no longer created; the flag enables the lock-free duration histograms
// instead. histogram_latencies enables the same duration histograms and the
// existing first-response histogram.
struct LatencyMetricToggles {
  bool duration_histograms = false;
  bool first_response_histogram = false;
  bool saw_summary_latencies = false;
  bool saw_summary_quantiles = false;
};

LatencyMetricToggles ResolveLatencyMetricToggles(
    const std::vector<std::pair<std::string, std::string>>& settings);

// Cumulative Prometheus histogram updated without a mutex. Each thread adds
// into its own shard with relaxed atomics. Scrapes sum the shards.
class ShardedLatencyHistogram {
 public:
  static constexpr size_t kShardCount = 32;

  ShardedLatencyHistogram(
      std::string name, std::string help,
      std::map<std::string, std::string> labels,
      std::vector<double> boundaries);

  void Observe(uint64_t value_us) noexcept;

  struct Snapshot {
    std::vector<double> boundaries;
    // cumulative[i] is the number of observations <= boundaries[i].
    std::vector<uint64_t> cumulative;
    uint64_t count = 0;
    uint64_t sum = 0;
  };
  Snapshot Collect() const;

  void AppendPrometheusText(std::string* out) const;

  const std::string& Name() const { return name_; }

 private:
  size_t BucketIndex(uint64_t value_us) const;

  std::string name_;
  std::string help_;
  std::map<std::string, std::string> labels_;
  std::vector<double> boundaries_;

  struct alignas(64) Shard {
    std::atomic<uint64_t> sum{0};
    std::atomic<uint64_t> count{0};
    std::unique_ptr<std::atomic<uint64_t>[]> buckets;
  };
  std::unique_ptr<Shard[]> shards_;
  size_t bucket_count_ = 0;
};

// Histograms registered here are appended to the Prometheus text exposition
// by Metrics::SerializedMetrics. Register and unregister happen on model load
// and unload, not on the request path.
class LatencyHistogramRegistry {
 public:
  static LatencyHistogramRegistry& Instance();

  void Register(std::shared_ptr<ShardedLatencyHistogram> histogram);
  void Unregister(const ShardedLatencyHistogram* histogram);
  std::string Serialize() const;
  void Clear();

 private:
  LatencyHistogramRegistry() = default;

  mutable std::mutex mu_;
  std::vector<std::shared_ptr<ShardedLatencyHistogram>> histograms_;
};

}}  // namespace triton::core
