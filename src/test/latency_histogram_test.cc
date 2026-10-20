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

#include "latency_histogram.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

// Prometheus promql.BucketQuantile. The last cumulative count is +Inf.
double
BucketQuantile(
    double quantile, const std::vector<double>& upper_bounds,
    const std::vector<uint64_t>& cumulative, uint64_t total)
{
  if (quantile < 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (quantile > 1.0) {
    return std::numeric_limits<double>::infinity();
  }
  if (upper_bounds.empty() || total == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double rank = quantile * static_cast<double>(total);
  size_t bucket = 0;
  for (; bucket < cumulative.size(); ++bucket) {
    if (static_cast<double>(cumulative[bucket]) >= rank) {
      break;
    }
  }
  if (bucket == cumulative.size()) {
    return upper_bounds.back();
  }
  if (bucket == 0 && upper_bounds[0] <= 0.0) {
    return upper_bounds[0];
  }

  const double bucket_end = upper_bounds[bucket];
  double bucket_start = 0.0;
  double count = static_cast<double>(cumulative[bucket]);
  double rank_in_bucket = rank;
  if (bucket > 0) {
    bucket_start = upper_bounds[bucket - 1];
    count -= static_cast<double>(cumulative[bucket - 1]);
    rank_in_bucket -= static_cast<double>(cumulative[bucket - 1]);
  }
  if (count == 0.0) {
    return bucket_start;
  }
  return bucket_start + (bucket_end - bucket_start) * (rank_in_bucket / count);
}

double
EmpiricalQuantile(std::vector<uint64_t> samples, double quantile)
{
  std::sort(samples.begin(), samples.end());
  const double rank = std::ceil(quantile * static_cast<double>(samples.size()));
  size_t index = rank <= 1.0 ? 0 : static_cast<size_t>(rank) - 1;
  if (index >= samples.size()) {
    index = samples.size() - 1;
  }
  return static_cast<double>(samples[index]);
}

class RegistryGuard : public ::testing::Test {
 protected:
  void SetUp() override { triton::core::LatencyHistogramRegistry::Instance().Clear(); }
  void TearDown() override
  {
    triton::core::LatencyHistogramRegistry::Instance().Clear();
  }
};

}  // namespace

namespace triton { namespace core {

TEST(DefaultLatencyHistogramBoundaries, GeometricFrom1UsTo60s)
{
  const std::vector<double> buckets = DefaultLatencyHistogramBoundaries();
  ASSERT_GE(buckets.size(), 80u);
  ASSERT_LE(buckets.size(), 120u);
  EXPECT_DOUBLE_EQ(buckets.front(), 1.0);
  EXPECT_GE(buckets.back(), 60.0 * 1000.0 * 1000.0);
  for (size_t i = 1; i + 1 < buckets.size(); ++i) {
    EXPECT_NEAR(buckets[i] / buckets[i - 1], 1.2, 1e-9);
  }
}

TEST(LatencyMetricToggles, DisabledByDefault)
{
  const LatencyMetricToggles toggles = ResolveLatencyMetricToggles({});
  EXPECT_FALSE(toggles.duration_histograms);
  EXPECT_FALSE(toggles.first_response_histogram);
  EXPECT_FALSE(toggles.saw_summary_latencies);
  EXPECT_FALSE(toggles.saw_summary_quantiles);
}

TEST(LatencyMetricToggles, SummaryFlagEnablesDurationHistogramsOnly)
{
  const LatencyMetricToggles toggles =
      ResolveLatencyMetricToggles({{"summary_latencies", "true"}});
  EXPECT_TRUE(toggles.duration_histograms);
  EXPECT_FALSE(toggles.first_response_histogram);
  EXPECT_TRUE(toggles.saw_summary_latencies);
}

TEST(LatencyMetricToggles, HistogramFlagEnablesDurationAndFirstResponse)
{
  const LatencyMetricToggles toggles =
      ResolveLatencyMetricToggles({{"histogram_latencies", "true"}});
  EXPECT_TRUE(toggles.duration_histograms);
  EXPECT_TRUE(toggles.first_response_histogram);
  EXPECT_FALSE(toggles.saw_summary_latencies);
}

TEST(LatencyMetricToggles, SummaryQuantilesAreNotedButDoNotEnableMetrics)
{
  const LatencyMetricToggles toggles = ResolveLatencyMetricToggles(
      {{"summary_quantiles", "0.5:0.05,0.99:0.001"}});
  EXPECT_TRUE(toggles.saw_summary_quantiles);
  EXPECT_FALSE(toggles.duration_histograms);
  EXPECT_FALSE(toggles.first_response_histogram);
}

TEST(ShardedLatencyHistogram, ExactBucketsCountAndSum)
{
  ShardedLatencyHistogram histogram(
      "nv_inference_queue_histogram_us", "queue latency",
      {{"model", "identity"}, {"version", "1"}}, {10.0, 20.0, 40.0});

  histogram.Observe(5);
  histogram.Observe(10);
  histogram.Observe(11);
  histogram.Observe(100);

  const auto snapshot = histogram.Collect();
  ASSERT_EQ(snapshot.boundaries.size(), 3u);
  ASSERT_EQ(snapshot.cumulative.size(), 3u);
  EXPECT_EQ(snapshot.cumulative[0], 2u);
  EXPECT_EQ(snapshot.cumulative[1], 3u);
  EXPECT_EQ(snapshot.cumulative[2], 3u);
  EXPECT_EQ(snapshot.count, 4u);
  EXPECT_EQ(snapshot.sum, 5u + 10u + 11u + 100u);

  std::string text;
  histogram.AppendPrometheusText(&text);
  EXPECT_NE(
      text.find(
          "nv_inference_queue_histogram_us_bucket{model=\"identity\",version=\"1\",le=\"10\"} 2"),
      std::string::npos);
  EXPECT_NE(
      text.find(
          "nv_inference_queue_histogram_us_bucket{model=\"identity\",version=\"1\",le=\"+Inf\"} 4"),
      std::string::npos);
  EXPECT_NE(
      text.find(
          "nv_inference_queue_histogram_us_sum{model=\"identity\",version=\"1\"} 126"),
      std::string::npos);
  EXPECT_NE(
      text.find(
          "nv_inference_queue_histogram_us_count{model=\"identity\",version=\"1\"} 4"),
      std::string::npos);
  EXPECT_EQ(text.find("quantile="), std::string::npos) << text;
  EXPECT_NE(text.find("# TYPE nv_inference_queue_histogram_us histogram"), std::string::npos);
}

TEST(ShardedLatencyHistogram, CustomBoundariesAreSorted)
{
  ShardedLatencyHistogram histogram(
      "nv_inference_request_histogram_us", "request", {{"model", "m"}},
      {100.0, 1.0, 10.0, 10.0});
  histogram.Observe(10);
  histogram.Observe(50);
  const auto snapshot = histogram.Collect();
  ASSERT_EQ(snapshot.boundaries.size(), 3u);
  EXPECT_DOUBLE_EQ(snapshot.boundaries[0], 1.0);
  EXPECT_DOUBLE_EQ(snapshot.boundaries[1], 10.0);
  EXPECT_DOUBLE_EQ(snapshot.boundaries[2], 100.0);
  EXPECT_EQ(snapshot.cumulative[0], 0u);
  EXPECT_EQ(snapshot.cumulative[1], 1u);
  EXPECT_EQ(snapshot.cumulative[2], 2u);
  EXPECT_EQ(snapshot.count, 2u);
}

TEST(ShardedLatencyHistogram, P99WithinBucketError)
{
  const std::vector<double> boundaries = DefaultLatencyHistogramBoundaries();
  ShardedLatencyHistogram histogram(
      "nv_inference_request_histogram_us", "request", {}, boundaries);

  constexpr int kSamples = 100000;
  std::mt19937 rng(42);
  std::lognormal_distribution<double> dist(std::log(1000.0), 1.0);
  std::vector<uint64_t> samples;
  samples.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    const auto sample = static_cast<uint64_t>(std::max(1.0, dist(rng)));
    samples.push_back(sample);
    histogram.Observe(sample);
  }

  const auto snapshot = histogram.Collect();
  EXPECT_EQ(snapshot.count, static_cast<uint64_t>(kSamples));
  uint64_t sum = 0;
  for (uint64_t sample : samples) {
    sum += sample;
  }
  EXPECT_EQ(snapshot.sum, sum);
  EXPECT_EQ(snapshot.cumulative.back(), snapshot.count);

  const double estimated = BucketQuantile(
      0.99, snapshot.boundaries, snapshot.cumulative, snapshot.count);
  const double empirical = EmpiricalQuantile(samples, 0.99);
  ASSERT_GT(empirical, 0.0);
  const double relative_error = std::abs(estimated - empirical) / empirical;
  EXPECT_LT(relative_error, 0.15)
      << "estimated=" << estimated << " empirical=" << empirical;
}

TEST(ShardedLatencyHistogram, ConcurrentObservationsAreNotLost)
{
  ShardedLatencyHistogram histogram(
      "nv_inference_compute_infer_histogram_us", "infer", {}, {1000.0});
  constexpr int kThreads = 8;
  constexpr int kPerThread = 50000;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&histogram] {
      for (int i = 0; i < kPerThread; ++i) {
        histogram.Observe(42);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  const auto snapshot = histogram.Collect();
  const uint64_t expected = static_cast<uint64_t>(kThreads) * kPerThread;
  EXPECT_EQ(snapshot.count, expected);
  EXPECT_EQ(snapshot.sum, expected * 42u);
  EXPECT_EQ(snapshot.cumulative[0], expected);
}

TEST(ShardedLatencyHistogram, CollectStaysConsistentWhileWritersRun)
{
  ShardedLatencyHistogram histogram(
      "nv_inference_request_histogram_us", "request", {}, {10.0, 100.0});
  std::atomic<bool> run{true};
  std::vector<std::thread> writers;
  for (int thread = 0; thread < 4; ++thread) {
    writers.emplace_back([&histogram, &run] {
      uint64_t value = 1;
      while (run.load(std::memory_order_relaxed)) {
        histogram.Observe(value);
        value = value == 1 ? 50 : 1;
      }
    });
  }

  for (int i = 0; i < 200; ++i) {
    const auto snapshot = histogram.Collect();
    uint64_t previous = 0;
    for (uint64_t cumulative : snapshot.cumulative) {
      EXPECT_GE(cumulative, previous);
      EXPECT_LE(cumulative, snapshot.count);
      previous = cumulative;
    }
    EXPECT_GE(snapshot.count, previous);
  }

  run.store(false, std::memory_order_relaxed);
  for (auto& writer : writers) {
    writer.join();
  }
  const auto snapshot = histogram.Collect();
  EXPECT_GT(snapshot.count, 0u);
  EXPECT_LE(snapshot.cumulative.back(), snapshot.count);
}

TEST_F(RegistryGuard, UnregisteredHistogramsAreOmitted)
{
  EXPECT_EQ(LatencyHistogramRegistry::Instance().Serialize(), "");

  std::map<std::string, std::string> labels{{"model", "m"}};
  auto histogram = std::make_shared<ShardedLatencyHistogram>(
      "nv_inference_request_histogram_us", "request", labels,
      std::vector<double>{10.0});
  histogram->Observe(1);
  LatencyHistogramRegistry::Instance().Register(histogram);
  const std::string exposed = LatencyHistogramRegistry::Instance().Serialize();
  EXPECT_NE(exposed.find("nv_inference_request_histogram_us_count"), std::string::npos);
  EXPECT_EQ(exposed.find("nv_inference_request_summary"), std::string::npos);

  LatencyHistogramRegistry::Instance().Unregister(histogram.get());
  EXPECT_EQ(LatencyHistogramRegistry::Instance().Serialize().find(
                "nv_inference_request_histogram_us"),
            std::string::npos);
}

TEST_F(RegistryGuard, OneHelpAndTypeLinePerMetricName)
{
  auto first = std::make_shared<ShardedLatencyHistogram>(
      "nv_inference_request_histogram_us", "request latency",
      std::map<std::string, std::string>{{"model", "a"}},
      std::vector<double>{10.0});
  auto second = std::make_shared<ShardedLatencyHistogram>(
      "nv_inference_request_histogram_us", "request latency",
      std::map<std::string, std::string>{{"model", "b"}},
      std::vector<double>{10.0});
  first->Observe(1);
  second->Observe(2);
  LatencyHistogramRegistry::Instance().Register(first);
  LatencyHistogramRegistry::Instance().Register(second);

  const std::string exposed = LatencyHistogramRegistry::Instance().Serialize();
  const std::string type_line =
      "# TYPE nv_inference_request_histogram_us histogram";
  const std::string help_line =
      "# HELP nv_inference_request_histogram_us request latency";
  EXPECT_EQ(exposed.find(type_line), exposed.rfind(type_line));
  EXPECT_EQ(exposed.find(help_line), exposed.rfind(help_line));
  EXPECT_NE(exposed.find("model=\"a\""), std::string::npos);
  EXPECT_NE(exposed.find("model=\"b\""), std::string::npos);
}

}}  // namespace triton::core
