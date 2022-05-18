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

#include "infer_stats.h"

#include <time.h>
#include "metric_model_reporter.h"
#include "metrics.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

#ifdef TRITON_ENABLE_STATS

void
InferenceStatsAggregator::UpdateFailure(
    MetricModelReporter* metric_reporter, const uint64_t request_start_ns,
    const uint64_t request_end_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  infer_stats_.failure_count_++;
  infer_stats_.failure_duration_ns_ += (request_end_ns - request_start_ns);

#ifdef TRITON_ENABLE_METRICS
  if (metric_reporter != nullptr) {
    metric_reporter->MetricInferenceFailure().Increment(1);
  }
#endif  // TRITON_ENABLE_METRICS
}

void
InferenceStatsAggregator::UpdateSuccess(
    MetricModelReporter* metric_reporter, const size_t batch_size,
    const uint64_t request_start_ns, const uint64_t queue_start_ns,
    const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
    const uint64_t compute_output_start_ns, const uint64_t compute_end_ns,
    const uint64_t request_end_ns)
{
  const uint64_t compute_input_duration_ns =
      compute_input_end_ns - compute_start_ns;
  const uint64_t compute_infer_duration_ns =
      compute_output_start_ns - compute_input_end_ns;
  const uint64_t compute_output_duration_ns =
      compute_end_ns - compute_output_start_ns;
  UpdateSuccessWithDuration(
      metric_reporter, batch_size, request_start_ns, queue_start_ns,
      compute_start_ns, request_end_ns, compute_input_duration_ns,
      compute_infer_duration_ns, compute_output_duration_ns);
}

void
InferenceStatsAggregator::UpdateSuccessWithDuration(
    MetricModelReporter* metric_reporter, const size_t batch_size,
    const uint64_t request_start_ns, const uint64_t queue_start_ns,
    const uint64_t compute_start_ns, const uint64_t request_end_ns,
    const uint64_t compute_input_duration_ns,
    const uint64_t compute_infer_duration_ns,
    const uint64_t compute_output_duration_ns)
{
  const uint64_t request_duration_ns = request_end_ns - request_start_ns;
  const uint64_t queue_duration_ns = compute_start_ns - queue_start_ns;

  std::lock_guard<std::mutex> lock(mu_);

  inference_count_ += batch_size;

  infer_stats_.success_count_++;
  infer_stats_.request_duration_ns_ += request_duration_ns;
  infer_stats_.queue_duration_ns_ += queue_duration_ns;
  infer_stats_.compute_input_duration_ns_ += compute_input_duration_ns;
  infer_stats_.compute_infer_duration_ns_ += compute_infer_duration_ns;
  infer_stats_.compute_output_duration_ns_ += compute_output_duration_ns;

#ifdef TRITON_ENABLE_METRICS
  if (metric_reporter != nullptr) {
    metric_reporter->MetricInferenceSuccess().Increment(1);
    metric_reporter->MetricInferenceCount().Increment(batch_size);
    metric_reporter->MetricInferenceRequestDuration().Increment(
        request_duration_ns / 1000);
    metric_reporter->MetricInferenceQueueDuration().Increment(
        queue_duration_ns / 1000);
    metric_reporter->MetricInferenceComputeInputDuration().Increment(
        compute_input_duration_ns / 1000);
    metric_reporter->MetricInferenceComputeInferDuration().Increment(
        compute_infer_duration_ns / 1000);
    metric_reporter->MetricInferenceComputeOutputDuration().Increment(
        compute_output_duration_ns / 1000);
  }
#endif  // TRITON_ENABLE_METRICS
}

// Currently cache hits will not go to the inference backend where metrics
// are typically updated, so this method allows us to update relevant metrics
// from a metric reporter rather than going through the backend.
void
InferenceStatsAggregator::UpdateSuccessCacheHit(
    MetricModelReporter* metric_reporter, const size_t batch_size,
    const uint64_t request_start_ns, const uint64_t queue_start_ns,
    const uint64_t cache_lookup_start_ns, const uint64_t request_end_ns,
    const uint64_t cache_hit_lookup_duration_ns)
{
  const uint64_t request_duration_ns = request_end_ns - request_start_ns;
  const uint64_t queue_duration_ns = cache_lookup_start_ns - queue_start_ns;

  std::lock_guard<std::mutex> lock(mu_);

  infer_stats_.success_count_++;
  infer_stats_.request_duration_ns_ += request_duration_ns;
  infer_stats_.queue_duration_ns_ += queue_duration_ns;
  infer_stats_.cache_hit_count_++;
  infer_stats_.cache_hit_lookup_duration_ns_ += cache_hit_lookup_duration_ns;

#ifdef TRITON_ENABLE_METRICS
  if (metric_reporter != nullptr) {
    metric_reporter->MetricInferenceSuccess().Increment(1);
    metric_reporter->MetricInferenceRequestDuration().Increment(
        request_duration_ns / 1000);
    metric_reporter->MetricInferenceQueueDuration().Increment(
        queue_duration_ns / 1000);
    metric_reporter->MetricCacheHitCount().Increment(1);
    metric_reporter->MetricCacheHitLookupDuration().Increment(
        cache_hit_lookup_duration_ns / 1000);
  }
#endif  // TRITON_ENABLE_METRICS
}

// Cache misses will go to the inference backend where metrics are typically
// updated, but cache insertion happens after the inference backend finishes.
// So we use this method to update cache miss stats and adjust the request
// duration to include cache insertion time.
void
InferenceStatsAggregator::UpdateSuccessCacheMiss(
    MetricModelReporter* metric_reporter,
    const uint64_t cache_miss_lookup_duration_ns,
    const uint64_t cache_miss_insertion_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  const uint64_t cache_miss_duration_ns =
      cache_miss_lookup_duration_ns + cache_miss_insertion_duration_ns;
  infer_stats_.request_duration_ns_ += cache_miss_duration_ns;
  infer_stats_.cache_miss_count_++;
  infer_stats_.cache_miss_lookup_duration_ns_ += cache_miss_lookup_duration_ns;
  infer_stats_.cache_miss_insertion_duration_ns_ +=
      cache_miss_insertion_duration_ns;

#ifdef TRITON_ENABLE_METRICS
  if (metric_reporter != nullptr) {
    // Add cache insertion time to request duration since insertion
    // happens after inference backend sets the request duration, and
    // cache lookup time was already included before the inference backend
    // was called
    metric_reporter->MetricInferenceRequestDuration().Increment(
        cache_miss_duration_ns / 1000);
    metric_reporter->MetricCacheMissCount().Increment(1);
    metric_reporter->MetricCacheMissLookupDuration().Increment(
        cache_miss_lookup_duration_ns / 1000);
    metric_reporter->MetricCacheMissInsertionDuration().Increment(
        cache_miss_insertion_duration_ns / 1000);
  }
#endif  // TRITON_ENABLE_METRICS
}

void
InferenceStatsAggregator::UpdateInferBatchStats(
    MetricModelReporter* metric_reporter, const size_t batch_size,
    const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
    const uint64_t compute_output_start_ns, const uint64_t compute_end_ns)
{
  auto compute_input_duration_ns = (compute_input_end_ns - compute_start_ns);
  auto compute_infer_duration_ns =
      (compute_output_start_ns - compute_input_end_ns);
  auto compute_output_duration_ns = (compute_end_ns - compute_output_start_ns);
  UpdateInferBatchStatsWithDuration(
      metric_reporter, batch_size, compute_input_duration_ns,
      compute_infer_duration_ns, compute_output_duration_ns);
}

void
InferenceStatsAggregator::UpdateInferBatchStatsWithDuration(
    MetricModelReporter* metric_reporter, size_t batch_size,
    const uint64_t compute_input_duration_ns,
    const uint64_t compute_infer_duration_ns,
    const uint64_t compute_output_duration_ns)
{
  uint64_t inference_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::lock_guard<std::mutex> lock(mu_);

  if (inference_ms > last_inference_ms_) {
    last_inference_ms_ = inference_ms;
  }

  execution_count_++;

  auto it = batch_stats_.find(batch_size);
  if (it == batch_stats_.end()) {
    it = batch_stats_.emplace(batch_size, InferBatchStats()).first;
  }
  it->second.count_++;
  it->second.compute_input_duration_ns_ += compute_input_duration_ns;
  it->second.compute_infer_duration_ns_ += compute_infer_duration_ns;
  it->second.compute_output_duration_ns_ += compute_output_duration_ns;

#ifdef TRITON_ENABLE_METRICS
  if (metric_reporter != nullptr) {
    metric_reporter->MetricInferenceExecutionCount().Increment(1);
  }
#endif  // TRITON_ENABLE_METRICS
}

#endif  // TRITON_ENABLE_STATS

}}  // namespace triton::core
