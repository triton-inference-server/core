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
//
#pragma once

#ifdef TRITON_ENABLE_METRICS

#include <atomic>
#include <istream>
#include <mutex>
#include <thread>

#include "prometheus/counter.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/serializer.h"
#include "prometheus/summary.h"
#include "prometheus/text_serializer.h"
#include "status.h"

#ifdef TRITON_ENABLE_METRICS_GPU
#include <dcgm_agent.h>
#endif  // TRITON_ENABLE_METRICS_GPU

namespace triton { namespace core {

using MetricsConfig = std::vector<std::pair<std::string, std::string>>;
using MetricsConfigMap = std::unordered_map<std::string, MetricsConfig>;

#ifdef TRITON_ENABLE_METRICS_CPU
using MemInfo = std::unordered_map<std::string, uint64_t>;

// References:
// - htop source: https://stackoverflow.com/a/23376195
// - Linux docs: https://www.kernel.org/doc/Documentation/filesystems/proc.txt
// guest/guestnice values are counted in user/nice so we skip parsing them
struct CpuInfo {
  uint64_t user = 0;     // normal processes executing in user mode
  uint64_t nice = 0;     // niced processes executing in user mode
  uint64_t system = 0;   // processes executing in kernel mode
  uint64_t idle = 0;     // twiddling thumbs
  uint64_t iowait = 0;   // waiting for I/O to complete
  uint64_t irq = 0;      // servicing interrupts
  uint64_t softirq = 0;  // servicing softirqs
  uint64_t steal = 0;    // involuntary wait
};

inline std::istream&
operator>>(std::istream& is, CpuInfo& info)
{
  is >> info.user >> info.nice >> info.system >> info.idle >> info.iowait >>
      info.irq >> info.softirq >> info.steal;
  return is;
}
#endif  // TRITON_ENABLE_METRICS_CPU

#ifdef TRITON_ENABLE_METRICS_GPU
struct DcgmMetadata {
  // DCGM handles for initialization and destruction
  dcgmHandle_t dcgm_handle_ = 0;
  dcgmGpuGrp_t groupId_ = 0;
  // DCGM Flags
  bool standalone_ = false;
  // DCGM Fields
  size_t field_count_ = 0;
  std::vector<unsigned short> fields_;
  // GPU Device Mapping
  std::map<uint32_t, uint32_t> cuda_ids_to_dcgm_ids_;
  std::vector<uint32_t> available_cuda_gpu_ids_;
  // Stop attempting metrics if they fail multiple consecutive
  // times for a device.
  const int fail_threshold_ = 3;
  // DCGM Failure Tracking
  std::vector<int> power_limit_fail_cnt_;
  std::vector<int> power_usage_fail_cnt_;
  std::vector<int> energy_fail_cnt_;
  std::vector<int> util_fail_cnt_;
  std::vector<int> mem_fail_cnt_;
  // DCGM Energy Tracking
  std::vector<unsigned long long> last_energy_;
  // Track if DCGM handle initialized successfully
  bool dcgm_initialized_ = false;
};
#endif  // TRITON_ENABLE_METRICS_GPU

class Metrics {
 public:
  // Return the hash value of the labels
  static size_t HashLabels(const std::map<std::string, std::string>& labels);

  // Are metrics enabled?
  static bool Enabled();

  // Enable reporting of metrics
  static void EnableMetrics();

  // Enable reporting of Pinned memory metrics
  static void EnablePinnedMemoryMetrics();

  // Enable reporting of GPU metrics
  static void EnableGPUMetrics();

  // Enable reporting of CPU metrics
  static void EnableCpuMetrics();

  // Start a thread for polling enabled metrics if any
  static void StartPollingThreadSingleton();

  // Set the time interval in secs at which metrics are collected
  static void SetMetricsInterval(uint64_t metrics_interval_ms);

  // Set the config for Metrics to control various options generically
  static void SetConfigMap(MetricsConfigMap cfg);

  // Get the config for Metrics
  static const MetricsConfigMap& ConfigMap();

  // Get the prometheus registry
  static std::shared_ptr<prometheus::Registry> GetRegistry();

  // Get serialized metrics
  static const std::string SerializedMetrics();

  // Get the UUID for a CUDA device. Return true and initialize 'uuid'
  // if a UUID is found, return false if a UUID cannot be returned.
  static bool UUIDForCudaDevice(int cuda_device, std::string* uuid);

  // Metric family counting successful inference requests
  static prometheus::Family<prometheus::Counter>& FamilyInferenceSuccess()
  {
    return GetSingleton()->inf_success_family_;
  }

  // Metric family counting failed inference requests
  static prometheus::Family<prometheus::Counter>& FamilyInferenceFailure()
  {
    return GetSingleton()->inf_failure_family_;
  }

  // Metric family counting inferences performed, where a batch-size
  // 'n' inference request is counted as 'n' inferences
  static prometheus::Family<prometheus::Counter>& FamilyInferenceCount()
  {
    return GetSingleton()->inf_count_family_;
  }

  // Metric family counting inferences performed, where a batch-size
  // 'n' inference request is counted as 'n' inferences
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceExecutionCount()
  {
    return GetSingleton()->inf_count_exec_family_;
  }

  // Metric family of cumulative inference request duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceRequestDuration()
  {
    return GetSingleton()->inf_request_duration_us_family_;
  }

  // Metric family of cumulative inference queuing duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>& FamilyInferenceQueueDuration()
  {
    return GetSingleton()->inf_queue_duration_us_family_;
  }

  // Metric family of cumulative inference compute durations, in
  // microseconds
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeInputDuration()
  {
    return GetSingleton()->inf_compute_input_duration_us_family_;
  }
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeInferDuration()
  {
    return GetSingleton()->inf_compute_infer_duration_us_family_;
  }
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeOutputDuration()
  {
    return GetSingleton()->inf_compute_output_duration_us_family_;
  }

  // Metric family of instantaneous inference queue size per model
  static prometheus::Family<prometheus::Gauge>& FamilyInferenceQueueSize()
  {
    return GetSingleton()->inf_pending_request_count_family_;
  }

  static prometheus::Family<prometheus::Histogram>&
  FamilyFirstResponseDuration()
  {
    return GetSingleton()->inf_first_response_histogram_ms_family_;
  }

  // Metric family of load time per model
  static prometheus::Family<prometheus::Gauge>& FamilyModelLoadTime()
  {
    return GetSingleton()->model_load_time_family_;
  }

  // Metric families of per-model response cache metrics
  // NOTE: These are used in infer_stats for perf_analyzer
  static prometheus::Family<prometheus::Counter>& FamilyCacheHitCount()
  {
    return GetSingleton()->cache_num_hits_model_family_;
  }
  static prometheus::Family<prometheus::Counter>& FamilyCacheHitDuration()
  {
    return GetSingleton()->cache_hit_duration_us_model_family_;
  }
  static prometheus::Family<prometheus::Counter>& FamilyCacheMissCount()
  {
    return GetSingleton()->cache_num_misses_model_family_;
  }
  static prometheus::Family<prometheus::Counter>& FamilyCacheMissDuration()
  {
    return GetSingleton()->cache_miss_duration_us_model_family_;
  }

  // Summaries
  static prometheus::Family<prometheus::Summary>&
  FamilyInferenceRequestSummary()
  {
    return GetSingleton()->inf_request_summary_us_family_;
  }
  static prometheus::Family<prometheus::Summary>& FamilyInferenceQueueSummary()
  {
    return GetSingleton()->inf_queue_summary_us_family_;
  }
  static prometheus::Family<prometheus::Summary>&
  FamilyInferenceComputeInputSummary()
  {
    return GetSingleton()->inf_compute_input_summary_us_family_;
  }
  static prometheus::Family<prometheus::Summary>&
  FamilyInferenceComputeInferSummary()
  {
    return GetSingleton()->inf_compute_infer_summary_us_family_;
  }
  static prometheus::Family<prometheus::Summary>&
  FamilyInferenceComputeOutputSummary()
  {
    return GetSingleton()->inf_compute_output_summary_us_family_;
  }
  static prometheus::Family<prometheus::Summary>& FamilyCacheHitSummary()
  {
    return GetSingleton()->cache_hit_summary_us_model_family_;
  }
  static prometheus::Family<prometheus::Summary>& FamilyCacheMissSummary()
  {
    return GetSingleton()->cache_miss_summary_us_model_family_;
  }

 private:
  Metrics();
  virtual ~Metrics();
  static Metrics* GetSingleton();
  bool InitializeDcgmMetrics();
  bool InitializeCpuMetrics();
  bool InitializePinnedMemoryMetrics();
  bool StartPollingThread();
  bool PollPinnedMemoryMetrics();
  bool PollDcgmMetrics();
  bool PollCpuMetrics();

  std::string dcgmValueToErrorMessage(double val);
  std::string dcgmValueToErrorMessage(int64_t val);

  std::shared_ptr<prometheus::Registry> registry_;
  std::unique_ptr<prometheus::Serializer> serializer_;

  // DLIS-4761: Refactor into groups of families
  prometheus::Family<prometheus::Counter>& inf_success_family_;
  prometheus::Family<prometheus::Counter>& inf_failure_family_;
  prometheus::Family<prometheus::Counter>& inf_count_family_;
  prometheus::Family<prometheus::Counter>& inf_count_exec_family_;
  prometheus::Family<prometheus::Counter>& inf_request_duration_us_family_;
  prometheus::Family<prometheus::Counter>& inf_queue_duration_us_family_;
  prometheus::Family<prometheus::Counter>&
      inf_compute_input_duration_us_family_;
  prometheus::Family<prometheus::Counter>&
      inf_compute_infer_duration_us_family_;
  prometheus::Family<prometheus::Counter>&
      inf_compute_output_duration_us_family_;
  prometheus::Family<prometheus::Gauge>& inf_pending_request_count_family_;
  prometheus::Family<prometheus::Gauge>& model_load_time_family_;

  prometheus::Family<prometheus::Gauge>& pinned_memory_pool_total_family_;
  prometheus::Family<prometheus::Gauge>& pinned_memory_pool_used_family_;
  prometheus::Gauge* pinned_memory_pool_total_;
  prometheus::Gauge* pinned_memory_pool_used_;

  // Per-model Response Cache metrics
  // NOTE: Per-model metrics are used in infer_stats for perf_analyzer. Global
  // cache metrics will be implemented by cache and published through
  // Metrics C API.
  prometheus::Family<prometheus::Counter>& cache_num_hits_model_family_;
  prometheus::Family<prometheus::Counter>& cache_hit_duration_us_model_family_;
  prometheus::Family<prometheus::Counter>& cache_num_misses_model_family_;
  prometheus::Family<prometheus::Counter>& cache_miss_duration_us_model_family_;

  // Histograms
  prometheus::Family<prometheus::Histogram>&
      inf_first_response_histogram_ms_family_;

  // Summaries
  prometheus::Family<prometheus::Summary>& inf_request_summary_us_family_;
  prometheus::Family<prometheus::Summary>& inf_queue_summary_us_family_;
  prometheus::Family<prometheus::Summary>& inf_compute_input_summary_us_family_;
  prometheus::Family<prometheus::Summary>& inf_compute_infer_summary_us_family_;
  prometheus::Family<prometheus::Summary>&
      inf_compute_output_summary_us_family_;
  prometheus::Family<prometheus::Summary>& cache_hit_summary_us_model_family_;
  prometheus::Family<prometheus::Summary>& cache_miss_summary_us_model_family_;

#ifdef TRITON_ENABLE_METRICS_GPU
  prometheus::Family<prometheus::Gauge>& gpu_utilization_family_;
  prometheus::Family<prometheus::Gauge>& gpu_memory_total_family_;
  prometheus::Family<prometheus::Gauge>& gpu_memory_used_family_;
  prometheus::Family<prometheus::Gauge>& gpu_power_usage_family_;
  prometheus::Family<prometheus::Gauge>& gpu_power_limit_family_;
  prometheus::Family<prometheus::Counter>& gpu_energy_consumption_family_;

  std::vector<prometheus::Gauge*> gpu_utilization_;
  std::vector<prometheus::Gauge*> gpu_memory_total_;
  std::vector<prometheus::Gauge*> gpu_memory_used_;
  std::vector<prometheus::Gauge*> gpu_power_usage_;
  std::vector<prometheus::Gauge*> gpu_power_limit_;
  std::vector<prometheus::Counter*> gpu_energy_consumption_;

  DcgmMetadata dcgm_metadata_;
#endif  // TRITON_ENABLE_METRICS_GPU

#ifdef TRITON_ENABLE_METRICS_CPU
  // Parses "/proc/meminfo" for metrics, currently only supported on Linux.
  Status ParseMemInfo(MemInfo& info);
  // Parses "/proc/stat" for metrics, currently only supported on Linux.
  Status ParseCpuInfo(CpuInfo& info);
  // Computes CPU utilization between "info_new" and "info_old" values
  double CpuUtilization(const CpuInfo& info_new, const CpuInfo& info_old);

  prometheus::Family<prometheus::Gauge>& cpu_utilization_family_;
  prometheus::Family<prometheus::Gauge>& cpu_memory_total_family_;
  prometheus::Family<prometheus::Gauge>& cpu_memory_used_family_;

  prometheus::Gauge* cpu_utilization_;
  prometheus::Gauge* cpu_memory_total_;
  prometheus::Gauge* cpu_memory_used_;
  CpuInfo last_cpu_info_;
#endif  // TRITON_ENABLE_METRICS_CPU

  // Thread for polling cpu/gpu metrics periodically
  std::unique_ptr<std::thread> poll_thread_;
  std::atomic<bool> poll_thread_exit_;
  bool metrics_enabled_;
  bool gpu_metrics_enabled_;
  bool cpu_metrics_enabled_;
  bool pinned_memory_metrics_enabled_;
  bool poll_thread_started_;
  std::mutex metrics_enabling_;
  std::mutex poll_thread_starting_;
  uint64_t metrics_interval_ms_;
  MetricsConfigMap config_;
};

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METRICS
