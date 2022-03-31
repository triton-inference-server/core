// Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_METRICS

#include "metrics.h"

#include <thread>
#include "constants.h"
#include "prometheus/detail/utils.h"
#include "triton/common/logging.h"

#ifdef TRITON_ENABLE_METRICS_GPU
#include <cuda_runtime_api.h>
#include <dcgm_agent.h>
#include <cstring>
#include <set>
#include <string>
#endif  // TRITON_ENABLE_METRICS_GPU

namespace triton { namespace core {

Metrics::Metrics()
    : registry_(std::make_shared<prometheus::Registry>()),
      serializer_(new prometheus::TextSerializer()),
      inf_success_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_success")
              .Help("Number of successful inference requests, all batch sizes")
              .Register(*registry_)),
      inf_failure_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_failure")
              .Help("Number of failed inference requests, all batch sizes")
              .Register(*registry_)),
      inf_count_family_(prometheus::BuildCounter()
                            .Name("nv_inference_count")
                            .Help("Number of inferences performed (does not "
                                  "include cached requests)")
                            .Register(*registry_)),
      inf_count_exec_family_(prometheus::BuildCounter()
                                 .Name("nv_inference_exec_count")
                                 .Help("Number of model executions performed "
                                       "(does not include cached requests)")
                                 .Register(*registry_)),
      inf_request_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_duration_us")
              .Help("Cumulative inference request duration in microseconds "
                    "(includes cached requests)")
              .Register(*registry_)),
      inf_queue_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_queue_duration_us")
              .Help("Cumulative inference queuing duration in microseconds "
                    "(includes cached requests)")
              .Register(*registry_)),
      inf_compute_input_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_compute_input_duration_us")
              .Help("Cumulative compute input duration in microseconds (does "
                    "not include cached requests)")
              .Register(*registry_)),
      inf_compute_infer_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_compute_infer_duration_us")
              .Help("Cumulative compute inference duration in microseconds "
                    "(does not include cached requests)")
              .Register(*registry_)),
      inf_compute_output_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_compute_output_duration_us")
              .Help("Cumulative inference compute output duration in "
                    "microseconds (does not include cached requests)")
              .Register(*registry_)),
      cache_num_entries_family_(
          prometheus::BuildGauge()
              .Name("nv_cache_num_entries")
              .Help("Number of responses stored in response cache")
              .Register(*registry_)),
      cache_num_lookups_family_(
          prometheus::BuildGauge()
              .Name("nv_cache_num_lookups")
              .Help("Number of cache lookups in response cache")
              .Register(*registry_)),
      cache_num_hits_family_(prometheus::BuildGauge()
                                 .Name("nv_cache_num_hits")
                                 .Help("Number of cache hits in response cache")
                                 .Register(*registry_)),
      cache_num_misses_family_(
          prometheus::BuildGauge()
              .Name("nv_cache_num_misses")
              .Help("Number of cache misses in response cache")
              .Register(*registry_)),
      cache_num_evictions_family_(
          prometheus::BuildGauge()
              .Name("nv_cache_num_evictions")
              .Help("Number of cache evictions in response cache")
              .Register(*registry_)),
      cache_lookup_duration_us_family_(
          prometheus::BuildGauge()
              .Name("nv_cache_lookup_duration")
              .Help(
                  "Total cache lookup duration (hit and miss), in microseconds")
              .Register(*registry_)),
      cache_insertion_duration_us_family_(
          prometheus::BuildGauge()
              .Name("nv_cache_insertion_duration")
              .Help("Total cache insertion duration, in microseconds")
              .Register(*registry_)),
      cache_util_family_(prometheus::BuildGauge()
                             .Name("nv_cache_util")
                             .Help("Cache utilization [0.0 - 1.0]")
                             .Register(*registry_)),
      // Per-model cache metric families
      cache_num_hits_model_family_(prometheus::BuildCounter()
                                       .Name("nv_cache_num_hits_per_model")
                                       .Help("Number of cache hits per model")
                                       .Register(*registry_)),
      cache_hit_lookup_duration_us_model_family_(
          prometheus::BuildCounter()
              .Name("nv_cache_hit_lookup_duration_per_model")
              .Help(
                  "Total cache hit lookup duration per model, in microseconds")
              .Register(*registry_)),
      cache_num_misses_model_family_(
          prometheus::BuildCounter()
              .Name("nv_cache_num_misses_per_model")
              .Help("Number of cache misses per model")
              .Register(*registry_)),
      cache_miss_lookup_duration_us_model_family_(
          prometheus::BuildCounter()
              .Name("nv_cache_miss_lookup_duration_per_model")
              .Help(
                  "Total cache miss lookup duration per model, in microseconds")
              .Register(*registry_)),
      cache_miss_insertion_duration_us_model_family_(
          prometheus::BuildCounter()
              .Name("nv_cache_miss_insertion_duration_per_model")
              .Help("Total cache miss insertion duration per model, in "
                    "microseconds")
              .Register(*registry_)),

#ifdef TRITON_ENABLE_METRICS_GPU
      gpu_utilization_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_utilization")
                                  .Help("GPU utilization rate [0.0 - 1.0)")
                                  .Register(*registry_)),
      gpu_memory_total_family_(prometheus::BuildGauge()
                                   .Name("nv_gpu_memory_total_bytes")
                                   .Help("GPU total memory, in bytes")
                                   .Register(*registry_)),
      gpu_memory_used_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_memory_used_bytes")
                                  .Help("GPU used memory, in bytes")
                                  .Register(*registry_)),
      gpu_power_usage_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_power_usage")
                                  .Help("GPU power usage in watts")
                                  .Register(*registry_)),
      gpu_power_limit_family_(prometheus::BuildGauge()
                                  .Name("nv_gpu_power_limit")
                                  .Help("GPU power management limit in watts")
                                  .Register(*registry_)),
      gpu_energy_consumption_family_(
          prometheus::BuildCounter()
              .Name("nv_energy_consumption")
              .Help("GPU energy consumption in joules since the Triton Server "
                    "started")
              .Register(*registry_)),
#endif  // TRITON_ENABLE_METRICS_GPU
      metrics_enabled_(false), gpu_metrics_enabled_(false),
      cache_metrics_enabled_(false), metrics_interval_ms_(2000)
{
}

size_t
Metrics::HashLabels(const std::map<std::string, std::string>& labels)
{
  return prometheus::detail::hash_labels(labels);
}

Metrics::~Metrics()
{
  // Signal the cache thread to exit and then wait for it...
  if (poll_thread_ != nullptr) {
    poll_thread_exit_.store(true);
    poll_thread_->join();
#ifdef TRITON_ENABLE_METRICS_GPU
    if (dcgm_metadata_.dcgm_initialized_) {
      dcgmReturn_t derr;
      // Group destroy will return an error if groupId invalid or dcgm not
      // initialized or configured correctly
      derr = dcgmGroupDestroy(
          dcgm_metadata_.dcgm_handle_, dcgm_metadata_.groupId_);
      if (derr != DCGM_ST_OK) {
        LOG_WARNING << "Unable to destroy DCGM group: " << errorString(derr);
      }

      // Stop and shutdown DCGM
      if (dcgm_metadata_.standalone_) {
        derr = dcgmDisconnect(dcgm_metadata_.dcgm_handle_);
      } else {
        derr = dcgmStopEmbedded(dcgm_metadata_.dcgm_handle_);
      }
      if (derr != DCGM_ST_OK) {
        LOG_WARNING << "Unable to stop DCGM: " << errorString(derr);
      }
      derr = dcgmShutdown();
      if (derr != DCGM_ST_OK) {
        LOG_WARNING << "Unable to shutdown DCGM: " << errorString(derr);
      }
    }
#endif  // TRITON_ENABLE_METRICS_GPU
  }
}

bool
Metrics::Enabled()
{
  auto singleton = GetSingleton();
  return singleton->metrics_enabled_;
}

void
Metrics::EnableMetrics()
{
  auto singleton = GetSingleton();
  singleton->metrics_enabled_ = true;
}

void
Metrics::EnableCacheMetrics(
    std::shared_ptr<RequestResponseCache> response_cache)
{
  auto singleton = GetSingleton();
  // Ensure thread-safe enabling of Cache Metrics
  std::lock_guard<std::mutex> lock(singleton->cache_metrics_enabling_);
  if (singleton->cache_metrics_enabled_) {
    return;
  }

  // Setup metric families for cache metrics
  singleton->InitializeCacheMetrics(response_cache);

  // Toggle flag so this function is only executed once
  singleton->cache_metrics_enabled_ = true;
}

void
Metrics::EnableGPUMetrics()
{
  auto singleton = GetSingleton();

  // Ensure thread-safe enabling of GPU Metrics
  std::lock_guard<std::mutex> lock(singleton->gpu_metrics_enabling_);
  if (singleton->gpu_metrics_enabled_) {
    return;
  }

  if (std::getenv("TRITON_SERVER_CPU_ONLY") == nullptr) {
    singleton->InitializeDcgmMetrics();
  }

  singleton->gpu_metrics_enabled_ = true;
}

void
Metrics::SetMetricsInterval(uint64_t metrics_interval_ms)
{
  auto singleton = GetSingleton();
  singleton->metrics_interval_ms_ = metrics_interval_ms;
}

void
Metrics::StartPollingThreadSingleton(
    std::shared_ptr<RequestResponseCache> response_cache)
{
  auto singleton = GetSingleton();

  // Ensure thread-safe start of polling thread
  std::lock_guard<std::mutex> lock(singleton->poll_thread_starting_);
  if (singleton->poll_thread_started_) {
    return;
  }

  // Start thread for polling cache/dcgm metrics
  singleton->StartPollingThread(response_cache);

  // Toggle flag so this function is only executed once
  singleton->poll_thread_started_ = true;
}

bool
Metrics::StartPollingThread(
    std::shared_ptr<RequestResponseCache> response_cache)
{
  // Nothing to poll if no polling metrics enabled, don't spawn a thread
  if (!cache_metrics_enabled_ && !gpu_metrics_enabled_) {
    LOG_WARNING << "Neither cache metrics nor gpu metrics are enabled. Not "
                   "polling for them.";
    return false;
  }
  poll_thread_exit_.store(false);

  // Start a separate thread for polling metrics at specified interval
  poll_thread_.reset(new std::thread([this, response_cache] {
    // Thread will update metrics indefinitely until exit flag set
    while (!poll_thread_exit_.load()) {
      // Sleep for metric interval
      std::this_thread::sleep_for(
          std::chrono::milliseconds(metrics_interval_ms_ / 2));

      // Poll Response Cache metrics
      if (cache_metrics_enabled_ && response_cache != nullptr) {
        PollCacheMetrics(response_cache);
      }

#ifdef TRITON_ENABLE_METRICS_GPU
      // Poll DCGM GPU metrics
      if (gpu_metrics_enabled_ &&
          dcgm_metadata_.available_cuda_gpu_ids_.size() > 0) {
        PollDcgmMetrics();
      }
#endif  // TRITON_ENABLE_METRICS_GPU
    }
  }));

  return true;
}

bool
Metrics::PollCacheMetrics(std::shared_ptr<RequestResponseCache> response_cache)
{
  if (response_cache == nullptr) {
    LOG_WARNING << "error polling cache metrics, cache metrics will not be "
                << "available: cache was nullptr";
    return false;
  }

  // Update global cache metrics
  cache_num_entries_global_->Set(response_cache->NumEntries());
  cache_num_lookups_global_->Set(response_cache->NumLookups());
  cache_num_hits_global_->Set(response_cache->NumHits());
  cache_num_misses_global_->Set(response_cache->NumMisses());
  cache_num_evictions_global_->Set(response_cache->NumEvictions());
  cache_lookup_duration_us_global_->Set(
      response_cache->TotalLookupLatencyNs() / 1000);
  cache_insertion_duration_us_global_->Set(
      response_cache->TotalInsertionLatencyNs() / 1000);
  cache_util_global_->Set(response_cache->TotalUtilization());
  return true;
}

bool
Metrics::PollDcgmMetrics()
{
#ifndef TRITON_ENABLE_METRICS_GPU
  return false;
#else

  if (dcgm_metadata_.available_cuda_gpu_ids_.size() == 0) {
    LOG_WARNING << "error polling GPU metrics, GPU metrics will not be "
                << "available: no available gpus to poll";
    return false;
  }

  dcgmUpdateAllFields(dcgm_metadata_.dcgm_handle_, 1 /* wait for update*/);
  for (unsigned int didx = 0;
       didx < dcgm_metadata_.available_cuda_gpu_ids_.size(); ++didx) {
    uint32_t cuda_id = dcgm_metadata_.available_cuda_gpu_ids_[didx];
    if (dcgm_metadata_.cuda_ids_to_dcgm_ids_.count(cuda_id) <= 0) {
      LOG_WARNING << "Cannot find DCGM id for CUDA id " << cuda_id;
      continue;
    }
    uint32_t dcgm_id = dcgm_metadata_.cuda_ids_to_dcgm_ids_.at(cuda_id);
    dcgmFieldValue_v1 field_values[dcgm_metadata_.field_count_];
    dcgmReturn_t dcgmerr = dcgmGetLatestValuesForFields(
        dcgm_metadata_.dcgm_handle_, dcgm_id, dcgm_metadata_.fields_.data(),
        dcgm_metadata_.field_count_, field_values);

    if (dcgmerr != DCGM_ST_OK) {
      dcgm_metadata_.power_limit_fail_cnt_[didx]++;
      dcgm_metadata_.power_usage_fail_cnt_[didx]++;
      dcgm_metadata_.energy_fail_cnt_[didx]++;
      dcgm_metadata_.util_fail_cnt_[didx]++;
      dcgm_metadata_.mem_fail_cnt_[didx]++;
      LOG_WARNING << "Unable to get field values for GPU ID " << cuda_id << ": "
                  << errorString(dcgmerr);
    } else {
      // Power limit
      if (dcgm_metadata_.power_limit_fail_cnt_[didx] <
          dcgm_metadata_.fail_threshold_) {
        double power_limit = field_values[0].value.dbl;
        if ((field_values[0].status == DCGM_ST_OK) &&
            (!DCGM_FP64_IS_BLANK(power_limit))) {
          dcgm_metadata_.power_limit_fail_cnt_[didx] = 0;
        } else {
          dcgm_metadata_.power_limit_fail_cnt_[didx]++;
          power_limit = 0;
          dcgmReturn_t status = dcgmReturn_t(field_values[0].status);
          LOG_WARNING << "Unable to get power limit for GPU " << cuda_id
                      << ". Status:" << errorString(status)
                      << ", value:" << dcgmValueToErrorMessage(power_limit);
        }
        gpu_power_limit_[didx]->Set(power_limit);
      }

      // Power usage
      if (dcgm_metadata_.power_usage_fail_cnt_[didx] <
          dcgm_metadata_.fail_threshold_) {
        double power_usage = field_values[1].value.dbl;
        if ((field_values[1].status == DCGM_ST_OK) &&
            (!DCGM_FP64_IS_BLANK(power_usage))) {
          dcgm_metadata_.power_usage_fail_cnt_[didx] = 0;
        } else {
          dcgm_metadata_.power_usage_fail_cnt_[didx]++;
          power_usage = 0;
          dcgmReturn_t status = dcgmReturn_t(field_values[1].status);
          LOG_WARNING << "Unable to get power usage for GPU " << cuda_id
                      << ". Status:" << errorString(status)
                      << ", value:" << dcgmValueToErrorMessage(power_usage);
        }
        gpu_power_usage_[didx]->Set(power_usage);
      }

      // Energy Consumption
      if (dcgm_metadata_.energy_fail_cnt_[didx] <
          dcgm_metadata_.fail_threshold_) {
        int64_t energy = field_values[2].value.i64;
        if ((field_values[2].status == DCGM_ST_OK) &&
            (!DCGM_INT64_IS_BLANK(energy))) {
          dcgm_metadata_.energy_fail_cnt_[didx] = 0;
          if (dcgm_metadata_.last_energy_[didx] == 0) {
            dcgm_metadata_.last_energy_[didx] = energy;
          }
          gpu_energy_consumption_[didx]->Increment(
              (double)(energy - dcgm_metadata_.last_energy_[didx]) * 0.001);
          dcgm_metadata_.last_energy_[didx] = energy;
        } else {
          dcgm_metadata_.energy_fail_cnt_[didx]++;
          energy = 0;
          dcgmReturn_t status = dcgmReturn_t(field_values[2].status);
          LOG_WARNING << "Unable to get energy consumption for "
                      << "GPU " << cuda_id << ". Status:" << errorString(status)
                      << ", value:" << dcgmValueToErrorMessage(energy);
        }
      }

      // Utilization
      if (dcgm_metadata_.util_fail_cnt_[didx] <
          dcgm_metadata_.fail_threshold_) {
        int64_t util = field_values[3].value.i64;
        if ((field_values[3].status == DCGM_ST_OK) &&
            (!DCGM_INT64_IS_BLANK(util))) {
          dcgm_metadata_.util_fail_cnt_[didx] = 0;
        } else {
          dcgm_metadata_.util_fail_cnt_[didx]++;
          util = 0;
          dcgmReturn_t status = dcgmReturn_t(field_values[3].status);
          LOG_WARNING << "Unable to get GPU utilization for GPU " << cuda_id
                      << ". Status:" << errorString(status)
                      << ", value:" << dcgmValueToErrorMessage(util);
        }
        gpu_utilization_[didx]->Set((double)util * 0.01);
      }

      // Memory Usage
      if (dcgm_metadata_.mem_fail_cnt_[didx] < dcgm_metadata_.fail_threshold_) {
        int64_t memory_used = field_values[4].value.i64;
        int64_t memory_total = field_values[5].value.i64;
        if ((field_values[4].status == DCGM_ST_OK) &&
            (!DCGM_INT64_IS_BLANK(memory_used)) &&
            (field_values[5].status == DCGM_ST_OK) &&
            (!DCGM_INT64_IS_BLANK(memory_total))) {
          dcgm_metadata_.mem_fail_cnt_[didx] = 0;
        } else {
          memory_total = 0;
          memory_used = 0;
          dcgm_metadata_.mem_fail_cnt_[didx]++;
          dcgmReturn_t usageStatus = dcgmReturn_t(field_values[4].status);
          dcgmReturn_t memoryTotaltatus = dcgmReturn_t(field_values[5].status);
          LOG_WARNING << "Unable to get memory usage for GPU " << cuda_id
                      << ". Memory usage status:" << errorString(usageStatus)
                      << ", value:" << dcgmValueToErrorMessage(memory_used)
                      << ". Memory total status:"
                      << errorString(memoryTotaltatus)
                      << ", value:" << dcgmValueToErrorMessage(memory_total);
        }
        gpu_memory_total_[didx]->Set(memory_total * 1024 * 1024);  // bytes
        gpu_memory_used_[didx]->Set(memory_used * 1024 * 1024);    // bytes
      }
    }
  }
  return true;
#endif  // TRITON_ENABLE_METRICS_GPU
}

bool
Metrics::InitializeCacheMetrics(
    std::shared_ptr<RequestResponseCache> response_cache)
{
  if (response_cache == nullptr) {
    LOG_WARNING
        << "error initializing cache metrics, cache metrics will not be "
        << "available: cache was nullptr";
    return false;
  }

  const std::map<std::string, std::string> cache_labels;
  cache_num_entries_global_ = &cache_num_entries_family_.Add(cache_labels);
  cache_num_lookups_global_ = &cache_num_lookups_family_.Add(cache_labels);
  cache_num_hits_global_ = &cache_num_hits_family_.Add(cache_labels);
  cache_num_misses_global_ = &cache_num_misses_family_.Add(cache_labels);
  cache_num_evictions_global_ = &cache_num_evictions_family_.Add(cache_labels);
  cache_lookup_duration_us_global_ =
      &cache_lookup_duration_us_family_.Add(cache_labels);
  cache_insertion_duration_us_global_ =
      &cache_insertion_duration_us_family_.Add(cache_labels);
  cache_util_global_ = &cache_util_family_.Add(cache_labels);
  return true;
}

bool
Metrics::InitializeDcgmMetrics()
{
#ifndef TRITON_ENABLE_METRICS_GPU
  return false;
#else
  dcgmReturn_t dcgmerr = dcgmInit();
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "error initializing DCGM, GPU metrics will not be "
                << "available: " << errorString(dcgmerr);
    return false;
  }

  if (dcgm_metadata_.standalone_) {
    char hostIpAddress[16] = {0};
    std::string ipAddress = "127.0.0.1";
    strncpy(hostIpAddress, ipAddress.c_str(), 15);
    dcgmerr = dcgmConnect(hostIpAddress, &dcgm_metadata_.dcgm_handle_);
  } else {
    dcgmerr = dcgmStartEmbedded(
        DCGM_OPERATION_MODE_MANUAL, &dcgm_metadata_.dcgm_handle_);
  }
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "DCGM unable to start: " << errorString(dcgmerr);
    return false;
  } else {
    // Set this flag to signal DCGM cleanup in destructor
    dcgm_metadata_.dcgm_initialized_ = true;
  }

  if (dcgm_metadata_.standalone_) {
    dcgmerr = dcgmUpdateAllFields(dcgm_metadata_.dcgm_handle_, 1);
    if (dcgmerr != DCGM_ST_OK) {
      LOG_WARNING << "DCGM unable to update all fields, GPU metrics will "
                     "not be available: "
                  << errorString(dcgmerr);
      return false;
    }
  }

  unsigned int dcgm_gpu_ids[DCGM_MAX_NUM_DEVICES];
  int dcgm_gpu_count;
  dcgmerr = dcgmGetAllDevices(
      dcgm_metadata_.dcgm_handle_, dcgm_gpu_ids, &dcgm_gpu_count);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "DCGM unable to get device info and count, GPU "
                   "metrics will not be available: "
                << errorString(dcgmerr);
    return false;
  }

  // Get PCI Bus ID to DCGM device Id map.
  // Some devices may have problems using DCGM API and
  // these devices needs to be ignored.
  std::map<std::string, size_t> pci_bus_id_to_dcgm_id;
  std::map<std::string, std::map<std::string, std::string> >
      pci_bus_id_to_gpu_labels;
  std::map<std::string, std::string> pci_bus_id_to_device_name;
  dcgmDeviceAttributes_t gpu_attributes[DCGM_MAX_NUM_DEVICES];
  for (int i = 0; i < dcgm_gpu_count; i++) {
    gpu_attributes[i].version = dcgmDeviceAttributes_version;
    dcgmerr = dcgmGetDeviceAttributes(
        dcgm_metadata_.dcgm_handle_, dcgm_gpu_ids[i], &gpu_attributes[i]);
    if (dcgmerr != DCGM_ST_OK) {
      LOG_WARNING << "DCGM unable to get device properties for DCGM device "
                  << dcgm_gpu_ids[i]
                  << ", GPU metrics will not be available for this device: "
                  << errorString(dcgmerr);
    } else {
      std::string pciBusId = gpu_attributes[i].identifiers.pciBusId;
      pci_bus_id_to_dcgm_id[pciBusId] = i;
      pci_bus_id_to_device_name[pciBusId] =
          std::string(gpu_attributes[i].identifiers.deviceName);
      std::map<std::string, std::string> gpu_labels;
      gpu_labels.insert(std::map<std::string, std::string>::value_type(
          kMetricsLabelGpuUuid,
          std::string(gpu_attributes[i].identifiers.uuid)));
      pci_bus_id_to_gpu_labels[pciBusId] = gpu_labels;
    }
  }


  // Get CUDA-visible PCI Bus Ids and get DCGM metrics for each CUDA-visible GPU
  int cuda_gpu_count;
  cudaError_t cudaerr = cudaGetDeviceCount(&cuda_gpu_count);
  if (cudaerr != cudaSuccess) {
    LOG_WARNING
        << "Cannot get CUDA device count, GPU metrics will not be available";
    return false;
  }
  for (int i = 0; i < cuda_gpu_count; ++i) {
    std::string pci_bus_id = "0000";  // pad 0's for uniformity
    char pcibusid_str[64];
    cudaerr = cudaDeviceGetPCIBusId(pcibusid_str, sizeof(pcibusid_str) - 1, i);
    if (cudaerr == cudaSuccess) {
      pci_bus_id.append(pcibusid_str);
      if (pci_bus_id_to_dcgm_id.count(pci_bus_id) <= 0) {
        LOG_INFO << "Skipping GPU:" << i
                 << " since it's not CUDA enabled. This should never happen!";
        continue;
      }
      // Filter out CUDA visible GPUs from GPUs found by DCGM
      LOG_INFO << "Collecting metrics for GPU " << i << ": "
               << pci_bus_id_to_device_name[pci_bus_id];
      auto& gpu_labels = pci_bus_id_to_gpu_labels[pci_bus_id];
      gpu_utilization_.push_back(&gpu_utilization_family_.Add(gpu_labels));
      gpu_memory_total_.push_back(&gpu_memory_total_family_.Add(gpu_labels));
      gpu_memory_used_.push_back(&gpu_memory_used_family_.Add(gpu_labels));
      gpu_power_usage_.push_back(&gpu_power_usage_family_.Add(gpu_labels));
      gpu_power_limit_.push_back(&gpu_power_limit_family_.Add(gpu_labels));
      gpu_energy_consumption_.push_back(
          &gpu_energy_consumption_family_.Add(gpu_labels));
      uint32_t dcgm_id = pci_bus_id_to_dcgm_id[pci_bus_id];
      dcgm_metadata_.cuda_ids_to_dcgm_ids_[i] = dcgm_id;
      dcgm_metadata_.available_cuda_gpu_ids_.emplace_back(i);
    } else {
      LOG_WARNING << "GPU metrics will not be available for device:" << i;
    }
  }

  // create a gpu group
  char groupName[] = "dcgm_group";
  dcgmerr = dcgmGroupCreate(
      dcgm_metadata_.dcgm_handle_, DCGM_GROUP_DEFAULT, groupName,
      &dcgm_metadata_.groupId_);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "Cannot make GPU group: " << errorString(dcgmerr);
  }

  // Initialize tracking vectors
  for (unsigned int didx = 0;
       didx < dcgm_metadata_.available_cuda_gpu_ids_.size(); ++didx) {
    dcgm_metadata_.power_limit_fail_cnt_.push_back(0);
    dcgm_metadata_.power_usage_fail_cnt_.push_back(0);
    dcgm_metadata_.energy_fail_cnt_.push_back(0);
    dcgm_metadata_.util_fail_cnt_.push_back(0);
    dcgm_metadata_.mem_fail_cnt_.push_back(0);
    dcgm_metadata_.last_energy_.push_back(0);
  }

  // Number of fields for DCGM to use from fields_ below
  dcgm_metadata_.field_count_ = 6;
  unsigned short util_flag = dcgm_metadata_.standalone_
                                 ? DCGM_FI_PROF_GR_ENGINE_ACTIVE
                                 : DCGM_FI_DEV_GPU_UTIL;
  dcgm_metadata_.fields_ = {
      DCGM_FI_DEV_POWER_MGMT_LIMIT,          // power limit, watts
      DCGM_FI_DEV_POWER_USAGE,               // power usage, watts
      DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,  // Total energy consumption, mJ
      util_flag,                             // util ratio, 1 = 1%
      DCGM_FI_DEV_FB_USED,                   // Frame buffer used, MiB
      DCGM_FI_DEV_FB_TOTAL,                  // Frame buffer used, MiB
  };

  char fieldName[] = "field_group";
  dcgmFieldGrp_t fieldGroupId;
  dcgmerr = dcgmFieldGroupCreate(
      dcgm_metadata_.dcgm_handle_, dcgm_metadata_.field_count_,
      dcgm_metadata_.fields_.data(), fieldName, &fieldGroupId);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "Cannot make field group: " << errorString(dcgmerr);
  }

  dcgmerr = dcgmWatchFields(
      dcgm_metadata_.dcgm_handle_, dcgm_metadata_.groupId_, fieldGroupId,
      metrics_interval_ms_ * 1000 /*update period, usec*/,
      5.0 /*maxKeepAge, sec*/, 5 /*maxKeepSamples*/);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_WARNING << "Cannot start watching fields: " << errorString(dcgmerr);
    return false;
  }

  return true;
#endif  // TRITON_ENABLE_METRICS_GPU
}

#ifdef TRITON_ENABLE_METRICS_GPU
std::string
Metrics::dcgmValueToErrorMessage(double val)
{
  if (DCGM_FP64_IS_BLANK(val)) {
    if (val == DCGM_FP64_BLANK) {
      return "Not Specified";
    } else if (val == DCGM_FP64_NOT_FOUND) {
      return "Not Found";
    } else if (val == DCGM_FP64_NOT_SUPPORTED) {
      return "Not Supported";
    } else if (val == DCGM_FP64_NOT_PERMISSIONED) {
      return "Insf. Permission";
    } else {
      return "Unknown";
    }
  } else {
    return std::to_string(val);
  }
}

std::string
Metrics::dcgmValueToErrorMessage(int64_t val)
{
  if (DCGM_INT64_IS_BLANK(val)) {
    switch (val) {
      case DCGM_INT64_BLANK:
        return "Not Specified";
      case DCGM_INT64_NOT_FOUND:
        return "Not Found";
      case DCGM_INT64_NOT_SUPPORTED:
        return "Not Supported";
      case DCGM_INT64_NOT_PERMISSIONED:
        return "Insf. Permission";
      default:
        return "Unknown";
    }
  } else {
    return std::to_string(val);
  }
}
#endif  // TRITON_ENABLE_METRICS_GPU

bool
Metrics::UUIDForCudaDevice(int cuda_device, std::string* uuid)
{
  // If metrics were not initialized then just silently fail since
  // with DCGM we can't get the CUDA device (and not worth doing
  // anyway since metrics aren't being reported).
  auto singleton = GetSingleton();
  if (!singleton->gpu_metrics_enabled_) {
    return false;
  }

  // If GPU metrics is not enabled just silently fail.
#ifndef TRITON_ENABLE_METRICS_GPU
  return false;
#else

  dcgmDeviceAttributes_t gpu_attributes;
  gpu_attributes.version = dcgmDeviceAttributes_version;
  dcgmReturn_t dcgmerr = dcgmGetDeviceAttributes(
      singleton->dcgm_metadata_.dcgm_handle_, cuda_device, &gpu_attributes);
  if (dcgmerr != DCGM_ST_OK) {
    LOG_ERROR << "Unable to get device UUID: " << errorString(dcgmerr);
    return false;
  }

  *uuid = gpu_attributes.identifiers.uuid;
  return true;
#endif  // TRITON_ENABLE_METRICS_GPU
}

std::shared_ptr<prometheus::Registry>
Metrics::GetRegistry()
{
  auto singleton = Metrics::GetSingleton();
  return singleton->registry_;
}

const std::string
Metrics::SerializedMetrics()
{
  auto singleton = Metrics::GetSingleton();
  return singleton->serializer_->Serialize(
      singleton->registry_.get()->Collect());
}

Metrics*
Metrics::GetSingleton()
{
  static Metrics singleton;
  return &singleton;
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METRICS
