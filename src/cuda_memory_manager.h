// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include "status.h"

namespace triton { namespace core {

// This is a singleton class responsible for maintaining CUDA memory pool
// used by the inference server. CUDA memory allocations and deallocations
// must be requested via functions provided by this class.
class CudaMemoryManager {
 public:
  // Options to configure CUDA memory manager.
  struct Options {
    Options(double cc = 6.0, const std::map<int, uint64_t>& s = {})
        : min_supported_compute_capability_(cc), memory_pool_byte_size_(s)
    {
    }

    // The minimum compute capability of the supported devices.
    double min_supported_compute_capability_;

    // The size of CUDA memory reserved for the specified devices.
    // The memory size will be rounded up to align with
    // the default granularity (512 bytes).
    // No memory will be reserved for devices that is not listed.
    std::map<int, uint64_t> memory_pool_byte_size_;
  };

  ~CudaMemoryManager();

  // Create the memory manager based on 'options' specified.
  // Return Status object indicating success or failure.
  static Status Create(const Options& options);

  // Provide explicit control on ending the memory manager lifecycle,
  // CUDA resource must be cleaned up before CUDA context is destroyed.
  static void Reset();

  // Allocate CUDA memory on GPU 'device_id' with
  // the requested 'size' and return the pointer in 'ptr'.
  // Return Status object indicating success or failure.
  static Status Alloc(void** ptr, uint64_t size, int64_t device_id);

  // Free the memory allocated by the memory manager on 'device_id'.
  // Return Status object indicating success or failure.
  static Status Free(void* ptr, int64_t device_id);

 private:
  CudaMemoryManager(bool has_allocation) : has_allocation_(has_allocation) {}
  bool has_allocation_;
  static std::unique_ptr<CudaMemoryManager> instance_;
  static std::mutex instance_mu_;
};

}}  // namespace triton::core
