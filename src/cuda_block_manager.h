// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "constants.h"
#include "status.h"


namespace triton { namespace core {

class Allocation;

///
/// CudaBlockManager uses CUDA Virtual Memory Management APIs to allow the
/// buffers to grow without having to create a new allocation. Currently,
/// it is only used with implicit state management.
///
class CudaBlockManager {
 public:
  static Status Create(double min_supported_compute_capability);
  static Status Allocate(
      size_t size, std::unique_ptr<Allocation>& allocation, int device_id);
  static Status Free(Allocation* allocation, int device_id);
  static size_t BlockSize() { return instance_->block_size_; }
  ~CudaBlockManager();
  static void Reset() { instance_.reset(); }

 private:
  CudaBlockManager() {};
  std::unordered_map<int, std::vector<CUmemGenericAllocationHandle>>
      free_blocks_;
  size_t block_size_;
  std::mutex mu_;
  static std::unique_ptr<CudaBlockManager> instance_;
};

///
/// Representing an Allocation object returned from CudaBlockManager.
///
class Allocation {
 public:
  Allocation(int device_id) : device_id_(device_id) {}

  /// Add a block to an existing allocation.
  ///
  /// \param block The memory block to be added.
  ///
  void AddBlock(CUmemGenericAllocationHandle block)
  {
    blocks_.push_back(block);
  }

  /// Merge one allocation with another one.
  ///
  /// \param allocation The other allocation to be merged with.
  ///
  void Merge(std::unique_ptr<Allocation>&& allocation);

  /// Get the list of all the memory blocks corresponding to this
  /// allocation.
  ///
  /// \return The list of all the memory blocks.
  std::vector<CUmemGenericAllocationHandle>& Blocks() { return blocks_; }

  ~Allocation() { CudaBlockManager::Free(this, device_id_); }

 private:
  DISALLOW_COPY_AND_ASSIGN(Allocation);
  std::vector<CUmemGenericAllocationHandle> blocks_;
  int device_id_;
};
}};  // namespace triton::core
