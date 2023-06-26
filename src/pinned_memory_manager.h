// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/interprocess/managed_external_buffer.hpp>
#include <map>
#include <memory>
#include <mutex>

#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

// This is a singleton class responsible for maintaining pinned memory pool
// used by the inference server. Pinned memory allocations and deallocations
// must be requested via functions provided by this class.
class PinnedMemoryManager {
 public:
  // Options to configure pinned memory manager.
  struct Options {
    Options(
        uint64_t b = 0,
        const triton::common::HostPolicyCmdlineConfigMap& host_policy_map = {})
        : pinned_memory_pool_byte_size_(b), host_policy_map_(host_policy_map)
    {
    }

    uint64_t pinned_memory_pool_byte_size_;
    triton::common::HostPolicyCmdlineConfigMap host_policy_map_;
  };

  ~PinnedMemoryManager();

  // Create the pinned memory manager based on 'options' specified.
  // Return Status object indicating success or failure.
  static Status Create(const Options& options);

  // Provide explicit control on ending the memory manager lifecycle,
  // CUDA resource must be cleaned up before CUDA context is destroyed.
  static void Reset();

  // Allocate pinned memory with the requested 'size' and return the pointer
  // in 'ptr'. If 'allow_nonpinned_fallback' is true, regular system memory
  // will be allocated as fallback in the case where pinned memory fails to
  // be allocated.
  // Return Status object indicating success or failure.
  static Status Alloc(
      void** ptr, uint64_t size, TRITONSERVER_MemoryType* allocated_type,
      bool allow_nonpinned_fallback);

  // Free the memory allocated by the pinned memory manager.
  // Return Status object indicating success or failure.
  static Status Free(void* ptr);

 private:
  class PinnedMemory {
   public:
    PinnedMemory(void* pinned_memory_buffer, uint64_t size);
    ~PinnedMemory();
    void* pinned_memory_buffer_;
    std::mutex buffer_mtx_;
    boost::interprocess::managed_external_buffer managed_pinned_memory_;
  };

  PinnedMemoryManager() = default;

  Status AllocInternal(
      void** ptr, uint64_t size, TRITONSERVER_MemoryType* allocated_type,
      bool allow_nonpinned_fallback, PinnedMemory* pinned_memory_buffer);
  Status FreeInternal(void* ptr);
  void AddPinnedMemoryBuffer(
      const std::shared_ptr<PinnedMemory>& pinned_memory_buffer,
      unsigned long node_mask);

  static std::unique_ptr<PinnedMemoryManager> instance_;
  static uint64_t pinned_memory_byte_size_;

  std::mutex info_mtx_;
  std::map<void*, std::pair<bool, PinnedMemory*>> memory_info_;
  std::map<unsigned long, std::shared_ptr<PinnedMemory>> pinned_memory_buffers_;
};

}}  // namespace triton::core
