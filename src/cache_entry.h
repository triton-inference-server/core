// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <boost/core/span.hpp>
#include <memory>
#include <mutex>
#include <vector>

#include "infer_response.h"
#include "triton/common/logging.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

struct CacheOutput {
  // Inference Response output name
  std::string name_ = "";
  // Inference Response output datatype
  inference::DataType dtype_;
  // Inference Response output shape
  std::vector<int64_t> shape_;
  // Inference Response output buffer
  // NOTE: Cache Output will only temporarily store pointer, it will managed by
  //       CacheEntry, and will be copied into InferenceResponse object
  void* buffer_ = nullptr;
  // Inference Response output buffer size
  uint64_t byte_size_ = 0;
};

// A Buffer is an arbitrary data blob whose type need not be known
// by the cache for storage and retrieval.
using Buffer = std::pair<void*, size_t>;

// A CacheEntrymay have several Buffers associated with it
//   ex: one request may have multiple responses
class CacheEntry {
 public:
  ~CacheEntry();
  const std::vector<Buffer>& Buffers();
  std::vector<Buffer>& MutableBuffers();
  size_t BufferCount();
  void AddBuffer(boost::span<Byte> buffer);
  void AddBuffer(void* base, size_t byte_size);

  /* Insert helpers */
  Status SerializeResponses(boost::span<InferenceResponse*> responses);

  // Adds a placeholder buffer to this CacheEntry object containing the
  // necessary size to hold the data we plan to insert into the cache.
  // - The 'nullptr' base indicates to the cache implementation that we want
  // the cache to allocate the buffer and overwrite this buffer's base address
  // with the cache-allocated buffer's base address in-place.
  // - This lets Triton copy directly into the cache allocated buffer through
  // the TRITONCACHE_Copy callback.
  void AddPlaceholderBuffer(size_t byte_size);

  // Calculates serialized response size to request allocated buffer from cache
  // and sets entry buffer sizes
  Status SetBufferSizes(boost::span<InferenceResponse*> responses);
  // Directly sets entry buffer sizes from provided buffers
  Status SetBufferSizes(std::vector<boost::span<Byte>> buffers);

  /* Lookup helpers */
  Status DeserializeBuffers(boost::span<InferenceResponse*> responses);

  // Typically, the cache entry will now own any associated buffers.
  // However, if a CacheAllocator wants the entry to own the buffers, this
  // can be used to signal that the entry should free its buffers on destruction
  void FreeBuffersOnExit() { free_buffers_ = true; }

 private:
  // Insert helpers
  Status SerializeResponse(InferenceResponse* response, Buffer& buffer);
  Status SerializeResponseOutput(
      const InferenceResponse::Output& output, Byte* buffer,
      size_t* output_size);
  Status SetBufferSize(InferenceResponse* response);
  // Calculates total byte size required to serialize response output and
  // returns it in packed_output_byte_size
  Status GetByteSize(
      const InferenceResponse::Output& output,
      uint64_t* packed_output_byte_size);

  // Lookup helpers
  Status DeserializeBuffer(InferenceResponse* response, const Buffer& buffer);
  Status DeserializeResponseOutput(
      boost::span<const Byte> packed_bytes, CacheOutput* output);

  // NOTE: performance gain may be possible by removing this mutex and
  //   guaranteeing that no two threads will access/modify an entry
  //   in parallel. This may be guaranteed by documenting that a cache
  //   implementation should not call TRITONCACHE_CacheEntryAddBuffer or
  //   TRITONCACHE_CacheEntryBuffer on the same entry in parallel.
  //   This will remain for simplicity until further profiling is done.
  std::mutex buffer_mu_;
  std::vector<Buffer> buffers_;
  // Free buffers on exit, default is false unless explicitly toggled
  bool free_buffers_ = false;
};

}}  // namespace triton::core
