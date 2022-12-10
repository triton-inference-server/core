// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <shared_mutex>
#include <vector>
#include "infer_response.h"
#include "triton/common/logging.h"
#include "tritonserver_apis.h"

// If TRITONSERVER error is non-OK, return std::nullopt for failed optional.
#define RETURN_NULLOPT_IF_TRITONSERVER_ERROR(E)      \
  do {                                               \
    TRITONSERVER_Error* err__ = (E);                 \
    if (err__ != nullptr) {                          \
      LOG_ERROR << TRITONSERVER_ErrorMessage(err__); \
      TRITONSERVER_ErrorDelete(err__);               \
      return std::nullopt;                           \
    }                                                \
  } while (false)

#define RETURN_NULLOPT_IF_STATUS_ERROR(S) \
  do {                                    \
    const Status& status__ = (S);         \
    if (!status__.IsOk()) {               \
      LOG_ERROR << status__.Message();    \
      return std::nullopt;                \
    }                                     \
  } while (false)

namespace triton { namespace core {

// For debugging
void printBytes(boost::span<const std::byte> buffer);

struct CacheOutput {
  std::string name_;
  inference::DataType dtype_;
  std::vector<int64_t> shape_;
  std::vector<std::byte> buffer_;
};

// A Buffer is an arbitrary data blob, whose type need not be known
// by the cache for storage and retrieval.
using Buffer = std::vector<std::byte>;

// A CacheEntryItem may have several Buffers associated with it
//   ex: one response may have multiple output buffers
class CacheEntryItem {
 public:
  Status FromResponse(const InferenceResponse* response);
  Status ToResponse(InferenceResponse* response);
  std::vector<Buffer> Buffers();
  size_t BufferCount();
  void AddBuffer(boost::span<const std::byte> buffer);

 private:
  std::optional<Buffer> ToBytes(const InferenceResponse::Output& output);
  std::optional<CacheOutput> FromBytes(
      boost::span<const std::byte> packed_bytes);


  // NOTE: performance gain may be possible by removing this mutex and
  //   guaranteeing that no two threads will access/modify an item
  //   in parallel. This may be guaranteed by documenting that a cache
  //   implementation should not call TRITONCACHE_CacheEntryItemAddBuffer or
  //   TRITONCACHE_CacheEntryItemBuffer on the same item in parallel.
  //   This will remain for simplicity until further profiling is done.
  // Shared mutex to support read-only and read-write locks
  std::shared_mutex buffer_mu_;
  std::vector<Buffer> buffers_;
};

// A CacheEntry may have several Items associated with it
//   ex: one request may have multiple responses
class CacheEntry {
 public:
  std::vector<std::shared_ptr<CacheEntryItem>> Items();
  size_t ItemCount();
  void AddItem(std::shared_ptr<CacheEntryItem> item);

 private:
  // NOTE: performance gain may be possible by removing this mutex and
  //   guaranteeing that no two threads will access/modify an entry
  //   in parallel. This may be guaranteed by documenting that a cache
  //   implementation should not call TRITONCACHE_CacheEntryAddItem or
  //   TRITONCACHE_CacheEntryItem on the same entry in parallel.
  //   This will remain for simplicity until further profiling is done.
  // Shared mutex to support read-only and read-write locks
  std::shared_mutex item_mu_;
  std::vector<std::shared_ptr<CacheEntryItem>> items_;
};

}}  // namespace triton::core