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
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cache_entry.h"
#include "constants.h"
#include "infer_request.h"
#include "infer_response.h"
#include "server_message.h"
#include "status.h"
#include "triton/common/model_config.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

//
// Custom Allocators to copy directly between Cache buffers <-> Triton buffers
// and avoid intermediate copies on Insert/Lookup.
//
class TritonCacheAllocator {
 public:
  virtual Status Allocate(TRITONCACHE_CacheEntry* entry) = 0;
};

class CacheToResponseAllocator : TritonCacheAllocator {
 public:
  CacheToResponseAllocator(boost::span<InferenceResponse*> responses);
  Status Allocate(TRITONCACHE_CacheEntry* entry);

 private:
  std::vector<InferenceResponse*> responses_;
};

class ResponseToCacheAllocator : TritonCacheAllocator {
 public:
  ResponseToCacheAllocator(boost::span<InferenceResponse*> responses);
  Status Allocate(TRITONCACHE_CacheEntry* entry);

 private:
  std::vector<InferenceResponse*> responses_;
};

// NOTE: Bytes-related allocators only used for unit testing currently
class CacheToBytesAllocator : TritonCacheAllocator {
 public:
  Status Allocate(TRITONCACHE_CacheEntry* entry);
};

class BytesToCacheAllocator : TritonCacheAllocator {
 public:
  BytesToCacheAllocator(std::vector<boost::span<Byte>> buffers);
  Status Allocate(TRITONCACHE_CacheEntry* entry);

 private:
  std::vector<boost::span<Byte>> buffers_;
};


//
// Proxy to a cache shared library.
//
class TritonCache {
 public:
  static Status Create(
      const std::string& name, const std::string& libpath,
      const std::string& cache_config, std::shared_ptr<TritonCache>* cache);
  ~TritonCache();

  const std::string& Name() const { return name_; }
  const std::string& CacheConfig() const { return cache_config_; }
  Status Insert(InferenceResponse* response, const std::string& key);
  Status Insert(
      boost::span<InferenceResponse*> responses, const std::string& key);
  Status Insert(std::vector<boost::span<Byte>> buffers, const std::string& key);
  Status Insert(
      CacheEntry* entry, const std::string& key,
      TRITONCACHE_Allocator* allocator);
  Status Lookup(InferenceResponse* response, const std::string& key);
  Status Lookup(
      boost::span<InferenceResponse*> responses, const std::string& key);
  Status Lookup(const std::string& key, CacheEntry* entry);
  Status Lookup(
      const std::string& key, CacheEntry* entry,
      TRITONCACHE_Allocator* allocator);
  // Hashes fields of request and stores it in "key"
  Status Hash(const InferenceRequest& request, std::string* key);

 private:
  TritonCache(
      const std::string& name, const std::string& libpath,
      const std::string& cache_config);

  void ClearHandles();
  Status LoadCacheLibrary();
  Status InitializeCacheImpl();
  // Helper function to hash data buffers used by "input"
  Status HashInputBuffers(const InferenceRequest::Input* input, size_t* seed);
  // Helper function to hash each input in "request"
  Status HashInputs(const InferenceRequest& request, size_t* seed);

  // The name of the cache.
  const std::string name_;

  // Full path to the cache shared library.
  const std::string libpath_;

  // Cache configuration as JSON string
  const std::string cache_config_;

  // Cache Implementation
  TRITONCACHE_Cache* cache_impl_;

  // dlopen / dlsym handles
  void* dlhandle_;
  typedef TRITONSERVER_Error* (*TritonCacheInitFn_t)(
      TRITONCACHE_Cache** cache, const char* config);
  TritonCacheInitFn_t init_fn_;
  typedef TRITONSERVER_Error* (*TritonCacheFiniFn_t)(TRITONCACHE_Cache* cache);
  TritonCacheFiniFn_t fini_fn_;
  typedef TRITONSERVER_Error* (*TritonCacheLookupFn_t)(
      TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry,
      TRITONCACHE_Allocator* allocator);
  TritonCacheLookupFn_t lookup_fn_;
  typedef TRITONSERVER_Error* (*TritonCacheInsertFn_t)(
      TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry,
      TRITONCACHE_Allocator* allocator);
  TritonCacheInsertFn_t insert_fn_;
};

//
// Manage communication with Triton caches and their lifecycle.
//
class TritonCacheManager {
 public:
  static Status Create(
      std::shared_ptr<TritonCacheManager>* manager, std::string cache_dir);

  Status CreateCache(
      const std::string& name, const std::string& cache_config,
      std::shared_ptr<TritonCache>* cache);

  std::shared_ptr<TritonCache> Cache() { return cache_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonCacheManager);
  TritonCacheManager(std::string cache_dir) : cache_dir_(cache_dir) {}
  // Global search path for cache libraries
  std::string cache_dir_;
  // This may be a map of caches in the future
  std::shared_ptr<TritonCache> cache_;
};

}}  // namespace triton::core
