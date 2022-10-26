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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "constants.h"
#include "infer_request.h"
#include "infer_response.h"
#include "server_message.h"
#include "status.h"
#include "triton/common/model_config.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

//
// Proxy to a cache shared library.
//
class TritonCache {
 public:
  static Status Create(
      const std::string& name, const std::string& dir,
      const std::string& libpath, const TritonServerMessage* cache_config,
      std::shared_ptr<TritonCache>* cache);
  ~TritonCache();

  const std::string& Name() const { return name_; }
  const std::string& Directory() const { return dir_; }
  const TritonServerMessage* CacheConfig() const { return cache_config_; }
  // TODO
  Status Insert(const InferenceResponse& response, uint64_t key);
  Status Lookup(InferenceResponse* response, uint64_t key);
  Status Hash(const InferenceRequest& request, uint64_t* key);
  Status Evict();

 private:
  TritonCache(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const TritonServerMessage* cache_config);

  void ClearHandles();
  Status LoadCacheLibrary();

  // TODO: needed?
  // The name of the cache.
  const std::string name_;

  // Full path to the directory holding cache shared library and
  // other artifacts.
  const std::string dir_;

  // Full path to the cache shared library.
  const std::string libpath_;

  // Cache configuration as JSON
  // TODO: const ref over ptr
  const TritonServerMessage* cache_config_;

  // dlopen / dlsym handles
  void* dlhandle_;
  // TODO: Create types for each function?
  std::function<void()> cache_init_fn_;
  std::function<void()> cache_fini_fn_;
  // TODO: Hash/Insert/Lookup
};

//
// Manage communication with Triton caches and their lifecycle.
//
class TritonCacheManager {
 public:
  static Status Create(std::shared_ptr<TritonCacheManager>* manager);

  Status CreateCache(
      const std::string& name, const std::string& dir,
      const std::string& libpath,
      const TritonServerMessage* cache_config,
      std::shared_ptr<TritonCache>* cache);

  std::shared_ptr<TritonCache> Cache() { return cache_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonCacheManager);
  TritonCacheManager() = default;
  std::shared_ptr<TritonCache> cache_;
};

}}  // namespace triton::core