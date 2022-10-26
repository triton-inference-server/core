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

#include "cache_manager.h"
#include "server_message.h"
#include "shared_library.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

//
// TritonCache
//
Status
TritonCache::Create(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const TritonServerMessage* cache_config,
    std::shared_ptr<TritonCache>* cache)
{
  auto local_cache = std::shared_ptr<TritonCache>(
      new TritonCache(name, dir, libpath, cache_config));

  // Load the library and initialize all the entrypoints
  RETURN_IF_ERROR(local_cache->LoadCacheLibrary());

  if (local_cache->cache_init_fn_ != nullptr) {
    // TODO: Implement functions and use shared library
  }

  *cache = std::move(local_cache);
  return Status::Success;
}

TritonCache::TritonCache(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const TritonServerMessage* cache_config)
    : name_(name), dir_(dir), libpath_(libpath), cache_config_(cache_config)
{
  ClearHandles();
}

TritonCache::~TritonCache()
{
  LOG_VERBOSE(1) << "unloading cache '" << name_ << "'";
  // TODO: Finalization/delete function
  ClearHandles();
}

void
TritonCache::ClearHandles()
{
  dlhandle_ = nullptr;
  cache_init_fn_ = nullptr;
  cache_fini_fn_ = nullptr;
}

Status
TritonCache::LoadCacheLibrary()
{
  // TODO
  return Status::Success;
}

Status
TritonCache::Hash(const InferenceRequest& request, uint64_t* key)
{
  // TODO: call cache_hash_fn__
  *key = 42;
  return Status::Success;
}

Status
TritonCache::Insert(const InferenceResponse& response, uint64_t key)
{
  // TODO: call cache_insert_fn_
  return Status(Status::Code::INTERNAL, "Insert Not Implemented");
}

Status
TritonCache::Lookup(InferenceResponse* response, uint64_t key)
{
  // TODO: call cache_lookup_fn_
  return Status(Status::Code::INTERNAL, "Lookup Not Implemented");
}

Status
TritonCache::Evict()
{
  // TODO: call cache_evict_fn_
  return Status(Status::Code::INTERNAL, "Evict Not Implemented");
}

//
// TritonCacheManager
//

// TODO: Weak Ptr (Backends) vs Singleton (Repo Agent) ?
static std::weak_ptr<TritonCacheManager> cache_manager_;
static std::mutex mu_;

Status
TritonCacheManager::Create(std::shared_ptr<TritonCacheManager>* manager)
{
  std::lock_guard<std::mutex> lock(mu_);

  // If there is already a manager then we just use it
  *manager = cache_manager_.lock();
  if (*manager != nullptr) {
    return Status::Success;
  }

  manager->reset(new TritonCacheManager());
  cache_manager_ = *manager;

  return Status::Success;
}

Status
TritonCacheManager::CreateCache(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const TritonServerMessage* cache_config,
    std::shared_ptr<TritonCache>* cache)
{
  std::lock_guard<std::mutex> lock(mu_);

  // TODO: Do we want to maintain a map, or only manage a single cache?
  if (cache_ != nullptr) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "TritonCacheManager already holds a cache");
  }

  RETURN_IF_ERROR(TritonCache::Create(name, dir, libpath, cache_config, &cache_));
  *cache = cache_;
  return Status::Success;
}

}}  // namespace triton::core
