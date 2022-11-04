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
  auto lcache = std::shared_ptr<TritonCache>(
      new TritonCache(name, dir, libpath, cache_config));

  RETURN_IF_ERROR(lcache->LoadCacheLibrary());
  RETURN_IF_ERROR(lcache->InitializeCacheImpl());
  RETURN_IF_ERROR(lcache->TestCacheImpl());  // TODO: Remove

  *cache = std::move(lcache);
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
  if (fini_fn_ != nullptr) {
    LOG_VERBOSE(1) << "Calling TRITONCACHE_CacheDelete from: '" << libpath_
                   << "'";
    LOG_TRITONSERVER_ERROR(fini_fn_(cache_impl_), "failed finalizing cache");
  } else {
    LOG_ERROR << "cache finalize function is nullptr";
  }

  ClearHandles();
}

void
TritonCache::ClearHandles()
{
  dlhandle_ = nullptr;
  cache_impl_ = nullptr;
  init_fn_ = nullptr;
  fini_fn_ = nullptr;
  lookup_fn_ = nullptr;
}

Status
TritonCache::LoadCacheLibrary()
{
  LOG_VERBOSE(1) << "Loading cache library: '" << name_ << "'";
  TritonCacheInitFn_t init_fn;
  TritonCacheFiniFn_t fini_fn;
  TritonCacheLookupFn_t lookup_fn;

  // Load the library and initialize all the entrypoints
  {
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->OpenLibraryHandle(libpath_, &dlhandle_));

    // Cache initialize and finalize functions, required
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheNew", false /* optional */,
        reinterpret_cast<void**>(&init_fn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheDelete", false /* optional */,
        reinterpret_cast<void**>(&fini_fn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheLookup", false /* optional */,
        reinterpret_cast<void**>(&lookup_fn)));
  }

  init_fn_ = init_fn;
  fini_fn_ = fini_fn;
  lookup_fn_ = lookup_fn;
  return Status::Success;
}

Status
TritonCache::TestCacheImpl()
{
  std::cout << "======================================" << std::endl;
  std::cout << "==== Testing Cache Implementation ====" << std::endl;
  std::cout << "======================================" << std::endl;
  // TODO: Remove
  auto status = Status::Success;
  InferenceResponse* response = nullptr;
  std::cout << "=============== Lookup ===============" << std::endl;
  status = Lookup(response, "test_key_123");
  if (!status.IsOk()) {
    return status;
  }

  std::cout << "======================================" << std::endl;
  std::cout << "============ Done Testing ============" << std::endl;
  std::cout << "======================================" << std::endl;
  return Status::Success;
}

Status
TritonCache::InitializeCacheImpl()
{
  // TODO: Shouldn't be needed since SharedLibrary should error
  //       if non-optional function not found / nullptr
  if (init_fn_ == nullptr) {
    return Status(Status::Code::NOT_FOUND, "cache init function is nullptr");
  }

  // Initialize cache implementation
  LOG_VERBOSE(1) << "Calling TRITONCACHE_CacheNew from: '" << libpath_ << "'";
  RETURN_IF_TRITONSERVER_ERROR(init_fn_(&cache_impl_));
  if (cache_impl_ == nullptr) {
    return Status(
        Status::Code::INTERNAL, "Failed to initialize cache implementation");
  }

  return Status::Success;
}

Status
TritonCache::Hash(const InferenceRequest& request, uint64_t* key)
{
  LOG_VERBOSE(1) << "Hashing into cache";
  // TODO: call cache_hash_fn__
  *key = 42;
  return Status::Success;
}

Status
TritonCache::Insert(const InferenceResponse& response, uint64_t key)
{
  LOG_VERBOSE(1) << "Inserting into cache";
  // TODO: call cache_insert_fn_
  return Status(Status::Code::INTERNAL, "Insert Not Implemented");
}

Status
TritonCache::Lookup(InferenceResponse* response, const std::string& key)
{
  LOG_VERBOSE(1) << "Looking up in cache";
  if (lookup_fn_ == nullptr) {
    return Status(Status::Code::NOT_FOUND, "cache lookup function is nullptr");
  }

  // Initialize cache implementation
  LOG_VERBOSE(1) << "Calling TRITONCACHE_CacheLookup from: '" << libpath_
                 << "'";
  void* entries = nullptr;
  size_t* sizes = nullptr;
  size_t num_entries = 42;  // TODO: should be reset
  RETURN_IF_TRITONSERVER_ERROR(
      lookup_fn_(cache_impl_, key.c_str(), &entries, &sizes, &num_entries));

  LOG_VERBOSE(1) << "Lookup: num_entries: " << num_entries;
  // TODO: Build Inference Response
  return Status::Success;
}

Status
TritonCache::Evict()
{
  LOG_VERBOSE(1) << "Evicting from cache";
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

  RETURN_IF_ERROR(
      TritonCache::Create(name, dir, libpath, cache_config, &cache_));
  *cache = cache_;
  return Status::Success;
}

}}  // namespace triton::core
