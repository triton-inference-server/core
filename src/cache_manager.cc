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

#include "cache_manager.h"
#include <boost/functional/hash.hpp>
#include "cache_entry.h"
#include "filesystem.h"
#include "server_message.h"
#include "shared_library.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

std::string
TritonCacheLibraryName()
{
#ifdef _WIN32
  return std::string("tritoncache.dll");
#else
  return std::string("libtritoncache.so");
#endif
}

//
// TritonCache
//
Status
TritonCache::Create(
    const std::string& name, const std::string& libpath,
    const std::string& cache_config, std::shared_ptr<TritonCache>* cache)
{
  LOG_INFO << "Creating TritonCache with name: '" << name << "', libpath: '"
           << libpath << "', cache_config: '" << cache_config << "'";

  auto lcache = std::shared_ptr<TritonCache>(
      new TritonCache(name, libpath, cache_config));

  RETURN_IF_ERROR(lcache->LoadCacheLibrary());
  RETURN_IF_ERROR(lcache->InitializeCacheImpl());

  *cache = std::move(lcache);
  return Status::Success;
}

TritonCache::TritonCache(
    const std::string& name, const std::string& libpath,
    const std::string& cache_config)
    : name_(name), libpath_(libpath), cache_config_(cache_config)
{
  ClearHandles();
}

TritonCache::~TritonCache()
{
  LOG_VERBOSE(1) << "unloading cache '" << name_ << "'";
  if (fini_fn_) {
    if (cache_impl_) {
      LOG_VERBOSE(1) << "Calling TRITONCACHE_CacheFinalize from: '" << libpath_
                     << "'";
      LOG_TRITONSERVER_ERROR(fini_fn_(cache_impl_), "failed finalizing cache");
    } else {
      LOG_ERROR << "cache implementation handle is nullptr";
    }
  } else {
    LOG_ERROR << "cache finalize function is nullptr";
  }

  if (dlhandle_) {
    std::unique_ptr<SharedLibrary> slib;
    LOG_STATUS_ERROR(SharedLibrary::Acquire(&slib), "~TritonCache");
    LOG_STATUS_ERROR(slib->CloseLibraryHandle(dlhandle_), "~TritonCache");
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
  insert_fn_ = nullptr;
}

Status
TritonCache::LoadCacheLibrary()
{
  LOG_VERBOSE(1) << "Loading cache library: '" << name_ << "' from: '"
                 << libpath_ << "'";
  TritonCacheInitFn_t init_fn;
  TritonCacheFiniFn_t fini_fn;
  TritonCacheLookupFn_t lookup_fn;
  TritonCacheInsertFn_t insert_fn;

  // Load the library and initialize all the entrypoints
  {
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->OpenLibraryHandle(libpath_, &dlhandle_));

    // Cache initialize and finalize functions, required
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheInitialize", false /* optional */,
        reinterpret_cast<void**>(&init_fn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheFinalize", false /* optional */,
        reinterpret_cast<void**>(&fini_fn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheLookup", false /* optional */,
        reinterpret_cast<void**>(&lookup_fn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONCACHE_CacheInsert", false /* optional */,
        reinterpret_cast<void**>(&insert_fn)));
  }

  init_fn_ = init_fn;
  fini_fn_ = fini_fn;
  lookup_fn_ = lookup_fn;
  insert_fn_ = insert_fn;
  return Status::Success;
}

Status
TritonCache::InitializeCacheImpl()
{
  if (init_fn_ == nullptr) {
    return Status(Status::Code::NOT_FOUND, "cache init function is nullptr");
  }
  // Initialize cache implementation
  RETURN_IF_TRITONSERVER_ERROR(init_fn_(&cache_impl_, cache_config_.c_str()));

  if (!cache_impl_) {
    return Status(
        Status::Code::INTERNAL, "Failed to initialize cache implementation");
  }

  return Status::Success;
}

Status
TritonCache::HashInputBuffers(
    const InferenceRequest::Input* input, size_t* seed)
{
  // Iterate over each data buffer in input in case of non-contiguous memory
  for (size_t idx = 0; idx < input->DataBufferCount(); ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    RETURN_IF_ERROR(input->DataBuffer(
        idx, &src_buffer, &src_byte_size, &src_memory_type,
        &src_memory_type_id));

    if (src_memory_type != TRITONSERVER_MEMORY_CPU &&
        src_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    // Add each byte of input buffer chunk to hash
    const unsigned char* tmp = static_cast<const unsigned char*>(src_buffer);
    for (uint64_t byte = 0; byte < src_byte_size; byte++) {
      boost::hash_combine(*seed, tmp[byte]);
    }
  }

  return Status::Success;
}


Status
TritonCache::HashInputs(const InferenceRequest& request, size_t* seed)
{
  const auto& inputs = request.ImmutableInputs();
  // Convert inputs to ordered map for consistency in hashing
  // inputs sorted by key (input) name
  std::map<std::string, InferenceRequest::Input*> ordered_inputs(
      inputs.begin(), inputs.end());
  for (const auto& input : ordered_inputs) {
    // Add input name to hash
    boost::hash_combine(*seed, input.second->Name());
    // Fetch input buffer for hashing raw data
    RETURN_IF_ERROR(HashInputBuffers(input.second, seed));
  }

  return Status::Success;
}


Status
TritonCache::Hash(const InferenceRequest& request, std::string* key)
{
  std::size_t seed = 0;
  // Add request model name to hash
  boost::hash_combine(seed, request.ModelName());
  // Add request model version to hash
  boost::hash_combine(seed, request.ActualModelVersion());
  RETURN_IF_ERROR(HashInputs(request, &seed));
  // NOTE: Could prepend model name/version in key for readability/debugging
  *key = std::to_string(seed);
  return Status::Success;
}

Status
TritonCache::Insert(
    const std::vector<std::shared_ptr<CacheEntryItem>>& items,
    const std::string& key)
{
  LOG_VERBOSE(2) << "Inserting items at cache key: " << key;
  if (insert_fn_ == nullptr) {
    return Status(Status::Code::NOT_FOUND, "cache insert function is nullptr");
  }
  // NOTE: Possible optimization - Check if key exists first before forming
  // Cache Entry

  // NOTE: Similar to Lookup, we are currently creating CacheEntry on Triton
  // side, and letting cache retrieve the Items/Buffers via C APIs. The cache
  // implementation will have to copy the buffers since Triton may invalidate
  // them shortly after the insert_fn call.
  const auto entry = std::make_unique<CacheEntry>();
  for (const auto& item : items) {
    entry->AddItem(item);
  }
  const auto opaque_entry =
      reinterpret_cast<TRITONCACHE_CacheEntry*>(entry.get());
  RETURN_IF_TRITONSERVER_ERROR(
      insert_fn_(cache_impl_, key.c_str(), opaque_entry));
  return Status::Success;
}

Status
TritonCache::Insert(
    boost::span<InferenceResponse*> responses, const std::string& key)
{
  if (insert_fn_ == nullptr) {
    return Status(Status::Code::NOT_FOUND, "cache insert function is nullptr");
  }

  std::vector<std::shared_ptr<CacheEntryItem>> items;
  for (const auto& response : responses) {
    if (!response) {
      return Status(Status::Code::INVALID_ARG, "response is nullptr");
    }
    auto item = std::make_shared<CacheEntryItem>();
    RETURN_IF_ERROR(item->FromResponse(response));
    items.push_back(item);
  }
  return Insert(items, key);
}

Status
TritonCache::Insert(InferenceResponse* response, const std::string& key)
{
  if (!response) {
    return Status(Status::Code::INVALID_ARG, "response is nullptr");
  }

  return Insert({&response, 1}, key);
}


std::pair<Status, std::vector<std::shared_ptr<CacheEntryItem>>>
TritonCache::Lookup(const std::string& key)
{
  LOG_VERBOSE(2) << "Looking up bytes at cache key: " << key;
  if (lookup_fn_ == nullptr) {
    auto fail = Status(Status::Code::INTERNAL, "lookup function is nullptr");
    return {fail, {}};
  }

  auto entry = std::make_unique<CacheEntry>();
  auto opaque_entry = reinterpret_cast<TRITONCACHE_CacheEntry*>(entry.get());
  auto err = lookup_fn_(cache_impl_, key.c_str(), opaque_entry);
  if (err) {
    auto fail = Status(
        TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
    return {fail, {}};
  }

  // NOTE: Copies entry's vector of item pointers, entry pointer will be
  // cleaned up.
  // Item pointers are currently created on cache impl side by
  // TRITONCACHE_CacheEntryItemNew and we need to make sure the cache doesn't
  // call TRITONCACHE_CacheEntryItemDelete before Triton is done with them.
  // Currently, we are letting Triton clean up the CacheEntryItems when they go
  // out of scope, so CacheEntryItemDelete API is not needed in cache impl.
  return {Status::Success, entry->Items()};
}

// NOTE: Multiple responses won't be expected until supporting decoupled
// or sequence models.
Status
TritonCache::Lookup(
    boost::span<InferenceResponse*> responses, const std::string& key)
{
  LOG_VERBOSE(2) << "Looking up responses at cache key: " << key;

  const auto& [status, items] = Lookup(key);
  if (!status.IsOk()) {
    return status;
  }

  if (items.size() != responses.size()) {
    return Status(
        Status::Code::INTERNAL,
        "Expected number of responses in cache does not match. Expected: " +
            std::to_string(responses.size()) +
            ", received: " + std::to_string(items.size()));
  }

  for (size_t i = 0; i < items.size(); i++) {
    items[i]->ToResponse(responses[i]);
  }

  return Status::Success;
}

Status
TritonCache::Lookup(InferenceResponse* response, const std::string& key)
{
  if (!response) {
    return Status(Status::Code::INVALID_ARG, "response is nullptr");
  }
  return Lookup({&response, 1}, key);
}

//
// TritonCacheManager
//

static std::weak_ptr<TritonCacheManager> cache_manager_;
static std::mutex mu_;

Status
TritonCacheManager::Create(
    std::shared_ptr<TritonCacheManager>* manager, std::string cache_dir)
{
  std::lock_guard<std::mutex> lock(mu_);

  // If there is already a manager then we just use it
  *manager = cache_manager_.lock();
  if (*manager != nullptr) {
    return Status::Success;
  }

  if (cache_dir.empty()) {
    return Status(
        Status::Code::INVALID_ARG, "cache directory can not be empty");
  }

  LOG_VERBOSE(1) << "Create CacheManager with cache_dir: '" << cache_dir << "'";
  manager->reset(new TritonCacheManager(cache_dir));
  cache_manager_ = *manager;

  return Status::Success;
}

Status
TritonCacheManager::CreateCache(
    const std::string& name, const std::string& cache_config,
    std::shared_ptr<TritonCache>* cache)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (cache_ != nullptr) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "TritonCacheManager already holds a cache");
  }

  // Get the path to the cache shared library. Search path is global
  // cache directory.
  const std::vector<std::string> search_paths = {cache_dir_};
  // Triton will only use a single cache library path for now,
  // regardless of implementation.
  std::string cache_libname = TritonCacheLibraryName();
  std::string libpath = "";
  for (const auto& path : search_paths) {
    const auto full_path = JoinPath({path, cache_libname});
    bool exists = false;
    RETURN_IF_ERROR(FileExists(full_path, &exists));
    if (exists) {
      libpath = full_path;
      break;
    }
  }

  if (libpath.empty()) {
    return Status(
        Status::Code::INVALID_ARG, "unable to find '" + cache_libname +
                                       "' for cache. Searched: " + cache_dir_);
  }

  RETURN_IF_ERROR(TritonCache::Create(name, libpath, cache_config, &cache_));
  *cache = cache_;
  return Status::Success;
}

}}  // namespace triton::core
