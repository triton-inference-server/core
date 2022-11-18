#include <iostream>  // debug
#include "cache_entry.h"

// For unknown reason, windows will not export the TRITONCACHE_*
// functions declared with dllexport in tritoncache.h. To get
// those functions exported it is (also?) necessary to mark the
// definitions in this file with dllexport as well.
#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

namespace triton { namespace core {

extern "C" {

/* CacheEntry Lifetime Management */

// TODO: flesh out
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryNew(TRITONCACHE_CacheEntry** entry)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  // TODO: remove
  std::cout << "[DEBUG] [tritoncache.cc] Creating new cache entry" << std::endl;
  *entry = reinterpret_cast<TRITONCACHE_CacheEntry*>(new CacheEntry());
  return nullptr;
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryDelete(TRITONCACHE_CacheEntry* entry)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  // TODO: remove
  std::cout << "[DEBUG] [tritoncache.cc] deleting cache entry" << std::endl;
  delete reinterpret_cast<CacheEntry*>(entry);
  return nullptr;
}

/* CacheEntry Field Management */

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemCount(TRITONCACHE_CacheEntry* entry, size_t* count)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock?
  const auto litems = lentry->Items();
  *count = litems.size();
  std::cout << "[DEBUG] [tritoncache.cc] entry->Items().size(): " << *count
            << std::endl;
  return nullptr;  // success
}

// Adds item to entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryAddItem(
    TRITONCACHE_CacheEntry* entry, const void* base, size_t byte_size)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock?
  lentry->AddItem({reinterpret_cast<const std::byte*>(base), byte_size});
  return nullptr;  // success
}

// Gets item at index from entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItem(
    TRITONCACHE_CacheEntry* entry, size_t index, void** base, size_t* byte_size)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock?
  const auto litems = lentry->Items();
  if (index >= litems.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  // Make a copy for Triton to own
  const std::vector<std::byte> buffer = litems[index];
  auto byte_base = reinterpret_cast<std::byte*>(malloc(buffer.size()));
  std::copy(buffer.begin(), buffer.end(), byte_base);

  *base = byte_base;
  *byte_size = buffer.size();
  std::cout << "[DEBUG] [tritoncache.cc] CacheEntryGetItem addr: " << byte_base
            << std::endl;
  std::cout << "[DEBUG] [tritoncache.cc] CacheEntryGetItem size: "
            << buffer.size() << std::endl;
  return nullptr;
}

}  // extern C

}}  // namespace triton::core
