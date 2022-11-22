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
  *count = lentry->ItemCount();
  std::cout << "[DEBUG] [tritoncache.cc] entry->ItemCount()" << *count
            << std::endl;
  return nullptr;  // success
}

// Adds item to entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryAddItem(
    TRITONCACHE_CacheEntry* entry, TRITONCACHE_CacheEntryItem* item)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  // TODO: lock?
  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  const auto litem = reinterpret_cast<CacheEntryItem*>(item);
  const auto item_copy = new CacheEntryItem(*litem);
  // TODO: Maintain pointer instead of making another copy
  lentry->AddItem(*item_copy);
  return nullptr;  // success
}

// Gets item at index from entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryGetItem(
    TRITONCACHE_CacheEntry* entry, size_t index,
    TRITONCACHE_CacheEntryItem** item)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock? copy?
  const auto litems = lentry->Items();
  if (index >= litems.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  const auto litem = litems[index].get();
  if (litem == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }
  const auto item_copy = new CacheEntryItem(*litem);
  *item = reinterpret_cast<TRITONCACHE_CacheEntryItem*>(item_copy);
  return nullptr;  // success
}

/* CacheEntryItem Lifetime Management */

// TODO: flesh out
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemNew(TRITONCACHE_CacheEntryItem** item)
{
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

  // TODO: remove
  std::cout << "[DEBUG] [tritoncache.cc] Creating new cache entry item"
            << std::endl;
  *item = reinterpret_cast<TRITONCACHE_CacheEntryItem*>(new CacheEntryItem());
  return nullptr;
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemDelete(TRITONCACHE_CacheEntryItem* item)
{
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

  // TODO: remove
  std::cout << "[DEBUG] [tritoncache.cc] deleting cache entry item"
            << std::endl;
  delete reinterpret_cast<CacheEntryItem*>(item);
  return nullptr;
}

/* CacheEntryItem Field Management */

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemBufferCount(
    TRITONCACHE_CacheEntryItem* item, size_t* count)
{
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

  const auto litem = reinterpret_cast<CacheEntryItem*>(item);
  *count = litem->BufferCount();
  std::cout << "[DEBUG] [tritoncache.cc] item->BufferCount()" << *count
            << std::endl;
  return nullptr;  // success
}

// Adds buffer to item
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemAddBuffer(
    TRITONCACHE_CacheEntryItem* item, const void* base, size_t byte_size)
{
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

  const auto litem = reinterpret_cast<CacheEntryItem*>(item);
  litem->AddBuffer({reinterpret_cast<const std::byte*>(base), byte_size});
  return nullptr;  // success
}

// Gets buffer at index from item
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemGetBuffer(
    TRITONCACHE_CacheEntryItem* item, size_t index, void** base,
    size_t* byte_size)
{
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

  const auto litem = reinterpret_cast<CacheEntryItem*>(item);
  const auto lbuffers = litem->Buffers();
  if (index >= lbuffers.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  // Make a copy for Triton to own
  const std::vector<std::byte> buffer = lbuffers[index];
  auto byte_base = reinterpret_cast<std::byte*>(malloc(buffer.size()));
  std::copy(buffer.begin(), buffer.end(), byte_base);

  *base = byte_base;
  *byte_size = buffer.size();
  std::cout << "[DEBUG] [tritoncache.cc] CacheEntryGetItemBuffer addr: "
            << byte_base << std::endl;
  std::cout << "[DEBUG] [tritoncache.cc] CacheEntryGetItemBuffer size: "
            << buffer.size() << std::endl;
  return nullptr;  // success
}

}  // extern C

}}  // namespace triton::core
