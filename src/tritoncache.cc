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

//
// TRITONCACHE API Version
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_ApiVersion(uint32_t* major, uint32_t* minor)
{
  *major = TRITONCACHE_API_VERSION_MAJOR;
  *minor = TRITONCACHE_API_VERSION_MINOR;
  return nullptr;  // success
}

//
// CacheEntry Lifetime Management
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryNew(TRITONCACHE_CacheEntry** entry)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

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

  delete reinterpret_cast<CacheEntry*>(entry);
  return nullptr;
}

//
// CacheEntry Field Management
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemCount(TRITONCACHE_CacheEntry* entry, size_t* count)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  *count = lentry->ItemCount();
  return nullptr;  // success
}

// Adds item to entry.
// NOTE: Triton takes ownership of item, so the cache should not delete it.
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryAddItem(
    TRITONCACHE_CacheEntry* entry, TRITONCACHE_CacheEntryItem* item)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  const auto litem = reinterpret_cast<CacheEntryItem*>(item);

  // Triton CacheEntry will explicitly take ownership of item
  std::unique_ptr<CacheEntryItem> uitem(litem);
  lentry->AddItem(std::move(uitem));
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
  const auto litems = lentry->Items();
  if (index >= litems.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  const auto& litem = litems[index];
  if (litem == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }
  // Passthrough item pointer, no copy needed here.
  *item = reinterpret_cast<TRITONCACHE_CacheEntryItem*>(litem.get());
  return nullptr;  // success
}

//
// CacheEntryItem Lifetime Management
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemNew(TRITONCACHE_CacheEntryItem** item)
{
  if (item == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "item was nullptr");
  }

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

  delete reinterpret_cast<CacheEntryItem*>(item);
  return nullptr;
}

//
// CacheEntryItem Field Management
//
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

  const auto buffer = lbuffers[index];
  // TODO: Probably shouldn't copy here. Caller should copy if needed.
  auto byte_base = reinterpret_cast<std::byte*>(malloc(buffer.size()));
  std::copy(buffer.begin(), buffer.end(), byte_base);
  *base = byte_base;
  *byte_size = buffer.size();
  return nullptr;  // success
}

}  // extern C

}}  // namespace triton::core
