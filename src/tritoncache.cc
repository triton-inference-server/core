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

  // Triton CacheEntry will take ownership of item, caller should not
  // invalidate it.
  std::shared_ptr<CacheEntryItem> sitem(litem);
  lentry->AddItem(sitem);
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
    TRITONCACHE_CacheEntryItem* item, const void* base,
    TRITONSERVER_BufferAttributes* buffer_attributes)
{
  if (!item || !base || !buffer_attributes) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "item, base, or buffer_attributes was nullptr");
  }

  // Get buffer attributes set by caller
  size_t byte_size = 0;
  TRITONSERVER_BufferAttributesByteSize(buffer_attributes, &byte_size);
  if (!byte_size) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "Buffer byte size was zero");
  }

  TRITONSERVER_MemoryType memory_type;
  TRITONSERVER_BufferAttributesMemoryType(buffer_attributes, &memory_type);
  // DLIS-2673: Add better memory_type support
  if (memory_type != TRITONSERVER_MEMORY_CPU &&
      memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Only buffers in CPU memory are allowed in cache currently");
  }
  const auto litem = reinterpret_cast<CacheEntryItem*>(item);
  // COPY: This will add a copy of the buffer to the item.
  litem->AddBuffer(base, byte_size);
  return nullptr;  // success
}

// Gets buffer at index from item
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemGetBuffer(
    TRITONCACHE_CacheEntryItem* item, size_t index, void** base,
    TRITONSERVER_BufferAttributes* buffer_attributes)
{
  if (!item || !base || !buffer_attributes) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "item, base, or buffer_attributes was nullptr");
  }

  const auto litem = reinterpret_cast<CacheEntryItem*>(item);
  const auto lbuffers = litem->Buffers();
  if (index >= lbuffers.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  const auto [buffer, buffer_size] = lbuffers[index];
  // No copy
  *base = buffer;
  // Set buffer attributes
  TRITONSERVER_BufferAttributesSetByteSize(buffer_attributes, buffer_size);
  // DLIS-2673: Add better memory_type support, default to CPU memory for now
  TRITONSERVER_BufferAttributesSetMemoryType(
      buffer_attributes, TRITONSERVER_MEMORY_CPU);
  TRITONSERVER_BufferAttributesSetMemoryTypeId(buffer_attributes, 0);
  return nullptr;  // success
}

}  // extern C

}}  // namespace triton::core
