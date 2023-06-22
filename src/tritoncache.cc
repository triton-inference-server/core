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

#include <iostream>

#include "cache_manager.h"

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
TRITONCACHE_CacheEntryBufferCount(TRITONCACHE_CacheEntry* entry, size_t* count)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  *count = lentry->BufferCount();
  return nullptr;  // success
}

// Adds buffer to entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryAddBuffer(
    TRITONCACHE_CacheEntry* entry, void* base,
    TRITONSERVER_BufferAttributes* attrs)
{
  if (!entry || !base || !attrs) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry, base, or attrs was nullptr");
  }

  // Get buffer attributes set by caller
  size_t byte_size = 0;
  TRITONSERVER_BufferAttributesByteSize(attrs, &byte_size);
  if (!byte_size) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "Buffer byte size was zero");
  }

  TRITONSERVER_MemoryType memory_type;
  TRITONSERVER_BufferAttributesMemoryType(attrs, &memory_type);
  // DLIS-2673: Add better memory_type support
  if (memory_type != TRITONSERVER_MEMORY_CPU &&
      memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Only buffers in CPU memory are allowed in cache currently");
  }
  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // This will add a short-lived reference to the corresponding cache
  // buffer of this entry. It should be copied into the target buffer either
  // directly or through a callback.
  lentry->AddBuffer({static_cast<Byte*>(base), byte_size});
  return nullptr;  // success
}

// Gets buffer at index from entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntryGetBuffer(
    TRITONCACHE_CacheEntry* entry, size_t index, void** base,
    TRITONSERVER_BufferAttributes* attrs)
{
  if (!entry || !base || !attrs) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry, base, or attrs was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  const auto& lbuffers = lentry->Buffers();
  if (index >= lbuffers.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  const auto& buffer = lbuffers[index].first;
  const auto& buffer_size = lbuffers[index].second;
  // No copy, this buffer needs to stay alive until it is copied into the cache
  *base = buffer;
  // Set buffer attributes
  TRITONSERVER_BufferAttributesSetByteSize(attrs, buffer_size);
  // DLIS-2673: Add better memory_type support, default to CPU memory for now
  TRITONSERVER_BufferAttributesSetMemoryType(attrs, TRITONSERVER_MEMORY_CPU);
  TRITONSERVER_BufferAttributesSetMemoryTypeId(attrs, 0);
  return nullptr;  // success
}

// Sets buffer at index in entry
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONCACHE_CacheEntrySetBuffer(
    TRITONCACHE_CacheEntry* entry, size_t index, void* new_base,
    TRITONSERVER_BufferAttributes* attrs)
{
  if (!entry) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  auto& lbuffers = lentry->MutableBuffers();
  if (index >= lbuffers.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  auto& base = lbuffers[index].first;
  auto& buffer_size = lbuffers[index].second;
  base = new_base;

  // Only overwrite attributes if provided, buffer may already have some and
  // not need to change if new buffer shares the same properties
  if (attrs) {
    size_t byte_size = 0;
    TRITONSERVER_BufferAttributesByteSize(attrs, &byte_size);
    // Overwrite corresponding buffer size if provided
    buffer_size = byte_size;

    TRITONSERVER_MemoryType memory_type;
    TRITONSERVER_BufferAttributesMemoryType(attrs, &memory_type);
    // DLIS-2673: Add better memory_type support
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Only buffers in CPU memory are allowed in cache currently");
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONCACHE_Copy(
    TRITONCACHE_Allocator* allocator, TRITONCACHE_CacheEntry* entry)
{
  if (!allocator || !entry) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "allocator or entry was nullptr");
  }

  const auto lallocator = reinterpret_cast<TritonCacheAllocator*>(allocator);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(lallocator->Allocate(entry));
  return nullptr;  // success
}

}  // extern C

}}  // namespace triton::core
