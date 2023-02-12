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

#include <stddef.h>
#include <stdint.h>
#include "triton/core/tritonserver.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _COMPILING_TRITONCACHE
#if defined(_MSC_VER)
#define TRITONCACHE_DECLSPEC __declspec(dllexport)
#define TRITONCACHE_ISPEC __declspec(dllimport)
#elif defined(__GNUC__)
#define TRITONCACHE_DECLSPEC __attribute__((__visibility__("default")))
#define TRITONCACHE_ISPEC
#else
#define TRITONCACHE_DECLSPEC
#define TRITONCACHE_ISPEC
#endif
#else
#if defined(_MSC_VER)
#define TRITONCACHE_DECLSPEC __declspec(dllimport)
#define TRITONCACHE_ISPEC __declspec(dllexport)
#else
#define TRITONCACHE_DECLSPEC
#define TRITONCACHE_ISPEC
#endif
#endif

struct TRITONCACHE_Cache;
struct TRITONCACHE_CacheEntry;
struct TRITONCACHE_CacheEntryItem;
struct TRITONCACHE_CacheAllocator;

///
/// TRITONCACHE API Version
///
/// The TRITONCACHE API is versioned with major and minor version
/// numbers. Any change to the API that does not impact backwards
/// compatibility (for example, adding a non-required function)
/// increases the minor version number. Any change that breaks
/// backwards compatibility (for example, deleting or changing the
/// behavior of a function) increases the major version number. A
/// cache implementation should check that the API version used to compile
/// the cache is compatible with the API version of the Triton server
/// that it is running in. This is typically done by code similar to
/// the following which makes sure that the major versions are equal
/// and that the minor version of Triton is >= the minor version used
/// to build the cache.
///
///   uint32_t api_version_major, api_version_minor;
///   TRITONCACHE_ApiVersion(&api_version_major, &api_version_minor);
///   if ((api_version_major != TRITONCACHE_API_VERSION_MAJOR) ||
///       (api_version_minor < TRITONCACHE_API_VERSION_MINOR)) {
///     return TRITONSERVER_ErrorNew(
///       TRITONSERVER_ERROR_UNSUPPORTED,
///       "triton cache API version does not support this cache");
///   }
///
#define TRITONCACHE_API_VERSION_MAJOR 0
#define TRITONCACHE_API_VERSION_MINOR 1

/// Get the TRITONCACHE API version supported by Triton. This
/// value can be compared against the
/// TRITONCACHE_API_VERSION_MAJOR and
/// TRITONCACHE_API_VERSION_MINOR used to build the cache to
/// ensure that Triton is compatible with the cache.
///
/// \param major Returns the TRITONCACHE API major version supported
/// by Triton.
/// \param minor Returns the TRITONCACHE API minor version supported
/// by Triton.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_ApiVersion(
    uint32_t* major, uint32_t* minor);

///
/// CacheEntry Field Management
///

/// Get the number of items available in the entry.
///
/// \param entry The CacheEntry object to query.
/// \param count Returns the number of items in entry.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemCount(
    TRITONCACHE_CacheEntry* entry, size_t* count);

/// Adds item to the entry.
///
/// \param entry The CacheEntry object to add item to.
/// \param item The CacheEntryItem being added to entry.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryAddItem(
    TRITONCACHE_CacheEntry* entry, TRITONCACHE_CacheEntryItem* item);

/// Gets the item at index from entry.
///
/// The caller does not own the returned item and must not modify or delete it.
/// The lifetime of the item extends until 'entry' or 'item' is deleted.
///
/// \param entry The CacheEntry object.
/// \param index The index of the item, must be 0 <= index < count, where
/// 'count' is the value returned by TRITONCACHE_CacheEntryItemCount.
/// \param item The item at index that is returned.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryGetItem(
    TRITONCACHE_CacheEntry* entry, size_t index,
    TRITONCACHE_CacheEntryItem** item);

///
/// CacheEntryItem Lifetime Management
///

/// Create a new cache entry item object. Typically, the caller will pass
/// ownership of the item to Triton to avoid unnecessary copies, and Triton
/// will manage releasing the object internally.
///
/// If for some reason ownership is not passed to Triton, then
/// the caller must take ownership of the item and must call
/// TRITONCACHE_CacheEntryItemDelete to release it when finished.
///
/// \param item Returns the new cache entry item object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemNew(
    TRITONCACHE_CacheEntryItem** item);

/// Delete a cache entry item object.
///
/// Typically ownership of a cache entry item will be passed to Triton and
/// it will be cleaned up internally. This API is only for manual cleanup in an
/// edge case where ownership is not passed.
///
/// \param item The cache entry item object to delete.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemDelete(
    TRITONCACHE_CacheEntryItem* item);

///
/// CacheEntryItem Field Management
///

/// Get the number of buffers held by item
///
/// \param item The CacheEntryItem object to query.
/// \param count Returns the number of buffers in item.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemBufferCount(
    TRITONCACHE_CacheEntryItem* item, size_t* count);

/// Adds buffer to item.
///
/// NOTE: (DLIS-4471) Currently this will make a copy of the buffer at 'base'
/// to avoid lifetime issues.
///
/// NOTE: (DLIS-2673) Only buffers in CPU memory supported currently.
///
/// \param item The CacheEntryItem object to add buffer to.
/// \param base The base address of the buffer to add.
/// \param buffer_attributes The buffer attributes associated with the buffer.
/// The caller must create the buffer attributes object, and set the relevant
/// fields through the BufferAttributes API.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemAddBuffer(
    TRITONCACHE_CacheEntryItem* item, const void* base,
    TRITONSERVER_BufferAttributes* buffer_attributes);

/// Gets the buffer at index from item.
///
/// The caller does not own the returned buffer and must not modify or delete
/// it. The lifetime of the buffer extends until 'item' is deleted. If the
/// buffer needs to persist long term, the caller should make a copy.
///
/// NOTE: Currently in the context of Triton, this API is used for the cache
/// implementation to access the buffers from the opaque item object passed by
/// Triton in TRITONCACHE_CacheInsert. It is expected that the cache
/// will get the buffer, and perform any necessary copy within the
/// TRITONCACHE_CacheInsert implementation. After TRITONCACHE_CacheInsert
/// returns, there is no guarantee that Triton won't delete the item holding
/// the buffer. This is also why the caller is expected to create and own the
/// BufferAttributes object, as a copy would be needed otherwise anyway.
///
/// \param item The CacheEntryItem object owning the buffer.
/// \param index The index of the buffer, must be 0 <= index < count, where
/// 'count' is the value returned by TRITONCACHE_CacheEntryItemBufferCount.
/// \param base The base address of the buffer at index that is returned.
/// \param buffer_attributes The buffer attributes associated with the buffer.
/// The caller must create the buffer attributes object, then Triton will
/// internally set the relevant fields on this object through the
/// BufferAttributes API.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemGetBuffer(
    TRITONCACHE_CacheEntryItem* item, size_t index, void** base,
    TRITONSERVER_BufferAttributes* buffer_attributes);

///
/// The following functions can be implemented by a cache. Functions
/// indicated as required must be implemented or the cache will fail
/// to load.
///

/// Intialize a new cache object.
///
/// This function is required to be implemented by the cache.
///
/// The caller takes ownership of the
/// TRITONCACHE_Cache object and must call
/// TRITONCACHE_CacheFinalize to cleanup and release the object.
///
/// This API is implemented by the user-provided cache implementation,
/// so specific details will be found within the cache implementation.
///
/// \param cache Returns the new cache object.
/// \param config The config options to initialize the cache with.
///               This will be passed as-is to the cache implementation, so
///               the expected format and parsing is up to the cache as well.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_ISPEC TRITONSERVER_Error* TRITONCACHE_CacheInitialize(
    TRITONCACHE_Cache** cache, const char* config);

/// Cleanup a cache object.
///
/// This function is required to be implemented by the cache.
///
/// This API is implemented by the user-provided cache implementation,
/// so specific details will be found within the cache implementation.
///
/// \param cache The cache object to delete.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_ISPEC TRITONSERVER_Error* TRITONCACHE_CacheFinalize(
    TRITONCACHE_Cache* cache);

/// Inserts entry into cache at specified key. Typically this will fail
/// if the key already exists, but a cache implementation may decide to allow
/// overwriting entries for existing keys.
///
/// This function is required to be implemented by the cache.
///
/// This API is implemented by the user-provided cache implementation,
/// so specific details will be found within the cache implementation.
///
/// \param cache The object that is used to communicate with the cache
///              implementation through a shared library.
/// \param key The key used to access the cache. Generally, this is some
///            unique value or hash representing the entry to avoid collisions.
/// \param entry The entry to be inserted into the cache.
/// \return a TRITONSERVER_Error indicating success or failure.
///         Specific errors will be up the cache implementation, but general
///         error best practices that should be followed are as follows:
///         - TRITONSERVER_ERROR_INVALID_ARG
///           - bad argument passed, nullptr, etc.
///         - TRITONSERVER_ERROR_ALREADY_EXISTS
///           - key already exists and will not be inserted again
///         - TRITONSERVER_ERROR_INTERNAL
///           - internal errors
///         - nullptr
///           - success
TRITONCACHE_ISPEC TRITONSERVER_Error* TRITONCACHE_CacheInsert(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry);

/// Retrieves entry from cache at specified key, if key exists.
///
/// This function is required to be implemented by the cache.
///
/// This API is implemented by the user-provided cache implementation,
/// so specific details will be found within the cache implementation.
///
/// \param cache The object that is used to communicate with the cache
///              implementation through a shared library.
/// \param key The key used to access the cache. Generally, this is some
///            unique value or hash representing the entry to avoid collisions.
/// \param entry The entry to be retrieved from the cache.
/// \param allocator Optional TritonCacheAllocator that can be used to copy
///                  cache data directly into user provided buffers. If not
///                  provided, an extra copy may be made.
/// \return a TRITONSERVER_Error indicating success or failure.
///         Specific errors will be up the cache implementation, but general
///         error best practices that should be followed are as follows:
///         - TRITONSERVER_ERROR_INVALID_ARG
///           - bad argument passed, nullptr, etc.
///         - TRITONSERVER_ERROR_NOT_FOUND
///           - key not found in cache
///         - TRITONSERVER_ERROR_INTERNAL
///           - other internal errors
///         - nullptr
///           - success
TRITONCACHE_ISPEC TRITONSERVER_Error* TRITONCACHE_CacheLookup(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry,
    TRITONCACHE_CacheAllocator* allocator);

#ifdef __cplusplus
}  // extern C
#endif
