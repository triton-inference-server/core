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
/// TODO: ownership/lifetime
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

/// Create a new cache entry item object. The caller takes ownership of the
/// TRITONCACHE_CacheEntryItem object and must call
/// TRITONCACHE_CacheEntryItemDelete to release the object.
///
/// \param item Returns the new cache entry item object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemNew(
    TRITONCACHE_CacheEntryItem** item);

/// Delete a cache entry item object.
///
/// \param item The cache entry item object to delete.
/// \return a TRITONSERVER_Error indicating success or failure.

// TODO: Not currently using this API. See TritonCache Lookup/Insert
// for explanation. Should clean up / clarify the expected ownerships here.
// However, should keep this API in case someone creates an item without
// adding it to a entry / transferring ownership, to allow manual cleanup.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemDelete(
    TRITONCACHE_CacheEntryItem* item);

///
/// CacheEntryItem Field Management
///

// TODO: Proper API descriptions, lifetimes, copies, etc.

// Returns number of buffers held by item
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemBufferCount(
    TRITONCACHE_CacheEntryItem* item, size_t* count);

// Adds buffer of specified memory_type/id to item. Only CPU memory supported
// currently.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemAddBuffer(
    TRITONCACHE_CacheEntryItem* item, const void* base,
    TRITONSERVER_BufferAttributes* buffer_attributes);

// Gets buffer at index from item where 0 <= index < count and
// 'count' is the value returned by TRITONCACHE_CacheEntryItemBufferCount
// and returns the memory_type info of the buffer
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItemGetBuffer(
    TRITONCACHE_CacheEntryItem* item, size_t index, void** base,
    TRITONSERVER_BufferAttributes* buffer_attributes);

///
/// The following functions can be implemented by a cache. Functions
/// indicated as required must be implemented or the cache will fail
/// to load.
///

/// Create a new cache object.
///
/// This function is required to be implemented by the cache.
///
/// The caller takes ownership of the
/// TRITONCACHE_Cache object and must call
/// TRITONCACHE_CacheDelete to release the object.
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

/// Delete a cache object.
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

/// Inserts entry into cache at specified key, unless key already exists.
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
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry);

#ifdef __cplusplus
}  // extern C
#endif
