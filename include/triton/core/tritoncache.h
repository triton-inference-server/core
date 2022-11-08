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

/* Cache Lifetime Management */

/// Create a new cache object. The caller takes ownership of the
/// TRITONCACHE_Cache object and must call
/// TRITONCACHE_CacheDelete to release the object.
///
/// \param cache Returns the new cache object.
/// \param config The config options to initialize the cache with.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheNew(
    TRITONCACHE_Cache** cache, TRITONSERVER_Message* config);

/// Delete a cache object.
///
/// \param cache The cache object to delete.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheDelete(
    TRITONCACHE_Cache* cache);

/* Cache Usage */

// TODO: Add API descriptions

TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheInsert(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry);

TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheLookup(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry);

TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEvict(
    TRITONCACHE_Cache* cache);

/* CacheEntry Lifetime Management */

/* Cache Entry implementation example
**
** TODO: Move to tritoncache.cc or similar
**
** struct CacheEntry {
**   void*  items;         // blobs of data to insert or retrieve with cache
**   size_t* byte_sizes;   // size of each item in items
**   size_t  num_items;    // number of items and byte_sizes
**   char**  tags;         // (optional) tags to associate entry, groups, etc.
**   size_t  num_tags;     // (optional) number of tags provided
** }
*/

// TODO: Add API descriptions
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryNew(
    TRITONCACHE_CacheEntry** entry);

TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryDelete(
    TRITONCACHE_CacheEntry* entry);

/* CacheEntry Field Management */

// TODO: Add API descriptions

// Sets items in entry
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntrySetItems(
    TRITONCACHE_CacheEntry* entry, void* items, size_t* byte_sizes,
    size_t num_items);

// Gets items from entry
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryItems(
    TRITONCACHE_CacheEntry* entry, void** items, size_t** byte_sizes,
    size_t* num_items);

// Sets tags in entry
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntrySetTags(
    TRITONCACHE_CacheEntry* entry, char** tags, size_t num_tags);

// Gets tags from entry
TRITONCACHE_DECLSPEC TRITONSERVER_Error* TRITONCACHE_CacheEntryTags(
    TRITONCACHE_CacheEntry* entry, char*** tags, size_t* num_tags);

#ifdef __cplusplus
}
#endif
