// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "backend_memory_manager.h"

#include "pinned_memory_manager.h"
#include "status.h"
#include "tritonserver_apis.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>

#include "cuda_memory_manager.h"
#endif  // TRITON_ENABLE_GPU

// For unknown reason, windows will not export the TRITONBACKEND_*
// functions declared with dllexport in tritonbackend.h. To get those
// functions exported it is (also?) necessary to mark the definitions
// in this file with dllexport as well.
#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

namespace triton { namespace core {

extern "C" {

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_MemoryManagerAllocate(
    TRITONBACKEND_MemoryManager* manager, void** buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id,
    const uint64_t byte_size)
{
  switch (memory_type) {
    case TRITONSERVER_MEMORY_GPU:
#ifdef TRITON_ENABLE_GPU
    {
      auto status = CudaMemoryManager::Alloc(buffer, byte_size, memory_type_id);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            StatusCodeToTritonCode(status.ErrorCode()),
            status.Message().c_str());
      }
      break;
    }
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "GPU memory allocation not supported");
#endif  // TRITON_ENABLE_GPU

    case TRITONSERVER_MEMORY_CPU_PINNED:
#ifdef TRITON_ENABLE_GPU
    {
      TRITONSERVER_MemoryType mt = memory_type;
      auto status = PinnedMemoryManager::Alloc(buffer, byte_size, &mt, false);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            StatusCodeToTritonCode(status.ErrorCode()),
            status.Message().c_str());
      }
      break;
    }
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Pinned memory allocation not supported");
#endif  // TRITON_ENABLE_GPU

    case TRITONSERVER_MEMORY_CPU: {
      *buffer = malloc(byte_size);
      if (*buffer == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNAVAILABLE, "CPU memory allocation failed");
      }
      break;
    }
  }

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_MemoryManagerFree(
    TRITONBACKEND_MemoryManager* manager, void* buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  switch (memory_type) {
    case TRITONSERVER_MEMORY_GPU: {
#ifdef TRITON_ENABLE_GPU
      auto status = CudaMemoryManager::Free(buffer, memory_type_id);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            StatusCodeToTritonCode(status.StatusCode()),
            status.Message().c_str());
      }
#endif  // TRITON_ENABLE_GPU
      break;
    }

    case TRITONSERVER_MEMORY_CPU_PINNED: {
#ifdef TRITON_ENABLE_GPU
      auto status = PinnedMemoryManager::Free(buffer);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            StatusCodeToTritonCode(status.StatusCode()),
            status.Message().c_str());
      }
#endif  // TRITON_ENABLE_GPU
      break;
    }

    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
  }

  return nullptr;  // success
}

}  // extern C

}}  // namespace triton::core
