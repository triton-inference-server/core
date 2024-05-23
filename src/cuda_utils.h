// Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <set>

#include "status.h"
#include "triton/common/sync_queue.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace core {

#ifdef TRITON_ENABLE_GPU
#define RETURN_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                       \
    cudaError_t err__ = (X);                                                 \
    if (err__ != cudaSuccess) {                                              \
      return Status(                                                         \
          Status::Code::INTERNAL, (MSG) + ": " + cudaGetErrorString(err__)); \
    }                                                                        \
  } while (false)

#define RETURN_IF_CUDA_DRIVER_ERR(X, MSG)                                   \
  do {                                                                      \
    CUresult cuda_err__ = (X);                                              \
    if (cuda_err__ != CUDA_SUCCESS) {                                       \
      const char* error_string__;                                           \
      CudaDriverHelper::GetInstance().CuGetErrorString(                     \
          &error_string__, cuda_err__);                                     \
      return Status(Status::Code::INTERNAL, (MSG) + ": " + error_string__); \
    }                                                                       \
  } while (false)
#endif  // TRITON_ENABLE_GPU

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // !TRITON_ENABLE_GPU

/// Get the memory info for the specified device.
/// \param device_id The device ID.
/// \param free Return free memory in bytes.
/// \param total Return total memory in bytes.
/// \return The error status. A non-OK status means failure to get memory info.
Status GetDeviceMemoryInfo(const int device_id, size_t* free, size_t* total);

/// Enable peer access for all GPU device pairs
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status means not all pairs are enabled
Status EnablePeerAccess(const double min_compute_capability);

/// Copy buffer from 'src' to 'dst' for given 'byte_size'. The buffer location
/// is identified by the memory type and id, and the corresponding copy will be
/// initiated.
/// \param msg The message to be prepended in error message.
/// \param src_memory_type The memory type CPU/GPU of the source.
/// \param src_memory_type_id The device id of the source.
/// \param dst_memory_type The memory type CPU/GPU of the destination.
/// \param dst_memory_type_id The device id of the destination.
/// \param byte_size The size in bytes to me copied from source to destination.
/// \param src The buffer start address of the source.
/// \param dst The buffer start address of the destination.
/// \param cuda_stream The stream to be associated with, and 0 can be
/// passed for default stream.
/// \param cuda_used returns whether a CUDA memory copy is initiated. If true,
/// the caller should synchronize on the given 'cuda_stream' to ensure data copy
/// is completed.
/// \param copy_on_stream whether the memory copies should be performed in cuda
/// host functions on the 'cuda_stream'.
/// \return The error status. A non-ok status indicates failure to copy the
/// buffer.
Status CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used,
    bool copy_on_stream = false);

#ifdef TRITON_ENABLE_GPU
/// Validates the compute capability of the GPU indexed
/// \param gpu_id The index of the target GPU.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status means the target GPU is
/// not supported.
Status CheckGPUCompatibility(
    const int gpu_id, const double min_compute_capability);

/// Obtains a set of gpu ids that is supported by triton.
/// \param supported_gpus Returns the set of integers which is
///  populated by ids of supported GPUS
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-ok status means there were
/// errors encountered while querying GPU devices.
Status GetSupportedGPUs(
    std::set<int>* supported_gpus, const double min_compute_capability);

/// Checks if the GPU specified is an integrated GPU and supports Zero-copy.
/// \param gpu_id The index of the target GPU.
/// \param zero_copy_support If true, Zero-copy is supported by this GPU.
/// \return The error status. A non-OK status means the target GPU is
/// not supported.
Status SupportsIntegratedZeroCopy(const int gpu_id, bool* zero_copy_support);


/// Set the CUDA context to the specified device ID
/// It will rollback to the previous device upon destruction.
class ScopedSetDevice {
 public:
  ScopedSetDevice(int device)
  {
    overriden_ = false;

    prev_device_ = device;
    cudaGetDevice(&prev_device_);

    if (prev_device_ != device) {
      overriden_ = true;
      cudaSetDevice(device);
    }
  }

  ~ScopedSetDevice()
  {
    if (overriden_) {
      cudaSetDevice(prev_device_);
    }
  }

 private:
  int prev_device_;
  bool overriden_;
};
#endif

// Helper around CopyBuffer that updates the completion queue with the returned
// status and cuda_used flag.
void CopyBufferHandler(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, void* response_ptr,
    triton::common::SyncQueue<std::tuple<Status, bool, void*>>*
        completion_queue);

struct CopyParams {
  CopyParams(void* dst, const void* src, const size_t byte_size)
      : dst_(dst), src_(src), byte_size_(byte_size)
  {
  }

  void* dst_;
  const void* src_;
  const size_t byte_size_;
};


#ifdef TRITON_ENABLE_GPU
/// A singleton for Cuda Driver APIs. In order to continue supporting Triton
/// deployments when GPUs are not available we need to use CUDA driver APIs
/// through dlopen
class CudaDriverHelper {
 public:
  static CudaDriverHelper& GetInstance()
  {
    static CudaDriverHelper instance;
    return instance;
  }

 private:
  void* dl_open_handle_ = nullptr;
  std::string error_str_;
  CUresult (*cu_mem_create_fn_)(
      CUmemGenericAllocationHandle*, size_t, CUmemAllocationProp*,
      unsigned long long) = nullptr;
  CUresult (*cu_mem_map_fn_)(
      CUdeviceptr ptr, size_t size, size_t offset,
      CUmemGenericAllocationHandle handle, unsigned long long flags) = nullptr;
  CUresult (*cu_mem_set_access_fn_)(
      CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
  CUresult (*cu_get_error_string_fn_)(CUresult, const char**) = nullptr;
  CUresult (*cu_mem_get_allocation_granularity_fn_)(
      size_t*, const CUmemAllocationProp*,
      CUmemAllocationGranularity_flags) = nullptr;
  CUresult (*cu_mem_release_fn_)(CUmemGenericAllocationHandle) = nullptr;
  CUresult (*cu_init_fn_)(unsigned int) = nullptr;
  CUresult (*cu_mem_address_reserve_fn_)(
      CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
      unsigned long long flags) = nullptr;
  CUresult (*cu_mem_unmap_fn_)(CUdeviceptr ptr, size_t size) = nullptr;
  CUresult (*cu_mem_address_free_fn_)(CUdeviceptr ptr, size_t size) = nullptr;
  CudaDriverHelper();

  ~CudaDriverHelper();

 public:
  CudaDriverHelper(CudaDriverHelper const&) = delete;
  void operator=(CudaDriverHelper const&) = delete;
  bool IsAvailable();
  const std::string& GetErrorString() const { return error_str_; }
  void ClearErrorString() { return error_str_.clear(); }
  Status CuMemGetAllocationGranularity(
      size_t* aligned_size, const CUmemAllocationProp* prop,
      CUmemAllocationGranularity_flags flags);
  Status CuMemCreate(
      CUmemGenericAllocationHandle* block, size_t byte_size,
      CUmemAllocationProp* prop, unsigned long long flags);
  Status CuMemSetAccess(
      CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
  Status CuMemMap(
      CUdeviceptr ptr, size_t size, size_t offset,
      CUmemGenericAllocationHandle handle, unsigned long long flags);
  Status CuMemRelease(CUmemGenericAllocationHandle handle);
  Status CuMemAddressFree(CUdeviceptr ptr, size_t size);
  Status CuMemUnmap(CUdeviceptr ptr, size_t size);
  Status CuMemAddressReserve(
      CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
      unsigned long long flags);
  void CuGetErrorString(const char** error_string, CUresult error);
};

/// Get the minimum allocation granularity.
/// \param aligned_size Returns minimum allocation granularity.
/// \return The error status. A non-OK status means there were some errors
/// when querying the allocation granularity.
Status GetAllocationGranularity(size_t& aligned_sz);
#endif


}}  // namespace triton::core
