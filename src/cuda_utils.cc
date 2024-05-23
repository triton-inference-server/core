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

#include "cuda_utils.h"

#include "model_config_utils.h"
#include "triton/common/logging.h"
#include "triton/common/nvtx.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#endif

namespace triton { namespace core {

#ifdef TRITON_ENABLE_GPU
void CUDART_CB
MemcpyHost(void* args)
{
  auto* copy_params = reinterpret_cast<CopyParams*>(args);
  memcpy(copy_params->dst_, copy_params->src_, copy_params->byte_size_);
  delete copy_params;
}
#endif  // TRITON_ENABLE_GPU

Status
GetDeviceMemoryInfo(const int device_id, size_t* free, size_t* total)
{
  *free = 0;
  *total = 0;
#ifdef TRITON_ENABLE_GPU
  // Make sure that correct device is set before creating stream and
  // then restore the device to what was set by the caller.
  int current_device;
  auto cuerr = cudaGetDevice(&current_device);
  bool overridden = false;
  if (cuerr == cudaSuccess) {
    overridden = (current_device != device_id);
    if (overridden) {
      cuerr = cudaSetDevice(device_id);
    }
  }

  if (cuerr == cudaSuccess) {
    cuerr = cudaMemGetInfo(free, total);
  }

  if (overridden) {
    cudaSetDevice(current_device);
  }

  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("unable to get memory info for device ") +
         std::to_string(device_id) + ": " + cudaGetErrorString(cuerr)));
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

Status
EnablePeerAccess(const double min_compute_capability)
{
#ifdef TRITON_ENABLE_GPU
  // If we can't enable peer access for one device pair, the best we can
  // do is skipping it...
  std::set<int> supported_gpus;
  bool all_enabled = false;
  if (GetSupportedGPUs(&supported_gpus, min_compute_capability).IsOk()) {
    all_enabled = true;
    int can_access_peer = false;
    for (const auto& host : supported_gpus) {
      auto cuerr = cudaSetDevice(host);

      if (cuerr == cudaSuccess) {
        for (const auto& peer : supported_gpus) {
          if (host == peer) {
            continue;
          }

          cuerr = cudaDeviceCanAccessPeer(&can_access_peer, host, peer);
          if ((cuerr == cudaSuccess) && (can_access_peer == 1)) {
            cuerr = cudaDeviceEnablePeerAccess(peer, 0);
          }

          all_enabled &= ((cuerr == cudaSuccess) && (can_access_peer == 1));
        }
      }
    }
  }
  if (!all_enabled) {
    return Status(
        Status::Code::UNSUPPORTED,
        "failed to enable peer access for some device pairs");
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

Status
CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used, bool copy_on_stream)
{
  NVTX_RANGE(nvtx_, "CopyBuffer");

  *cuda_used = false;

  // For CUDA memcpy, all host to host copy will be blocked in respect to the
  // host, so use memcpy() directly. In this case, need to be careful on whether
  // the src buffer is valid.
  if ((src_memory_type != TRITONSERVER_MEMORY_GPU) &&
      (dst_memory_type != TRITONSERVER_MEMORY_GPU)) {
#ifdef TRITON_ENABLE_GPU
    if (copy_on_stream) {
      auto params = new CopyParams(dst, src, byte_size);
      cudaLaunchHostFunc(
          cuda_stream, MemcpyHost, reinterpret_cast<void*>(params));
      *cuda_used = true;
    } else {
      memcpy(dst, src, byte_size);
    }
#else
    memcpy(dst, src, byte_size);
#endif  // TRITON_ENABLE_GPU
  } else {
#ifdef TRITON_ENABLE_GPU
    RETURN_IF_CUDA_ERR(
        cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDefault, cuda_stream),
        msg + ": failed to perform CUDA copy");

    *cuda_used = true;
#else
    return Status(
        Status::Code::INTERNAL,
        msg + ": try to use CUDA copy while GPU is not supported");
#endif  // TRITON_ENABLE_GPU
  }

  return Status::Success;
}

void
CopyBufferHandler(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, void* response_ptr,
    triton::common::SyncQueue<std::tuple<Status, bool, void*>>*
        completion_queue)
{
  bool cuda_used = false;
  Status status = CopyBuffer(
      msg, src_memory_type, src_memory_type_id, dst_memory_type,
      dst_memory_type_id, byte_size, src, dst, cuda_stream, &cuda_used);
  completion_queue->Put(std::make_tuple(status, cuda_used, response_ptr));
}

#ifdef TRITON_ENABLE_GPU
Status
CheckGPUCompatibility(const int gpu_id, const double min_compute_capability)
{
  // Query the compute capability from the device
  cudaDeviceProp cuprops;
  cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + cudaGetErrorString(cuerr));
  }

  double compute_compability = cuprops.major + (cuprops.minor / 10.0);
  if ((compute_compability > min_compute_capability) ||
      (abs(compute_compability - min_compute_capability) < 0.01)) {
    return Status::Success;
  } else {
    return Status(
        Status::Code::UNSUPPORTED,
        "gpu " + std::to_string(gpu_id) + " has compute capability '" +
            std::to_string(cuprops.major) + "." +
            std::to_string(cuprops.minor) +
            "' which is less than the minimum supported of '" +
            std::to_string(min_compute_capability) + "'");
  }
}

Status
GetSupportedGPUs(
    std::set<int>* supported_gpus, const double min_compute_capability)
{
  // Make sure set is empty before starting
  supported_gpus->clear();

  int device_cnt;
  cudaError_t cuerr = cudaGetDeviceCount(&device_cnt);
  if ((cuerr == cudaErrorNoDevice) || (cuerr == cudaErrorInsufficientDriver)) {
    device_cnt = 0;
  } else if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to get number of CUDA devices: " +
                                    std::string(cudaGetErrorString(cuerr)));
  }

  // populates supported_gpus
  for (int gpu_id = 0; gpu_id < device_cnt; gpu_id++) {
    Status status = CheckGPUCompatibility(gpu_id, min_compute_capability);
    if (status.IsOk()) {
      supported_gpus->insert(gpu_id);
    }
  }
  return Status::Success;
}

Status
SupportsIntegratedZeroCopy(const int gpu_id, bool* zero_copy_support)
{
  // Query the device to check if integrated
  cudaDeviceProp cuprops;
  cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != cudaSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get CUDA device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + cudaGetErrorString(cuerr));
  }

  // Zero-copy supported only on integrated GPU when it can map host memory
  if (cuprops.integrated && cuprops.canMapHostMemory) {
    *zero_copy_support = true;
  } else {
    *zero_copy_support = false;
  }

  return Status::Success;
}

Status
GetAllocationGranularity(size_t& aligned_sz)
{
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  RETURN_IF_ERROR(CudaDriverHelper::GetInstance().CuMemGetAllocationGranularity(
      &aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return Status::Success;
}

CudaDriverHelper::CudaDriverHelper()
{
  dl_open_handle_ = nullptr;
#ifndef _WIN32
  dl_open_handle_ = dlopen("libcuda.so", RTLD_LAZY);
  if (dl_open_handle_ != nullptr) {
    void* cu_mem_create_fn = dlsym(dl_open_handle_, "cuMemCreate");
    if (cu_mem_create_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemCreate";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_create_fn_) = cu_mem_create_fn;

    void* cu_get_error_string_fn = dlsym(dl_open_handle_, "cuGetErrorString");
    if (cu_get_error_string_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuGetErrorString";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_get_error_string_fn_) = cu_get_error_string_fn;

    void* cu_init_fn = dlsym(dl_open_handle_, "cuInit");
    if (cu_init_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuInit";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_init_fn_) = cu_init_fn;

    void* cu_mem_set_access_fn = dlsym(dl_open_handle_, "cuMemSetAccess");
    if (cu_mem_set_access_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemSetAccess";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_set_access_fn_) = cu_mem_set_access_fn;

    void* cu_mem_release_fn = dlsym(dl_open_handle_, "cuMemRelease");
    if (cu_mem_release_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemRelease";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_release_fn_) = cu_mem_release_fn;

    void* cu_mem_get_allocation_granularity_fn =
        dlsym(dl_open_handle_, "cuMemGetAllocationGranularity");
    if (cu_mem_get_allocation_granularity_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemGetAllocationGranularity";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_get_allocation_granularity_fn_) =
        cu_mem_get_allocation_granularity_fn;

    void* cu_mem_address_free_fn = dlsym(dl_open_handle_, "cuMemAddressFree");
    if (cu_mem_address_free_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemAddressFree";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_address_free_fn_) = cu_mem_address_free_fn;

    void* cu_mem_unmap_fn = dlsym(dl_open_handle_, "cuMemUnmap");
    if (cu_mem_unmap_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemUnmap";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_unmap_fn_) = cu_mem_unmap_fn;

    void* cu_mem_address_reserve_fn =
        dlsym(dl_open_handle_, "cuMemAddressReserve");
    if (cu_mem_address_reserve_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemAddressReserve";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_address_reserve_fn_) = cu_mem_address_reserve_fn;

    void* cu_mem_map_fn = dlsym(dl_open_handle_, "cuMemMap");
    if (cu_mem_map_fn == nullptr) {
      LOG_WARNING << "Failed to dlsym cuMemMap";
      dl_open_handle_ = nullptr;
      return;
    }
    *((void**)&cu_mem_map_fn_) = cu_mem_map_fn;

    // Initialize the driver API.
    CUresult cuda_err = (*cu_init_fn_)(0 /* flags */);
    if (cuda_err != CUDA_SUCCESS) {
      const char* error_string;
      (*cu_get_error_string_fn_)(cuda_err, &error_string);
      error_str_ = std::string("failed to call cuInit: ") + error_string;
      dlclose(dl_open_handle_);
      dl_open_handle_ = nullptr;
    }
  }
#endif
}

bool
CudaDriverHelper::IsAvailable()
{
  return dl_open_handle_ != nullptr;
}

Status
CudaDriverHelper::CuMemGetAllocationGranularity(
    size_t* aligned_size, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags flags)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_get_allocation_granularity_fn_(aligned_size, prop, flags),
      std::string("failed to call cuMemGetAllocationGranularity"));
  return Status::Success;
}

Status
CudaDriverHelper::CuMemCreate(
    CUmemGenericAllocationHandle* handle, size_t allocation_size,
    CUmemAllocationProp* prop, unsigned long long flags)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_create_fn_(handle, allocation_size, prop, flags),
      std::string("failed to call cuMemCreate"));
  return Status::Success;
}

Status
CudaDriverHelper::CuMemSetAccess(
    CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_set_access_fn_(ptr, size, desc, count),
      std::string("failed to call cuMemSetAccess"));
  return Status::Success;
}

Status
CudaDriverHelper::CuMemMap(
    CUdeviceptr ptr, size_t size, size_t offset,
    CUmemGenericAllocationHandle handle, unsigned long long flags)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_map_fn_(ptr, size, offset, handle, flags),
      std::string("failed to call cuMemMap"));
  return Status::Success;
}

Status
CudaDriverHelper::CuMemRelease(CUmemGenericAllocationHandle handle)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_release_fn_(handle), std::string("failed to call cuMemRelease"));
  return Status::Success;
}

void
CudaDriverHelper::CuGetErrorString(const char** error_string, CUresult error)
{
  *error_string = nullptr;
  if (IsAvailable()) {
    cu_get_error_string_fn_(error, error_string);
  }
}

Status
CudaDriverHelper::CuMemAddressFree(CUdeviceptr ptr, size_t size)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_address_free_fn_(ptr, size),
      std::string("failed to call cuMemAddressFree"));
  return Status::Success;
}

Status
CudaDriverHelper::CuMemUnmap(CUdeviceptr ptr, size_t size)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_unmap_fn_(ptr, size), std::string("failed to call cuMemUnmap"));
  return Status::Success;
}

Status
CudaDriverHelper::CuMemAddressReserve(
    CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
    unsigned long long flags)
{
  if (!IsAvailable()) {
    return Status(
        Status::Code::INTERNAL, "CudaDriverHelper has not been initialized.");
  }
  RETURN_IF_CUDA_DRIVER_ERR(
      cu_mem_address_reserve_fn_(ptr, size, alignment, addr, flags),
      std::string("failed to call cuMemAddressReserve"));
  return Status::Success;
}


CudaDriverHelper::~CudaDriverHelper()
{
#ifndef _WIN32
  if (dl_open_handle_ != nullptr) {
    dlclose(dl_open_handle_);
  }
#endif
}
#endif

}}  // namespace triton::core
