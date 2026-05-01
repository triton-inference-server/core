// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda_memory_manager.h"

#include <cnmem.h>
#include <string.h>

#include <set>

#include "cuda_utils.h"
#include "triton/common/logging.h"

namespace {

#define RETURN_IF_CNMEM_ERROR(S, MSG)                    \
  do {                                                   \
    auto status__ = (S);                                 \
    if (status__ != CNMEM_STATUS_SUCCESS) {              \
      return Status(                                     \
          Status::Code::INTERNAL,                        \
          (MSG) + ": " + cnmemGetErrorString(status__)); \
    }                                                    \
  } while (false)

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

}  // namespace

namespace triton { namespace core {

std::unique_ptr<CudaMemoryManager> CudaMemoryManager::instance_;
std::mutex CudaMemoryManager::instance_mu_;

CudaMemoryManager::~CudaMemoryManager()
{
  if (has_allocation_) {
    auto status = cnmemFinalize();
    if (status != CNMEM_STATUS_SUCCESS) {
      LOG_ERROR << "Failed to finalize CUDA memory manager: [" << status << "] "
                << cnmemGetErrorString(status);
    }
  }
}

void
CudaMemoryManager::Reset()
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  instance_.reset();
}

Status
CudaMemoryManager::Create(const CudaMemoryManager::Options& options)
{
  // Ensure thread-safe creation of CUDA memory pool
  std::lock_guard<std::mutex> lock(instance_mu_);
  if (instance_ != nullptr) {
    LOG_WARNING << "New CUDA memory pools could not be created since they "
                   "already exists";
    return Status::Success;
  }

  std::set<int> supported_gpus;
  auto status = GetSupportedGPUs(
      &supported_gpus, options.min_supported_compute_capability_);
  if (status.IsOk()) {
    std::vector<cnmemDevice_t> devices;
    for (auto gpu : supported_gpus) {
      const auto it = options.memory_pool_byte_size_.find(gpu);
      if ((it != options.memory_pool_byte_size_.end()) && (it->second != 0)) {
        devices.emplace_back();
        auto& device = devices.back();
        memset(&device, 0, sizeof(device));
        device.device = gpu;
        device.size = it->second;

        LOG_INFO << "CUDA memory pool is created on device " << device.device
                 << " with size " << device.size;
      }
    }

    if (!devices.empty()) {
      RETURN_IF_CNMEM_ERROR(
          cnmemInit(devices.size(), devices.data(), CNMEM_FLAGS_CANNOT_GROW),
          std::string("Failed to finalize CUDA memory manager"));
    } else {
      LOG_INFO << "CUDA memory pool disabled";
    }

    // Use to finalize CNMeM properly when out of scope
    instance_.reset(new CudaMemoryManager(!devices.empty()));
  } else {
    return Status(
        status.ErrorCode(),
        "Failed to initialize CUDA memory manager: " + status.Message());
  }

  return Status::Success;
}

Status
CudaMemoryManager::Alloc(void** ptr, uint64_t size, int64_t device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "CudaMemoryManager has not been created");
  } else if (!instance_->has_allocation_) {
    return Status(
        Status::Code::UNAVAILABLE,
        "CudaMemoryManager has no preallocated CUDA memory");
  }

  int current_device;
  RETURN_IF_CUDA_ERR(
      cudaGetDevice(&current_device), std::string("Failed to get device"));
  bool overridden = (current_device != device_id);
  if (overridden) {
    RETURN_IF_CUDA_ERR(
        cudaSetDevice(device_id), std::string("Failed to set device"));
  }

  // Defer returning error to make sure the device is recovered
  auto err = cnmemMalloc(ptr, size, nullptr);

  if (overridden) {
    cudaSetDevice(current_device);
  }

  RETURN_IF_CNMEM_ERROR(
      err, std::string("Failed to allocate CUDA memory with byte size ") +
               std::to_string(size) + " on GPU " + std::to_string(device_id));
  return Status::Success;
}

Status
CudaMemoryManager::Free(void* ptr, int64_t device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "CudaMemoryManager has not been created");
  } else if (!instance_->has_allocation_) {
    return Status(
        Status::Code::UNAVAILABLE,
        "CudaMemoryManager has no preallocated CUDA memory");
  }

  int current_device;
  RETURN_IF_CUDA_ERR(
      cudaGetDevice(&current_device), std::string("Failed to get device"));
  bool overridden = (current_device != device_id);
  if (overridden) {
    RETURN_IF_CUDA_ERR(
        cudaSetDevice(device_id), std::string("Failed to set device"));
  }

  // Defer returning error to make sure the device is recovered
  auto err = cnmemFree(ptr, nullptr);

  if (overridden) {
    cudaSetDevice(current_device);
  }

  RETURN_IF_CNMEM_ERROR(
      err, std::string("Failed to deallocate CUDA memory at address ") +
               PointerToString(ptr) + " on GPU " + std::to_string(device_id));
  return Status::Success;
}

}}  // namespace triton::core
