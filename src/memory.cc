// Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "memory.h"

#include "pinned_memory_manager.h"
#include "triton/common/logging.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>

#include "cuda_memory_manager.h"
#include "cuda_utils.h"
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace core {

//
// MemoryReference
//
MemoryReference::MemoryReference() : Memory() {}

const char*
MemoryReference::BufferAt(
    size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id) const
{
  if (idx >= buffer_.size()) {
    *byte_size = 0;
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
    return nullptr;
  }
  *memory_type = buffer_[idx].buffer_attributes_.MemoryType();
  *memory_type_id = buffer_[idx].buffer_attributes_.MemoryTypeId();
  *byte_size = buffer_[idx].buffer_attributes_.ByteSize();
  return buffer_[idx].buffer_;
}

const char*
MemoryReference::BufferAt(size_t idx, BufferAttributes** buffer_attributes)
{
  if (idx >= buffer_.size()) {
    *buffer_attributes = nullptr;
    return nullptr;
  }

  *buffer_attributes = &(buffer_[idx].buffer_attributes_);
  return buffer_[idx].buffer_;
}

size_t
MemoryReference::AddBuffer(
    const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  total_byte_size_ += byte_size;
  buffer_count_++;
  buffer_.emplace_back(buffer, byte_size, memory_type, memory_type_id);
  return buffer_.size() - 1;
}

size_t
MemoryReference::AddBuffer(
    const char* buffer, BufferAttributes* buffer_attributes)
{
  total_byte_size_ += buffer_attributes->ByteSize();
  buffer_count_++;
  buffer_.emplace_back(buffer, buffer_attributes);
  return buffer_.size() - 1;
}

size_t
MemoryReference::AddBufferFront(
    const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  total_byte_size_ += byte_size;
  buffer_count_++;
  buffer_.emplace(
      buffer_.begin(), buffer, byte_size, memory_type, memory_type_id);
  return buffer_.size() - 1;
}

//
// MutableMemory
//
MutableMemory::MutableMemory(
    char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
    : Memory(), buffer_(buffer),
      buffer_attributes_(
          BufferAttributes(byte_size, memory_type, memory_type_id, nullptr))
{
  total_byte_size_ = byte_size;
  buffer_count_ = (byte_size == 0) ? 0 : 1;
}

const char*
MutableMemory::BufferAt(
    size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id) const
{
  if (idx != 0) {
    *byte_size = 0;
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
    return nullptr;
  }
  *byte_size = total_byte_size_;
  *memory_type = buffer_attributes_.MemoryType();
  *memory_type_id = buffer_attributes_.MemoryTypeId();
  return buffer_;
}

const char*
MutableMemory::BufferAt(size_t idx, BufferAttributes** buffer_attributes)
{
  if (idx != 0) {
    *buffer_attributes = nullptr;
    return nullptr;
  }

  *buffer_attributes = &buffer_attributes_;
  return buffer_;
}

char*
MutableMemory::MutableBuffer(
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  if (memory_type != nullptr) {
    *memory_type = buffer_attributes_.MemoryType();
  }
  if (memory_type_id != nullptr) {
    *memory_type_id = buffer_attributes_.MemoryTypeId();
  }

  return buffer_;
}

//
// AllocatedMemory
//
AllocatedMemory::AllocatedMemory(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
    : MutableMemory(nullptr, byte_size, memory_type, memory_type_id)
{
  if (total_byte_size_ != 0) {
    // Allocate memory with the following fallback policy:
    // CUDA memory -> pinned system memory -> non-pinned system memory
    switch (buffer_attributes_.MemoryType()) {
#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU: {
        auto status = CudaMemoryManager::Alloc(
            (void**)&buffer_, total_byte_size_,
            buffer_attributes_.MemoryTypeId());
        if (!status.IsOk()) {
          static bool warning_logged = false;
          if (!warning_logged) {
            LOG_WARNING << status.Message()
                        << ", falling back to pinned system memory";
            warning_logged = true;
          }

          goto pinned_memory_allocation;
        }
        break;
      }
      pinned_memory_allocation:
#endif  // TRITON_ENABLE_GPU
      default: {
        TRITONSERVER_MemoryType memory_type = buffer_attributes_.MemoryType();
        auto status = PinnedMemoryManager::Alloc(
            (void**)&buffer_, total_byte_size_, &memory_type, true);
        buffer_attributes_.SetMemoryType(memory_type);
        if (!status.IsOk()) {
          LOG_ERROR << status.Message();
          buffer_ = nullptr;
        }
        break;
      }
    }
  }
  total_byte_size_ = (buffer_ == nullptr) ? 0 : total_byte_size_;
}

AllocatedMemory::~AllocatedMemory()
{
  if (buffer_ != nullptr) {
    switch (buffer_attributes_.MemoryType()) {
      case TRITONSERVER_MEMORY_GPU: {
#ifdef TRITON_ENABLE_GPU
        auto status =
            CudaMemoryManager::Free(buffer_, buffer_attributes_.MemoryTypeId());
        if (!status.IsOk()) {
          LOG_ERROR << status.Message();
        }
#endif  // TRITON_ENABLE_GPU
        break;
      }

      default: {
        auto status = PinnedMemoryManager::Free(buffer_);
        if (!status.IsOk()) {
          LOG_ERROR << status.Message();
          buffer_ = nullptr;
        }
        break;
      }
    }
    buffer_ = nullptr;
  }
}

GrowableMemory::GrowableMemory(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, std::unique_ptr<Allocation>&& allocation,
    size_t virtual_address_size)
    : MutableMemory(nullptr, byte_size, memory_type, memory_type_id)
{
  allocation_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  allocation_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  allocation_prop_.location.id = memory_type_id;

  access_desc_.location = allocation_prop_.location;
  access_desc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  virtual_address_size_ = virtual_address_size;
}

Status
GrowableMemory::Create(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, std::unique_ptr<GrowableMemory>& growable_memory,
    size_t virtual_address_size)
{
  std::unique_ptr<Allocation> allocation =
      std::make_unique<Allocation>(memory_type_id);

  if (memory_type != TRITONSERVER_MEMORY_GPU) {
    return Status(
        Status::Code::INVALID_ARG,
        std::string("Only TRITONSERVER_MEMORY_GPU is supported for growable "
                    "memory. Found '") +
            TRITONSERVER_MemoryTypeString(memory_type) + "'.");
  }

  if (byte_size > virtual_address_size) {
    return Status(
        Status::Code::INVALID_ARG,
        std::string("'byte_size' requested for GrowableMemory cannot be smaller"
                    " than the virtual address size. byte_size: ") +
            std::to_string(byte_size) +
            ", virtual_address_size:" + std::to_string(virtual_address_size));
  }

  RETURN_IF_ERROR(
      CudaBlockManager::Allocate(byte_size, allocation, memory_type_id));

  void* buffer;
  RETURN_IF_CUDA_DRIVER_ERR(
      cuMemAddressReserve(
          reinterpret_cast<CUdeviceptr*>(&buffer), virtual_address_size,
          0 /* alignment */, 0 /* start_address */, 0 /* flags */),
      std::string("cuMemAddressReserve failed:"));
  growable_memory->buffer_ = reinterpret_cast<char*>(buffer);
  growable_memory = std::move(std::make_unique<GrowableMemory>(
      CudaBlockManager::BlockSize() * allocation->Blocks().size(), memory_type,
      memory_type_id, std::move(allocation), virtual_address_size));

  for (auto& block : allocation->Blocks()) {
    RETURN_IF_ERROR(growable_memory->Map(block));
  }
  return Status::Success;
}

Status
GrowableMemory::Map(CUmemGenericAllocationHandle& block)
{
  RETURN_IF_CUDA_DRIVER_ERR(
      cuMemMap(
          reinterpret_cast<CUdeviceptr>(buffer_) + virtual_address_offset_,
          CudaBlockManager::BlockSize(), 0UL, block, 0UL /* flags */),
      std::string("cuMemMap failed:"));
  RETURN_IF_CUDA_DRIVER_ERR(
      cuMemSetAccess(
          reinterpret_cast<CUdeviceptr>(buffer_) + virtual_address_offset_,
          CudaBlockManager::BlockSize(), &access_desc_,
          1ULL /* Mapping size */),
      std::string("cuMemSetAccess failed:"));
  virtual_address_offset_ += CudaBlockManager::BlockSize();
  return Status::Success;
}

Status
GrowableMemory::Resize(size_t size)
{
  if (size > virtual_address_size_) {
    return Status(
        Status::Code::INVALID_ARG,
        std::string(
            "Failed to resize the GrowableMemory. The requested size is larger"
            " than the virtual page size. requested size: ") +
            std::to_string(size) +
            ", virtual_address_size:" + std::to_string(virtual_address_size_));
  }

  if (size < buffer_attributes_.ByteSize()) {
    return Status::Success;
  } else {
    size_t new_size = size - buffer_attributes_.ByteSize();
    std::unique_ptr<Allocation> allocation =
        std::make_unique<Allocation>(buffer_attributes_.MemoryTypeId());
    RETURN_IF_ERROR(CudaBlockManager::Allocate(
        new_size, allocation, buffer_attributes_.MemoryTypeId()));
    for (auto& block : allocation->Blocks()) {
      RETURN_IF_ERROR(Map(block));
    }
    allocation_->Merge(std::move(allocation));
  }
  buffer_attributes_.SetByteSize(
      allocation_->Blocks().size() * CudaBlockManager::BlockSize());

  return Status::Success;
}

}}  // namespace triton::core
