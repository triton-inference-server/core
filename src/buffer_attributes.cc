// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "buffer_attributes.h"

#include <cstring>

#include "constants.h"

namespace triton { namespace core {
void
BufferAttributes::SetByteSize(const size_t& byte_size)
{
  byte_size_ = byte_size;
}

void
BufferAttributes::SetMemoryType(const TRITONSERVER_MemoryType& memory_type)
{
  memory_type_ = memory_type;
}

void
BufferAttributes::SetMemoryTypeId(const int64_t& memory_type_id)
{
  memory_type_id_ = memory_type_id;
}

void
BufferAttributes::SetCudaIpcHandle(void* cuda_ipc_handle)
{
  char* lcuda_ipc_handle = reinterpret_cast<char*>(cuda_ipc_handle);
  cuda_ipc_handle_.clear();
  std::copy(
      lcuda_ipc_handle, lcuda_ipc_handle + CUDA_IPC_STRUCT_SIZE,
      std::back_inserter(cuda_ipc_handle_));
}

void*
BufferAttributes::CudaIpcHandle()
{
  if (cuda_ipc_handle_.empty()) {
    return nullptr;
  } else {
    return reinterpret_cast<void*>(cuda_ipc_handle_.data());
  }
}

size_t
BufferAttributes::ByteSize() const
{
  return byte_size_;
}

TRITONSERVER_MemoryType
BufferAttributes::MemoryType() const
{
  return memory_type_;
}

int64_t
BufferAttributes::MemoryTypeId() const
{
  return memory_type_id_;
}

BufferAttributes::BufferAttributes(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, char* cuda_ipc_handle)
    : byte_size_(byte_size), memory_type_(memory_type),
      memory_type_id_(memory_type_id)
{
  // cuda ipc handle size
  cuda_ipc_handle_.reserve(CUDA_IPC_STRUCT_SIZE);

  if (cuda_ipc_handle != nullptr) {
    std::copy(
        cuda_ipc_handle, cuda_ipc_handle + CUDA_IPC_STRUCT_SIZE,
        std::back_inserter(cuda_ipc_handle_));
  }
}
}}  // namespace triton::core
