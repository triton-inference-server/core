// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <iterator>
#include <vector>

#include "tritonserver_apis.h"

#pragma once

namespace triton { namespace core {
//
// A class to hold information about the buffer allocation.
//
class BufferAttributes {
 public:
  BufferAttributes(
      size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id, char cuda_ipc_handle[64]);
  BufferAttributes()
  {
    memory_type_ = TRITONSERVER_MEMORY_CPU;
    memory_type_id_ = 0;
    cuda_ipc_handle_.reserve(64);
  }

  // Set the buffer byte size
  void SetByteSize(const size_t& byte_size);

  // Set the buffer memory_type
  void SetMemoryType(const TRITONSERVER_MemoryType& memory_type);

  // Set the buffer memory type id
  void SetMemoryTypeId(const int64_t& memory_type_id);

  // Set the cuda ipc handle
  void SetCudaIpcHandle(void* cuda_ipc_handle);

  // Get the cuda ipc handle
  void* CudaIpcHandle();

  // Get the byte size
  size_t ByteSize() const;

  // Get the memory type
  TRITONSERVER_MemoryType MemoryType() const;

  // Get the memory type id
  int64_t MemoryTypeId() const;

 private:
  size_t byte_size_;
  TRITONSERVER_MemoryType memory_type_;
  int64_t memory_type_id_;
  std::vector<char> cuda_ipc_handle_;
};
}}  // namespace triton::core
