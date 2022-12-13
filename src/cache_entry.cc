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

#include "cache_entry.h"
#include <iostream>

namespace triton { namespace core {

/* CacheEntry */

size_t
CacheEntry::ItemCount()
{
  // Read-only, can be shared
  std::shared_lock lk(item_mu_);
  return items_.size();
}

std::vector<std::shared_ptr<CacheEntryItem>>
CacheEntry::Items()
{
  // Read-only, can be shared
  std::shared_lock lk(item_mu_);
  return items_;
}

void
CacheEntry::AddItem(std::shared_ptr<CacheEntryItem> item)
{
  // Read-write, cannot be shared
  std::unique_lock lk(item_mu_);
  // TODO: Look at this flow, ownership, etc. - Currently, cache needs to not
  // delete/invalidate the item.
  items_.push_back(item);
}

/* CacheEntryItem */

size_t
CacheEntryItem::BufferCount()
{
  // Read-only, can be shared
  std::shared_lock lk(buffer_mu_);
  return buffers_.size();
}

std::vector<Buffer>
CacheEntryItem::Buffers()
{
  // Read-only, can be shared
  std::shared_lock lk(buffer_mu_);
  return buffers_;
}

void
CacheEntryItem::AddBuffer(const void* base, size_t byte_size)
{
  // Read-write, cannot be shared
  std::unique_lock lk(buffer_mu_);
  // COPY: Make a copy of buffer for Triton to own
  void* new_base = malloc(byte_size);
  memcpy(new_base, base, byte_size);
  buffers_.emplace_back(std::make_pair(new_base, byte_size));
}


void
CacheEntryItem::AddBuffer(std::pair<void*, size_t> buffer_pair)
{
  AddBuffer(buffer_pair.first, buffer_pair.second);
}

void
CacheEntryItem::AddBuffer(boost::span<const std::byte> byte_span)
{
  const void* base = static_cast<const void*>(byte_span.data());
  AddBuffer(base, byte_span.size());
}


/* CacheResponseOutput */

Status
CacheEntryItem::FromResponse(const InferenceResponse* response)
{
  if (!response) {
    return Status(Status::Code::INTERNAL, "response was nullptr");
  }

  // Build cache entry item from response outputs
  for (const auto& output : response->Outputs()) {
    auto buffer = ToBytes(output);
    if (!buffer.has_value()) {
      return Status(
          Status::Code::INTERNAL, "failed to convert output to bytes");
    }
    AddBuffer(buffer.value());
  }

  return Status::Success;
}

Status
CacheEntryItem::ToResponse(InferenceResponse* response)
{
  if (!response) {
    return Status(Status::Code::INTERNAL, "response was nullptr");
  }

  const auto buffers = Buffers();
  for (const auto& [base, byte_size] : buffers) {
    auto opt_cache_output =
        FromBytes({static_cast<std::byte*>(base), byte_size});
    if (!opt_cache_output.has_value()) {
      return Status(
          Status::Code::INTERNAL, "failed to convert bytes to response output");
    }
    const auto& cache_output = opt_cache_output.value();

    InferenceResponse::Output* response_output = nullptr;
    RETURN_IF_ERROR(response->AddOutput(
        cache_output.name_, cache_output.dtype_, cache_output.shape_,
        &response_output));

    if (response_output == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "InferenceResponse::Output pointer as nullptr");
    }

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

    // Allocate buffer for inference response
    void* output_buffer;
    RETURN_IF_ERROR(response_output->AllocateDataBuffer(
        &output_buffer, cache_output.byte_size_, &memory_type,
        &memory_type_id));

    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Status(
          Status::Code::INTERNAL,
          "Only input buffers in CPU memory are allowed in cache currently");
    }

    if (!output_buffer) {
      return Status(
          Status::Code::INTERNAL,
          "failed to allocate buffer for output '" + cache_output.name_ + "'");
    }
    // COPY: cached output buffer to allocated response output buffer
    memcpy(output_buffer, cache_output.buffer_, cache_output.byte_size_);
  }

  return Status::Success;
}

std::optional<Buffer>
CacheEntryItem::ToBytes(const InferenceResponse::Output& output)
{
  // Fetch output buffer details
  const void* output_base = nullptr;
  size_t output_byte_size = 0;
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  void* userp;
  RETURN_NULLOPT_IF_STATUS_ERROR(output.DataBuffer(
      &output_base, &output_byte_size, &memory_type, &memory_type_id, &userp));

  if (memory_type != TRITONSERVER_MEMORY_CPU &&
      memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    LOG_ERROR
        << "Only input buffers in CPU memory are allowed in cache currently";
    return std::nullopt;
  }

  // Exit early if response buffer from output is invalid
  if (!output_base) {
    LOG_ERROR << "Response buffer from output was nullptr";
    return std::nullopt;
  }

  size_t total_byte_size = 0;

  // Name
  std::string name = output.Name();
  uint32_t name_byte_size = name.size();
  total_byte_size += sizeof(uint32_t);
  total_byte_size += name_byte_size;

  // Dtype
  std::string dtype = triton::common::DataTypeToProtocolString(output.DType());
  uint32_t dtype_byte_size = dtype.size();
  total_byte_size += sizeof(uint32_t);
  total_byte_size += dtype_byte_size;

  // Shape
  std::vector<int64_t> shape = output.Shape();
  uint32_t shape_byte_size = shape.size() * sizeof(int64_t);
  total_byte_size += sizeof(uint32_t);
  total_byte_size += shape_byte_size;

  // Output Buffer: Convert size_t to uint64_t for a fixed-size guarantee
  uint64_t u64_output_byte_size = static_cast<uint64_t>(output_byte_size);
  total_byte_size += sizeof(uint64_t);
  total_byte_size += u64_output_byte_size;

  // Allocate full buffer and pack everything into it
  std::byte* packed_bytes = static_cast<std::byte*>(malloc(total_byte_size));
  uint64_t position = 0;

  // Name
  memcpy(packed_bytes + position, &name_byte_size, sizeof(uint32_t));
  position += sizeof(uint32_t);
  memcpy(packed_bytes + position, name.c_str(), name_byte_size);
  position += name_byte_size;

  // Dtype
  memcpy(packed_bytes + position, &dtype_byte_size, sizeof(uint32_t));
  position += sizeof(uint32_t);
  memcpy(packed_bytes + position, dtype.c_str(), dtype_byte_size);
  position += dtype_byte_size;

  // Shape
  memcpy(packed_bytes + position, &shape_byte_size, sizeof(uint32_t));
  position += sizeof(uint32_t);
  memcpy(packed_bytes + position, shape.data(), shape_byte_size);
  position += shape_byte_size;

  // Output Buffer
  memcpy(packed_bytes + position, &u64_output_byte_size, sizeof(uint64_t));
  position += sizeof(uint64_t);
  memcpy(packed_bytes + position, output_base, u64_output_byte_size);
  position += u64_output_byte_size;

  return std::make_pair(packed_bytes, total_byte_size);
}

std::optional<CacheOutput>
CacheEntryItem::FromBytes(boost::span<const std::byte> packed_bytes)
{
  // Name
  size_t position = 0;
  uint32_t name_byte_size = 0;
  memcpy(&name_byte_size, packed_bytes.begin() + position, sizeof(uint32_t));
  position += sizeof(uint32_t);

  std::string name(name_byte_size, 'x');
  memcpy(name.data(), packed_bytes.begin() + position, name_byte_size);
  position += name_byte_size;

  // Dtype
  uint32_t dtype_byte_size = 0;
  memcpy(&dtype_byte_size, packed_bytes.begin() + position, sizeof(uint32_t));
  position += sizeof(uint32_t);

  std::string dtype(dtype_byte_size, 'x');
  memcpy(dtype.data(), packed_bytes.begin() + position, dtype_byte_size);
  position += dtype_byte_size;

  // Shape
  uint32_t shape_byte_size = 0;
  memcpy(&shape_byte_size, packed_bytes.begin() + position, sizeof(uint32_t));
  position += sizeof(uint32_t);

  std::vector<int64_t> shape(shape_byte_size / sizeof(int64_t), 0);
  memcpy(shape.data(), packed_bytes.begin() + position, shape_byte_size);
  position += shape_byte_size;

  // Output Buffer
  uint64_t output_byte_size = 0;
  memcpy(&output_byte_size, packed_bytes.begin() + position, sizeof(uint64_t));
  position += sizeof(uint64_t);

  std::byte* output_buffer = static_cast<std::byte*>(malloc(output_byte_size));
  memcpy(output_buffer, packed_bytes.begin() + position, output_byte_size);
  position += output_byte_size;

  if (packed_bytes.begin() + position != packed_bytes.end()) {
    LOG_ERROR << "Unexpected number of bytes. Received " << packed_bytes.size()
              << ", expected: " << position;
    return std::nullopt;
  }

  auto output = CacheOutput();
  output.name_ = name;
  output.dtype_ = triton::common::ProtocolStringToDataType(dtype);
  output.shape_ = shape;
  output.buffer_ = output_buffer;
  output.byte_size_ = output_byte_size;
  return output;
}

}}  // namespace triton::core
