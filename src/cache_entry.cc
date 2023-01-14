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

#include "cache_entry.h"
#include <iostream>

namespace triton { namespace core {

/* CacheEntry */

size_t
CacheEntry::ItemCount()
{
  std::unique_lock lk(item_mu_);
  return items_.size();
}

std::vector<std::shared_ptr<CacheEntryItem>>
CacheEntry::Items()
{
  std::unique_lock lk(item_mu_);
  return items_;
}

void
CacheEntry::AddItem(std::shared_ptr<CacheEntryItem> item)
{
  std::unique_lock lk(item_mu_);
  // TODO: Look at this flow, ownership, etc. - Currently, cache needs to not
  // delete/invalidate the item.
  items_.push_back(item);
}

/* CacheEntryItem */

size_t
CacheEntryItem::BufferCount()
{
  std::unique_lock lk(buffer_mu_);
  return buffers_.size();
}

std::vector<Buffer>
CacheEntryItem::Buffers()
{
  std::unique_lock lk(buffer_mu_);
  return buffers_;
}


void
CacheEntryItem::AddBufferCopy(const void* base, size_t byte_size)
{
  std::unique_lock lk(buffer_mu_);
  // COPY: Make a copy of buffer for Triton to own
  void* new_base = malloc(byte_size);
  memcpy(new_base, base, byte_size);
  buffers_.emplace_back(std::make_pair(new_base, byte_size));
}

void
CacheEntryItem::AddBufferCopy(boost::span<const std::byte> byte_span)
{
  const void* base = static_cast<const void*>(byte_span.data());
  AddBufferCopy(base, byte_span.size());
}

void
CacheEntryItem::AddBuffer(void* base, size_t byte_size, bool copy)
{
  std::unique_lock lk(buffer_mu_);
  if (copy) {
    AddBufferCopy(base, byte_size);
  } else {
    buffers_.emplace_back(std::make_pair(base, byte_size));
  }
}

void
CacheEntryItem::AddBuffer(Buffer buffer, bool copy)
{
  AddBuffer(buffer.first, buffer.second, copy);
}

CacheEntryItem::~CacheEntryItem()
{
  for (auto& [buffer, byte_size] : buffers_) {
    if (buffer) {
      free(buffer);
    }
  }
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
    auto buffer = Buffer(nullptr, 0);
    RETURN_IF_ERROR(ToBytes(output, &buffer));
    // ToBytes allocated new memory, so we pass it through: no copy here
    AddBuffer(buffer, false /* copy */);
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
    if (!base) {
      return Status(Status::Code::INTERNAL, "buffer was nullptr");
    }
    auto cache_output = CacheOutput();
    RETURN_IF_ERROR(
        FromBytes({static_cast<std::byte*>(base), byte_size}, &cache_output));

    InferenceResponse::Output* response_output = nullptr;
    RETURN_IF_ERROR(response->AddOutput(
        cache_output.name_, cache_output.dtype_, cache_output.shape_,
        &response_output));

    if (!response_output) {
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

Status
CacheEntryItem::ToBytes(const InferenceResponse::Output& output, Buffer* buffer)
{
  if (!buffer) {
    return Status(Status::Code::INVALID_ARG, "buffer arg was nullptr");
  }

  // Fetch output buffer details
  const void* output_base = nullptr;
  size_t output_byte_size = 0;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;
  void* userp = nullptr;
  auto status = output.DataBuffer(
      &output_base, &output_byte_size, &memory_type, &memory_type_id, &userp);
  if (!status.IsOk()) {
    return status;
  }

  // DLIS-2673: Add better memory_type support
  if (memory_type != TRITONSERVER_MEMORY_CPU &&
      memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    return Status(
        Status::Code::INVALID_ARG,
        "Only input buffers in CPU memory are allowed in cache currently");
  }

  // Exit early if response buffer from output is invalid
  if (!output_base) {
    return Status(
        Status::Code::INTERNAL, "Response buffer from output was nullptr");
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

  *buffer = Buffer(packed_bytes, total_byte_size);
  return Status::Success;
}

Status
CacheEntryItem::FromBytes(
    boost::span<const std::byte> packed_bytes, CacheOutput* output)
{
  if (!output) {
    return Status(Status::Code::INVALID_ARG, "output arg was nullptr");
  }

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

  // NOTE: Reference buffer section of packed bytes directly, DO NOT copy here.
  // We will copy this buffer into the response object in ToResponse, so the
  // buffer must remain valid until then. They should remain valid until the
  // CacheEntryItem is destructed.
  const void* output_buffer =
      static_cast<const void*>(packed_bytes.data() + position);
  position += output_byte_size;

  // Verify packed bytes matched expected size before allocating output_buffer
  if (packed_bytes.begin() + position != packed_bytes.end()) {
    return Status(
        Status::Code::INTERNAL, "Unexpected number of bytes received: " +
                                    std::to_string(packed_bytes.size()) +
                                    ", expected: " + std::to_string(position));
  }

  output->name_ = name;
  output->dtype_ = triton::common::ProtocolStringToDataType(dtype);
  output->shape_ = shape;
  output->byte_size_ = output_byte_size;
  output->buffer_ = const_cast<void*>(output_buffer);
  return Status::Success;
}

}}  // namespace triton::core
