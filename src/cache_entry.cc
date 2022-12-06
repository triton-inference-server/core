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

// For debugging
void
printBytes(boost::span<const std::byte> buffer)
{
  // Capture blank std::cout state
  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  std::cout << "[DEBUG] [cache_entry.cc] [LOOKUP] Buffer bytes: ";
  for (const auto& byte : buffer) {
    std::cout << std::hex << "0x" << std::to_integer<int>(byte) << " ";
  }
  std::cout << std::endl;

  // Reset std::cout state
  std::cout.copyfmt(oldState);
}

/* Helpers */

void
AppendBytes(Buffer& dst, boost::span<const std::byte> src)
{
  dst.insert(dst.end(), src.begin(), src.end());
}

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
  // CacheEntry will take ownership of item pointer
  // Items will be cleaned up when CacheEntry is cleaned up
  items_.push_back(std::move(item));
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
CacheEntryItem::AddBuffer(boost::span<const std::byte> byte_span)
{
  // Read-write, cannot be shared
  std::unique_lock lk(buffer_mu_);
  // Make a copy of buffer for Triton to own
  buffers_.emplace_back(byte_span.begin(), byte_span.end());
}

/* CacheResponseOutput */

Status
CacheEntryItem::FromResponse(const InferenceResponse* response)
{
  // TODO: pass const ref?
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
  for (const auto& buffer : buffers) {
    auto opt_cache_output = FromBytes(buffer);
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
        &output_buffer, cache_output.buffer_.size(), &memory_type,
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
    // Copy cached output buffer to allocated response output buffer
    std::memcpy(
        output_buffer, cache_output.buffer_.data(),
        cache_output.buffer_.size());
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

  // TODO: Magic number at start to indicate this is actually a response
  //       packed into bytes and not some other data?
  Buffer packed_bytes;

  // Name
  std::string name = output.Name();
  uint32_t name_byte_size = name.size();
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(&name_byte_size), sizeof(uint32_t)});
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(name.data()), name_byte_size});

  // Dtype
  std::string dtype = triton::common::DataTypeToProtocolString(output.DType());
  uint32_t dtype_byte_size = dtype.size();
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(&dtype_byte_size), sizeof(uint32_t)});
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(dtype.data()), dtype_byte_size});

  // Shape
  std::vector<int64_t> shape = output.Shape();
  uint32_t shape_byte_size = shape.size() * sizeof(int64_t);
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(&shape_byte_size), sizeof(uint32_t)});
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(shape.data()), shape_byte_size});

  // Output Buffer
  // Convert size_t to uint64_t for a fixed-size guarantee
  uint64_t u64_output_byte_size = static_cast<uint64_t>(output_byte_size);
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<std::byte*>(&u64_output_byte_size), sizeof(uint64_t)});
  AppendBytes(
      packed_bytes,
      {reinterpret_cast<const std::byte*>(output_base), u64_output_byte_size});

  // TODO: Remove
  LOG_VERBOSE(2) << "[ENCODE] name_byte_size=" << name_byte_size;
  LOG_VERBOSE(2) << "[ENCODE] name=" << name;
  LOG_VERBOSE(2) << "[ENCODE] dtype_byte_size=" << dtype_byte_size;
  LOG_VERBOSE(2) << "[ENCODE] dtype=" << dtype;
  LOG_VERBOSE(2) << "[ENCODE] shape_byte_size=" << shape_byte_size;
  LOG_VERBOSE(2) << "[ENCODE] u64_output_byte_size=" << u64_output_byte_size;
  LOG_VERBOSE(2) << "[ENCODE] output=";
  printBytes(
      {reinterpret_cast<const std::byte*>(output_base), u64_output_byte_size});
  return packed_bytes;
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

  Buffer output_buffer(output_byte_size);
  memcpy(
      output_buffer.data(), packed_bytes.begin() + position, output_byte_size);
  position += output_byte_size;

  if (packed_bytes.begin() + position != packed_bytes.end()) {
    LOG_ERROR << "Unexpected number of bytes. Received " << packed_bytes.size()
              << ", expected: " << position;
    return std::nullopt;
  }

  // TODO: Remove
  LOG_VERBOSE(2) << "[DECODE] name_byte_size=" << name_byte_size;
  LOG_VERBOSE(2) << "[DECODE] name=" << name;
  LOG_VERBOSE(2) << "[DECODE] dtype_byte_size=" << dtype_byte_size;
  LOG_VERBOSE(2) << "[DECODE] dtype=" << dtype;
  LOG_VERBOSE(2) << "[DECODE] shape_byte_size=" << shape_byte_size;
  LOG_VERBOSE(2) << "[DECODE] shape=";
  for (const auto& dim : shape) {
    LOG_VERBOSE(2) << dim << " ";
  }
  LOG_VERBOSE(2) << "[DECODE] output_byte_size=" << output_byte_size;
  LOG_VERBOSE(2) << "[DECODE] output=";
  printBytes(output_buffer);

  auto output = CacheOutput();
  output.name_ = name;
  output.dtype_ = triton::common::ProtocolStringToDataType(dtype);
  output.shape_ = shape;
  output.buffer_ = output_buffer;
  return output;
}

}}  // namespace triton::core
