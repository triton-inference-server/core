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
CacheEntry::BufferCount()
{
  std::unique_lock<std::mutex> lk(buffer_mu_);
  return buffers_.size();
}

const std::vector<Buffer>&
CacheEntry::Buffers()
{
  std::unique_lock<std::mutex> lk(buffer_mu_);
  return buffers_;
}

std::vector<Buffer>&
CacheEntry::MutableBuffers()
{
  std::unique_lock<std::mutex> lk(buffer_mu_);
  return buffers_;
}

void
CacheEntry::AddBuffer(boost::span<Byte> byte_span)
{
  void* base = static_cast<void*>(byte_span.data());
  AddBuffer(base, byte_span.size());
}

void
CacheEntry::AddBuffer(void* base, size_t byte_size)
{
  std::unique_lock<std::mutex> lk(buffer_mu_);
  buffers_.emplace_back(std::make_pair(base, byte_size));
}

CacheEntry::~CacheEntry()
{
  std::unique_lock<std::mutex> lk(buffer_mu_);
  if (!free_buffers_) {
    return;
  }

  for (auto& iter : buffers_) {
    auto& base = iter.first;
    if (base) {
      free(base);
      base = nullptr;
    }
  }
}

void
CacheEntry::AddPlaceholderBuffer(size_t byte_size)
{
  AddBuffer(nullptr, byte_size);
}

// Set the size of each buffer in the CacheEntry object so the cache knows
// how much to allocate for each entry before insertion.
Status
CacheEntry::SetBufferSizes(std::vector<boost::span<Byte>> buffers)
{
  for (const auto buffer : buffers) {
    AddPlaceholderBuffer(buffer.size());
  }
  return Status::Success;
}

Status
CacheEntry::SetBufferSizes(boost::span<InferenceResponse*> responses)
{
  for (const auto response : responses) {
    RETURN_IF_ERROR(SetBufferSize(response));
  }
  return Status::Success;
}

Status
CacheEntry::SetBufferSize(InferenceResponse* response)
{
  if (!response) {
    return Status(Status::Code::INTERNAL, "response was nullptr");
  }

  // The packed_response buffer will look like:
  //   [num_outputs, sizeof(output1), output1, ..., sizeof(outputN), outputN]
  // So the entire packed_response_buffer will be the sum of all the output
  // sizes and the metadata included to parse them.
  uint64_t packed_response_byte_size = 0;
  // 1. First the packed buffer will hold the number of outputs as a uint32_t
  packed_response_byte_size += sizeof(uint32_t);
  // These sizes will be used to request allocated buffers from the cache
  // to copy directly into
  for (const auto& output : response->Outputs()) {
    uint64_t packed_output_byte_size = 0;
    RETURN_IF_ERROR(GetByteSize(output, &packed_output_byte_size));
    // 2. Then the packed buffer will hold pairs of (output_size, output_bytes)
    packed_response_byte_size += sizeof(uint64_t);
    packed_response_byte_size += packed_output_byte_size;
  }

  AddPlaceholderBuffer(packed_response_byte_size);
  return Status::Success;
}

Status
CacheEntry::SerializeResponses(boost::span<InferenceResponse*> responses)
{
  if (buffers_.size() != responses.size()) {
    return Status(
        Status::Code::INTERNAL,
        "Expected number of responses in cache does not match. Expected: " +
            std::to_string(responses.size()) +
            ", received: " + std::to_string(buffers_.size()));
  }

  for (size_t i = 0; i < responses.size(); i++) {
    RETURN_IF_ERROR(SerializeResponse(responses[i], buffers_[i]));
  }

  return Status::Success;
}

Status
CacheEntry::SerializeResponse(InferenceResponse* response, Buffer& buffer)
{
  if (!response) {
    return Status(Status::Code::INTERNAL, "response was nullptr");
  }

  // The packed_response buffer will look like:
  //   [num_outputs, sizeof(output1), output1, ..., sizeof(outputN), outputN]
  size_t position = 0;
  Byte* base = static_cast<Byte*>(buffer.first);
  // 1. First the packed buffer will hold the number of outputs as a uint32_t
  uint32_t num_outputs = response->Outputs().size();
  std::memcpy(base, &num_outputs, sizeof(uint32_t));
  position += sizeof(uint32_t);

  for (const auto& output : response->Outputs()) {
    uint64_t packed_output_byte_size = 0;
    RETURN_IF_ERROR(SerializeResponseOutput(
        output, base + position, &packed_output_byte_size));
    // 2. Then the packed buffer will hold pairs of (output_size, output_bytes)
    position += sizeof(packed_output_byte_size);
    position += packed_output_byte_size;
  }

  // Validate serialization fit expected size
  uint64_t total_buffer_size = buffer.second;
  if (position != total_buffer_size) {
    return Status(
        Status::Code::INTERNAL,
        "Serialized buffer size does not match. Expected: " +
            std::to_string(position) +
            ", received: " + std::to_string(total_buffer_size));
  }

  return Status::Success;
}

Status
CacheEntry::SerializeResponseOutput(
    const InferenceResponse::Output& output, Byte* buffer, size_t* output_size)
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

  // Pack everything into provided buffer
  uint64_t position = 0;
  // Total serialized output size
  memcpy(buffer + position, &total_byte_size, sizeof(uint64_t));
  position += sizeof(uint64_t);

  // Name
  memcpy(buffer + position, &name_byte_size, sizeof(uint32_t));
  position += sizeof(uint32_t);
  memcpy(buffer + position, name.c_str(), name_byte_size);
  position += name_byte_size;

  // Dtype
  memcpy(buffer + position, &dtype_byte_size, sizeof(uint32_t));
  position += sizeof(uint32_t);
  memcpy(buffer + position, dtype.c_str(), dtype_byte_size);
  position += dtype_byte_size;

  // Shape
  memcpy(buffer + position, &shape_byte_size, sizeof(uint32_t));
  position += sizeof(uint32_t);
  memcpy(buffer + position, shape.data(), shape_byte_size);
  position += shape_byte_size;

  // Response Output Buffer
  memcpy(buffer + position, &u64_output_byte_size, sizeof(uint64_t));
  position += sizeof(uint64_t);
  memcpy(buffer + position, output_base, u64_output_byte_size);
  position += u64_output_byte_size;

  // Used to increment global position in total response buffer
  *output_size = total_byte_size;
  return Status::Success;
}

Status
CacheEntry::DeserializeBuffers(boost::span<InferenceResponse*> responses)
{
  if (buffers_.size() != responses.size()) {
    return Status(
        Status::Code::INTERNAL,
        "Expected number of responses in cache does not match. Expected: " +
            std::to_string(responses.size()) +
            ", received: " + std::to_string(buffers_.size()));
  }

  for (size_t i = 0; i < responses.size(); i++) {
    RETURN_IF_ERROR(DeserializeBuffer(responses[i], buffers_[i]));
  }

  return Status::Success;
}

Status
CacheEntry::DeserializeBuffer(InferenceResponse* response, const Buffer& buffer)
{
  if (!response) {
    return Status(Status::Code::INTERNAL, "response was nullptr");
  }


  const Byte* base = static_cast<Byte*>(buffer.first);
  if (!base) {
    return Status(Status::Code::INTERNAL, "buffer was nullptr");
  }

  uint64_t position = 0;
  // Number of response outputs serialized in buffer
  uint32_t num_outputs = 0;
  std::memcpy(&num_outputs, base, sizeof(num_outputs));
  position += sizeof(num_outputs);


  for (size_t i = 0; i < num_outputs; i++) {
    // Get size of packed output
    uint64_t packed_output_size = 0;
    std::memcpy(
        &packed_output_size, base + position, sizeof(packed_output_size));
    position += sizeof(packed_output_size);

    // Parse packed output
    auto cache_output = CacheOutput();
    RETURN_IF_ERROR(DeserializeResponseOutput(
        {base + position, packed_output_size}, &cache_output));
    position += packed_output_size;

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
CacheEntry::GetByteSize(
    const InferenceResponse::Output& output, uint64_t* packed_output_byte_size)
{
  if (!packed_output_byte_size) {
    return Status(Status::Code::INVALID_ARG, "byte_size arg was null");
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

  *packed_output_byte_size = total_byte_size;
  return Status::Success;
}

Status
CacheEntry::DeserializeResponseOutput(
    boost::span<const Byte> packed_bytes, CacheOutput* output)
{
  if (!output) {
    return Status(Status::Code::INVALID_ARG, "output arg was nullptr");
  }

  // Name
  size_t position = 0;
  uint32_t name_byte_size = 0;
  memcpy(&name_byte_size, packed_bytes.begin() + position, sizeof(uint32_t));
  position += sizeof(uint32_t);

  auto name_start = packed_bytes.begin() + position;
  std::string name(name_start, name_start + name_byte_size);
  position += name_byte_size;

  // Dtype
  uint32_t dtype_byte_size = 0;
  memcpy(&dtype_byte_size, packed_bytes.begin() + position, sizeof(uint32_t));
  position += sizeof(uint32_t);

  auto dtype_start = packed_bytes.begin() + position;
  std::string dtype(dtype_start, dtype_start + dtype_byte_size);
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
  // We will copy this buffer into the response object in BufferToResponse, so
  // the buffer must remain valid until then. They should remain valid until the
  // CacheEntry is destructed.
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
