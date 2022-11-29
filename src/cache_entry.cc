#include "cache_entry.h"
#include <iostream>  // debug

namespace triton { namespace core {

/* CacheEntry */

size_t
CacheEntry::ItemCount()
{
  // Read-only, can be shared
  std::shared_lock lk(item_mu_);
  return items_.size();
}

const std::vector<std::shared_ptr<CacheEntryItem>>&
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
  std::cout << "[DEBUG] [cache_entry.cc] items_.size() after AddItem(): "
            << items_.size() << std::endl;
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
  std::cout << "[DEBUG] [cache_entry.cc] buffers_.size() after AddBuffer(): "
            << buffers_.size() << std::endl;
}

/* CacheResponseOutput */

Status
CacheEntryItem::FromResponse(const InferenceResponse* response)
{
  // TODO: pass const ref
  if (!response) {
    return Status(Status::Code::INTERNAL, "response was nullptr");
  }

  // Build cache entry item from response outputs
  for (const auto& output : response->Outputs()) {
    const auto buffer = ResponseOutputToBytes(output);
    if (!buffer.has_value()) {
      return Status(
          Status::Code::INTERNAL, "failed to convert output to bytes");
    }
    AddBuffer(buffer.value());
  }

  return Status::Success;
}

std::optional<Buffer>
CacheEntryItem::ResponseOutputToBytes(const InferenceResponse::Output& output)
{
  // Fetch output buffer details
  const void* base = nullptr;
  size_t byte_size = 0;
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  void* userp;
  RETURN_NULLOPT_IF_STATUS_ERROR(output.DataBuffer(
      &base, &byte_size, &memory_type, &memory_type_id, &userp));

  if (memory_type != TRITONSERVER_MEMORY_CPU &&
      memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    LOG_ERROR
        << "Only input buffers in CPU memory are allowed in cache currently";
    return std::nullopt;
  }

  // Exit early if response buffer from output is invalid
  if (!base) {
    LOG_ERROR << "Response buffer from output was nullptr";
    return std::nullopt;
  }

  // TODO: Aggregate metadata
  // const auto name_ = output.Name();
  // const auto dtype_ = output.DType();
  // const auto shape_ = output.Shape();
  // const auto buffer_size_ = static_cast<uint64_t>(byte_size);*/

  // TODO: Use span to copy buffer data
  // boost::span<std::byte> bs{reinterpret_cast<const std::byte*>(base),
  // byte_size};

  // TODO: unused variables
  std::cout << "Unused vars: " << base << byte_size << memory_type_id << userp
            << std::endl;

  Buffer serial_bytes;
  // TODO: Form output into byte buffer
  // serial_bytes.insert(serial_bytes.end(), span.begin(), span.end());
  // serial_bytes.insert(serial_bytes.end(), span.begin(), span.end());
  // serial_bytes.insert(serial_bytes.end(), span.begin(), span.end());

  return serial_bytes;
}

}}  // namespace triton::core
