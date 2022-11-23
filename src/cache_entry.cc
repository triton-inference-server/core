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
  // std::move?

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

}}  // namespace triton::core
