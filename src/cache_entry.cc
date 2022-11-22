#include "cache_entry.h"
#include <iostream>  // debug

namespace triton { namespace core {

/* CacheEntry */

size_t
CacheEntry::ItemCount()
{
  // TODO: lock?
  return items_.size();
}

// TODO: smart ptr, etc.
std::vector<std::shared_ptr<CacheEntryItem>>
CacheEntry::Items()
{
  // TODO: lock?
  return items_;
}

void
CacheEntry::AddItem(const CacheEntryItem& item)
{
  // TODO: lock?
  // std::move?
  items_.emplace_back(std::make_shared<CacheEntryItem>(item));
  std::cout << "[DEBUG] [cache_entry.cc] items_.size() after AddItem(): "
            << items_.size() << std::endl;
}

/* CacheEntryItem */

size_t
CacheEntryItem::BufferCount()
{
  // TODO: lock?
  return buffers_.size();
}

std::vector<Buffer>
CacheEntryItem::Buffers()
{
  // TODO: lock?
  return buffers_;
}

void
CacheEntryItem::AddBuffer(boost::span<const std::byte> byte_span)
{
  // TODO: lock?
  // Make a copy for cache to own
  buffers_.emplace_back(byte_span.begin(), byte_span.end());
  std::cout << "[DEBUG] [cache_entry.cc] buffers_.size() after AddBuffer(): "
            << buffers_.size() << std::endl;
}

}}  // namespace triton::core
