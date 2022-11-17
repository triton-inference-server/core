#include "cache_entry.h"
#include <iostream>  // debug

namespace triton { namespace core {

void
CacheEntry::AddItem(boost::span<const std::byte> byte_span)
{
  // TODO: lock?
  // Make a vector byte copy for cache to own
  std::cout << "[DEBUG] [cache_entry.cc] items_.size() before: "
            << items_.size() << std::endl;
  items_.emplace_back(byte_span.begin(), byte_span.end());
  std::cout << "[DEBUG] [cache_entry.cc] items_.size() after: " << items_.size()
            << std::endl;
}

}}  // namespace triton::core
