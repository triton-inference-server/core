#include "cache_entry.h"

namespace triton { namespace core {

void
CacheEntry::AddItem(boost::span<const std::byte> byte_span)
{
  // TODO: lock?
  // Make a vector byte copy for cache to own
  items_.emplace_back(byte_span.begin(), byte_span.end());
}

}}  // namespace triton::core
