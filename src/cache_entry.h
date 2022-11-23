#pragma once
#include <boost/core/span.hpp>
#include <memory>
#include <shared_mutex>
#include <vector>
#include "tritonserver_apis.h"

namespace triton { namespace core {

// A Buffer is an arbitrary data blob, whose type need not be known
// by the cache for storage and retrieval
// using Buffer = std::pair<void*, size_t>;
using Buffer = std::vector<std::byte>;

// An Item may have several Buffers associated with it
// ex: 1 response has multiple outputs
// NOTE: For now, Item's will be a single continuous buffer
// Multiple buffers can be concatenated as needed
// A CacheEntry may have several Items associated with it
// ex: 1 request has multiple responses
//   using Item = std::vector<Buffer>;
//   std::vector<Item> items_;

class CacheEntryItem {
 public:
  // TODO: immutable buffers?
  std::vector<Buffer> Buffers();
  size_t BufferCount();
  void AddBuffer(boost::span<const std::byte> buffer);

 private:
  // Shared mutex to support read-only and read-write locks
  std::shared_mutex buffer_mu_;
  // TODO: Pointer to be more lightweight depending on usage?
  std::vector<Buffer> buffers_;
};

class CacheEntry {
 public:
  const std::vector<std::shared_ptr<CacheEntryItem>>& Items();
  size_t ItemCount();
  void AddItem(std::shared_ptr<CacheEntryItem> item);

 private:
  // Shared mutex to support read-only and read-write locks
  std::shared_mutex item_mu_;
  // TODO: Pointer to be more lightweight depending on usage?
  std::vector<std::shared_ptr<CacheEntryItem>> items_;
};

}}  // namespace triton::core
