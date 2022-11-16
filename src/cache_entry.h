#include <vector>
#include <boost/core/span.hpp>
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace core {

// A Buffer is an arbitrary data blob, whose type need not be known
// by the cache for storage and retrieval
//using Buffer = std::pair<void*, size_t>;
using Buffer = std::vector<std::byte>;

// An Item may have several Buffers associated with it
// ex: 1 response has multiple outputs
// NOTE: For now, Item's will be a single continuous buffer
// Multiple buffers can be concatenated as needed 
// A CacheEntry may have several Items associated with it
// ex: 1 request has multiple responses
//   using Item = std::vector<Buffer>;

class CacheEntry {
  public:
    std::vector<Buffer> Items() { return items_; }
    size_t Count() { return items_.size(); }
    void AddItem(boost::span<std::byte> buffer);
  private:
    std::vector<Buffer> items_;
};

}}  // namespace triton::core
