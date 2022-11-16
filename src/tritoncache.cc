#include "cache_entry.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

extern "C" {

/* CacheEntry Field Management */

TRITONSERVER_Error*
TRITONCACHE_CacheEntryItemCount(
    TRITONCACHE_CacheEntry* entry, size_t* count)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock?
  const auto litems = lentry->Items();
  *count = litems.size();
  return nullptr;  // success
}

// Adds item to entry
TRITONSERVER_Error*
TRITONCACHE_CacheEntryAddItem(
    TRITONCACHE_CacheEntry* entry, void* base, size_t byte_size)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock?
  lentry->AddItem({reinterpret_cast<std::byte*>(base), byte_size});
  return nullptr;  // success
}

// Gets item at index from entry
TRITONSERVER_Error*
TRITONCACHE_CacheEntryItem(
    TRITONCACHE_CacheEntry* entry, size_t index, void** base, size_t* byte_size)
{
  if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "entry was nullptr");
  }

  const auto lentry = reinterpret_cast<CacheEntry*>(entry);
  // TODO: lock?
  const auto litems = lentry->Items();
  if (index >= litems.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "index was greater than count");
  }

  // Make a copy for Triton to own
  const std::vector<std::byte>& buffer = litems[index];
  auto byte_base = reinterpret_cast<std::byte*>(*base);
  std::copy(buffer.begin(), buffer.end(), byte_base);
  *byte_size = buffer.size();
  return nullptr;
}

}  // extern C

}}  // namespace triton::core
