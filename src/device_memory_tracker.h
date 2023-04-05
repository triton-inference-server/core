
#pragma once

#include <cupti.h>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "triton/common/logging.h"
#include "tritonserver_apis.h"

static_assert(
    sizeof(uint64_t) >= sizeof(uintptr_t),
    "The implementation is storing address pointer as uint64_t, "
    "must ensure the space for pointer is <= sizeof(uint64_t).");

namespace triton { namespace core {

// singleton
// Specialize on getting device memory allocation
class DeviceMemoryTracker {
 public:
  struct ScopedMemoryUsage {
    ~ScopedMemoryUsage()
    {
      if (tracked_) {
        UntrackThreadMemoryUsage(this);
      }
    }
    // Byte size of allocated memory tracked,
    // 'system_byte_size_' is likely to be empty as system memory allocation
    // is not controlled by CUDA driver. But keeping it for completeness.
    std::map<int64_t, size_t> system_byte_size_;
    std::map<int64_t, size_t> pinned_byte_size_;
    std::map<int64_t, size_t> cuda_byte_size_;
    bool tracked_{false};
  };
  static void InitTrace();

  // Currently can distinguish activity by correlation id which is
  // thread specific, which implies that switching threads to handle
  // activities may not be tracked?
  // The memory usage will be updated until it's untracked, usage must
  // be valid until untrack is called.
  // [WIP] Document thread-local nature..
  static void TrackThreadMemoryUsage(ScopedMemoryUsage* usage);
  static void UntrackThreadMemoryUsage(ScopedMemoryUsage* usage);

  static void TrackActivity(CUpti_Activity* record)
  {
    tracker_->TrackActivityInternal(record);
  }

 private:
  void TrackActivityInternal(CUpti_Activity* record);

  static std::unique_ptr<DeviceMemoryTracker> tracker_;
  std::mutex mtx_;
  std::unordered_map<uint32_t, uintptr_t> activity_to_memory_usage_;
};

std::unique_ptr<DeviceMemoryTracker> DeviceMemoryTracker::tracker_{nullptr};

// Boilerplate from CUPTI examples
namespace {

#define CUPTI_CALL(call)                                      \
  do {                                                        \
    CUptiResult _status = call;                               \
    if (_status != CUPTI_SUCCESS) {                           \
      const char* errstr;                                     \
      cuptiGetResultString(_status, &errstr);                 \
      LOG_ERROR << #call << " failed with error: " << errstr; \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

void
bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
  uint8_t* bfr = (uint8_t*)malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr != nullptr) {
    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
  } else {
    LOG_ERROR << "Failed to allocate buffer for CUPIT: out of memory";
  }
}

void
bufferCompleted(
    CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
    size_t validSize)
{
  CUptiResult status;
  CUpti_Activity* record = nullptr;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        DeviceMemoryTracker::TrackActivity(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG_WARNING << "Dropped " << dropped << " activity records";
    }
  }

  free(buffer);
}

}  // namespace

void
DeviceMemoryTracker::InitTrace()
{
  if (tracker_ == nullptr) {
    tracker_.reset(new DeviceMemoryTracker());
    CUPTI_CALL(
        cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  }
}

void
DeviceMemoryTracker::TrackThreadMemoryUsage(ScopedMemoryUsage* usage)
{
  if (tracker_ == nullptr) {
    InitTrace();
  }
  CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN,
      reinterpret_cast<uint64_t>(usage)));
  usage->tracked_ = true;
}

void
DeviceMemoryTracker::UntrackThreadMemoryUsage(ScopedMemoryUsage* usage)
{
  uint64_t id;
  if (tracker_ == nullptr) {
    InitTrace();
  }
  CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));
  CUPTI_CALL(cuptiActivityFlushAll(0));
  usage->tracked_ = false;
}

void
DeviceMemoryTracker::TrackActivityInternal(CUpti_Activity* record)
{
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMORY2: {
      CUpti_ActivityMemory3* memory_record = (CUpti_ActivityMemory3*)record;
      ScopedMemoryUsage* usage = nullptr;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = activity_to_memory_usage_.find(memory_record->correlationId);
        if (it != activity_to_memory_usage_.end()) {
          usage = reinterpret_cast<ScopedMemoryUsage*>(it->second);
          activity_to_memory_usage_.erase(it);
        }
      }
      // Igore memory record that is not associated with a ScopedMemoryUsage
      // object
      if (usage != nullptr) {
        const bool is_allocation =
            (memory_record->memoryOperationType ==
             CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION);
        if (is_allocation || (memory_record->memoryOperationType ==
                              CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE)) {
          switch (memory_record->memoryKind) {
            case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE: {
              if (is_allocation) {
                usage->cuda_byte_size_[memory_record->deviceId] +=
                    memory_record->bytes;
              } else {
                usage->cuda_byte_size_[memory_record->deviceId] -=
                    memory_record->bytes;
              }
            } break;
            case CUPTI_ACTIVITY_MEMORY_KIND_PINNED: {
              if (is_allocation) {
                usage->pinned_byte_size_[memory_record->deviceId] +=
                    memory_record->bytes;
              } else {
                usage->pinned_byte_size_[memory_record->deviceId] -=
                    memory_record->bytes;
              }
              break;
            }
            case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE: {
              if (is_allocation) {
                usage->system_byte_size_[memory_record->deviceId] +=
                    memory_record->bytes;
              } else {
                usage->system_byte_size_[memory_record->deviceId] -=
                    memory_record->bytes;
              }
              break;
            }
            default:
              LOG_WARNING << "Unrecognized type of memory is allocated, kind "
                          << memory_record->memoryKind;
              break;
          }
        }
      }
      break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
      CUpti_ActivityExternalCorrelation* corr =
          (CUpti_ActivityExternalCorrelation*)record;
      std::lock_guard<std::mutex> lk(mtx_);
      activity_to_memory_usage_[corr->correlationId] =
          reinterpret_cast<uintptr_t>(corr->externalId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME: {
      // DO NOTHING, runtime API will be captured and reported to properly
      // initalize records for CUPTI_ACTIVITY_KIND_MEMORY2.
      break;
    }
    default:
      LOG_ERROR << "Unexpected capture of cupti record, kind: " << record->kind;
      break;
  }
}

}}  // namespace triton::core
