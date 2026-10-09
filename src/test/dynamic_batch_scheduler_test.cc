// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "triton/core/tritonserver.h"

namespace {

// Far more in-flight filler requests than the global payload pool holds
// (MAX_PAYLOAD_BUCKET_COUNT = 1000 in rate_limiter.cc), so the pool stays full
// while the backlog drains.
constexpr size_t kFillerRequestCount = 8000;
// Time for the first 1000 one-ms filler executions to fill the pool.
constexpr auto kPoolFillDelay = std::chrono::milliseconds(1500);
constexpr int kBatchedExecDelayMs = 100;
// The preferred batch size (4) plus two: the batcher sends 4 requests at once
// and keeps the other 2 in its current payload, which the instance later
// merges into the next batch (4 + 2 <= max_batch_size 8).
constexpr size_t kBurstSize = 6;
// Slightly longer than one 'batched' execution, so bursts arrive while the
// instance is still busy with the previous one.
constexpr auto kBurstPeriod =
    std::chrono::milliseconds(kBatchedExecDelayMs + 5);
constexpr auto kBurstDuration = std::chrono::seconds(5);
constexpr auto kCompletionTimeout = std::chrono::seconds(120);

// While armed, the operator delete below keeps freed sizeof(std::mutex) blocks
// (such as a payload's exec mutex) out of the allocator and records their
// bytes, so a later write into a freed mutex is detected. It must not allocate.

struct QuarantineSlot {
  void* ptr;  // nullptr when the slot is empty
  unsigned char bytes[sizeof(std::mutex)];
};

constexpr size_t kQuarantineCapacity = 1 << 16;
QuarantineSlot quarantine_[kQuarantineCapacity];  // zero-initialized
size_t quarantine_next_ = 0;                      // guarded by spinlock
std::atomic_flag detector_spinlock_{};
std::atomic<bool> detector_armed_{false};
std::atomic<uint64_t> quarantined_count_{0};
std::atomic<uint64_t> written_after_free_count_{0};
const void* first_written_ptr_{nullptr};  // guarded by spinlock
size_t first_written_offset_{0};          // guarded by spinlock

struct DetectorResult {
  uint64_t quarantined;
  uint64_t written_after_free;
  const void* first_written_ptr;
  size_t first_written_offset;
};

// Caller must hold detector_spinlock_.
void
VerifyAndFree(QuarantineSlot& slot)
{
  if (slot.ptr == nullptr) {
    return;
  }
  const unsigned char* current =
      reinterpret_cast<const unsigned char*>(slot.ptr);
  for (size_t i = 0; i < sizeof(std::mutex); ++i) {
    if (current[i] != slot.bytes[i]) {
      if (written_after_free_count_.fetch_add(1, std::memory_order_relaxed) ==
          0) {
        first_written_ptr_ = slot.ptr;
        first_written_offset_ = i;
      }
      break;
    }
  }
  std::free(slot.ptr);
  slot.ptr = nullptr;
}

void
ArmDetector()
{
  detector_armed_.store(true, std::memory_order_release);
}

DetectorResult
DisarmDetectorAndVerify()
{
  detector_armed_.store(false, std::memory_order_release);
  while (detector_spinlock_.test_and_set(std::memory_order_acquire)) {
  }
  for (auto& slot : quarantine_) {
    VerifyAndFree(slot);
  }
  DetectorResult result{
      quarantined_count_.load(std::memory_order_relaxed),
      written_after_free_count_.load(std::memory_order_relaxed),
      first_written_ptr_, first_written_offset_};
  detector_spinlock_.clear(std::memory_order_release);
  return result;
}

#define FAIL_TEST_IF_ERR(X)                                                   \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = preferred_memory_type_id;
  *buffer = (byte_size == 0) ? nullptr : malloc(byte_size);
  *buffer_userp = nullptr;
  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::free(buffer);
  return nullptr;  // Success
}

constexpr char kFillerModelName[] = "filler";
constexpr char kBatchedModelName[] = "batched";

// Per-request shared progress, waited on by the test thread.
struct RequestTracker {
  bool Done() const { return completed >= sent && released >= sent; }

  std::mutex mu;
  std::condition_variable cv;
  size_t sent{0}, completed{0}, errored{0}, released{0};
};

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (flags & TRITONSERVER_REQUEST_RELEASE_ALL) {
    TRITONSERVER_InferenceRequestDelete(request);
    auto* tracker = reinterpret_cast<RequestTracker*>(userp);
    std::lock_guard<std::mutex> lk(tracker->mu);
    ++tracker->released;
    tracker->cv.notify_all();
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  auto* tracker = reinterpret_cast<RequestTracker*>(userp);
  std::lock_guard<std::mutex> lk(tracker->mu);
  if (response != nullptr) {
    if (TRITONSERVER_InferenceResponseError(response) != nullptr) {
      ++tracker->errored;
    }
    TRITONSERVER_InferenceResponseDelete(response);
  }
  if (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
    ++tracker->completed;
  }
  tracker->cv.notify_all();
}

}  // namespace

class DynamicBatchSchedulerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    repo_dir_ =
        std::filesystem::temp_directory_path() /
        ("dynamic_batch_scheduler_test." +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    WriteConfig(kFillerModelName, R"(
backend: "identity"
max_batch_size: 1
input [ { name: "INPUT0" data_type: TYPE_INT32 dims: [ 16 ] } ]
output [ { name: "OUTPUT0" data_type: TYPE_INT32 dims: [ 16 ] } ]
instance_group [ { count: 1 kind: KIND_CPU } ]
parameters [ { key: "execute_delay_ms" value: { string_value: "1" } } ]
)");
    WriteConfig(
        kBatchedModelName, R"(
backend: "identity"
max_batch_size: 8
dynamic_batching { preferred_batch_size: [ 4 ] max_queue_delay_microseconds: 1000 }
input [ { name: "INPUT0" data_type: TYPE_INT32 dims: [ 16 ] } ]
output [ { name: "OUTPUT0" data_type: TYPE_INT32 dims: [ 16 ] } ]
instance_group [ { count: 1 kind: KIND_CPU } ]
parameters [ { key: "execute_delay_ms" value: { string_value: ")" +
                               std::to_string(kBatchedExecDelayMs) + R"(" } } ]
)");

    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
        server_options, repo_dir_.c_str()));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(
        server_options, "/opt/tritonserver/backends"));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
        server_options, "/opt/tritonserver/repoagents"));
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true));
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetLogVerbose(server_options, 0));

    FAIL_TEST_IF_ERR(TRITONSERVER_ServerNew(&server_, server_options));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options));

    FAIL_TEST_IF_ERR(TRITONSERVER_ResponseAllocatorNew(
        &allocator_, ResponseAlloc, ResponseRelease, nullptr /* start_fn */));
  }

  static void TearDownTestSuite()
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_));
    FAIL_TEST_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator_));
    std::error_code ec;
    std::filesystem::remove_all(repo_dir_, ec);
  }

  static void WriteConfig(const std::string& name, const std::string& config)
  {
    const std::filesystem::path dir = repo_dir_ / name;
    std::error_code ec;
    ASSERT_TRUE(std::filesystem::create_directories(dir / "1", ec))
        << ec.message();
    std::ofstream out(dir / "config.pbtxt");
    ASSERT_TRUE(out.good());
    out << config;
  }

  void SetUp() override
  {
    ASSERT_TRUE(server_ != nullptr) << "Server has not been created";
    // Wait until the server is live and ready and both models are loaded.
    const auto deadline = std::chrono::steady_clock::now() + kCompletionTimeout;
    bool live = false, ready = false;
    while ((!live || !ready) && std::chrono::steady_clock::now() < deadline) {
      FAIL_TEST_IF_ERR(TRITONSERVER_ServerIsLive(server_, &live));
      FAIL_TEST_IF_ERR(TRITONSERVER_ServerIsReady(server_, &ready));
      for (const char* model : {kFillerModelName, kBatchedModelName}) {
        bool model_ready = false;
        FAIL_TEST_IF_ERR(
            TRITONSERVER_ServerModelIsReady(server_, model, -1, &model_ready));
        ready = ready && model_ready;
      }
      if (!live || !ready) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    ASSERT_TRUE(live && ready)
        << "Timed out waiting for healthy inference server and models";
  }

  bool SendRequest(const char* model_name, RequestTracker* tracker)
  {
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    std::shared_ptr<TRITONSERVER_Error> err(
        TRITONSERVER_InferenceRequestNew(
            &irequest, server_, model_name, -1 /* model_version */),
        TRITONSERVER_ErrorDelete);
    if (err != nullptr) {
      return false;
    }
    const int64_t shape[] = {1, 16};
    err.reset(
        TRITONSERVER_InferenceRequestAddInput(
            irequest, "INPUT0", TRITONSERVER_TYPE_INT32, shape, 2),
        TRITONSERVER_ErrorDelete);
    if (err == nullptr) {
      err.reset(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, "INPUT0", input0_data_.data(),
              input0_data_.size() * sizeof(input0_data_[0]),
              TRITONSERVER_MEMORY_CPU, 0),
          TRITONSERVER_ErrorDelete);
    }
    if (err == nullptr) {
      err.reset(
          TRITONSERVER_InferenceRequestSetReleaseCallback(
              irequest, InferRequestComplete, tracker),
          TRITONSERVER_ErrorDelete);
    }
    if (err == nullptr) {
      err.reset(
          TRITONSERVER_InferenceRequestSetResponseCallback(
              irequest, allocator_, nullptr /* response_allocator_userp */,
              InferResponseComplete, tracker),
          TRITONSERVER_ErrorDelete);
    }
    if (err == nullptr) {
      std::shared_ptr<TRITONSERVER_Error> infer_err(
          TRITONSERVER_ServerInferAsync(server_, irequest, nullptr /* trace */),
          TRITONSERVER_ErrorDelete);
      if (infer_err == nullptr) {
        // Success: ownership passed to the server.
        std::lock_guard<std::mutex> lk(tracker->mu);
        ++tracker->sent;
        return true;
      }
    }
    TRITONSERVER_InferenceRequestDelete(irequest);
    return false;
  }

  void WaitComplete(RequestTracker* tracker)
  {
    std::unique_lock<std::mutex> lk(tracker->mu);
    ASSERT_TRUE(tracker->cv.wait_for(
        lk, kCompletionTimeout, [&] { return tracker->Done(); }))
        << "Timed out: sent=" << tracker->sent
        << " completed=" << tracker->completed
        << " released=" << tracker->released;
  }

  // Requests still in flight when a test fails use these, so they must outlive
  // the server.
  static TRITONSERVER_Server* server_;
  static std::filesystem::path repo_dir_;
  static TRITONSERVER_ResponseAllocator* allocator_;
  static std::vector<int32_t> input0_data_;
  static RequestTracker filler_tracker_;
  static RequestTracker batched_tracker_;
};

TRITONSERVER_Server* DynamicBatchSchedulerTest::server_ = nullptr;
std::filesystem::path DynamicBatchSchedulerTest::repo_dir_;
TRITONSERVER_ResponseAllocator* DynamicBatchSchedulerTest::allocator_ = nullptr;
std::vector<int32_t> DynamicBatchSchedulerTest::input0_data_(16, 1);
RequestTracker DynamicBatchSchedulerTest::filler_tracker_;
RequestTracker DynamicBatchSchedulerTest::batched_tracker_;

// libstdc++'s operator new pairs with std::free, hence the same here.
void
operator delete(void* ptr, std::size_t size) noexcept
{
  if (ptr == nullptr) {
    return;
  }
  // Keyed on size only: any write into a freed mutex-sized block counts.
  if (size == sizeof(std::mutex) &&
      detector_armed_.load(std::memory_order_acquire)) {
    while (detector_spinlock_.test_and_set(std::memory_order_acquire)) {
    }
    // Reusing a slot: verify its previous occupant before overwriting it.
    VerifyAndFree(quarantine_[quarantine_next_]);
    QuarantineSlot& slot = quarantine_[quarantine_next_];
    slot.ptr = ptr;
    std::memcpy(slot.bytes, ptr, sizeof(std::mutex));
    quarantine_next_ = (quarantine_next_ + 1) % kQuarantineCapacity;
    quarantined_count_.fetch_add(1, std::memory_order_relaxed);
    detector_spinlock_.clear(std::memory_order_release);
    return;
  }
  std::free(ptr);
}

void
operator delete(void* ptr) noexcept
{
  if (ptr != nullptr) {
    std::free(ptr);
  }
}

TEST_F(DynamicBatchSchedulerTest, MergedPayloadExecMutexNotUsedAfterFree)
{
  ArmDetector();

  // Fill the rate limiter's payload pool: the filler model has no dynamic
  // batching, so each request owns a payload and payloads are never merged.
  for (size_t i = 0; i < kFillerRequestCount; ++i) {
    ASSERT_TRUE(SendRequest(kFillerModelName, &filler_tracker_))
        << "Failed to send filler request " << i;
  }

  // Let the pool saturate (1000 payloads) while the filler backlog drains.
  std::this_thread::sleep_for(kPoolFillDelay);

  // Each burst yields one saturated payload and one leftover payload that
  // the batcher keeps; the next instance dequeue merges and releases the
  // leftover to the full pool while the batcher is its sole owner.
  const auto burst_start = std::chrono::steady_clock::now();
  for (size_t burst = 0;; ++burst) {
    if (std::chrono::steady_clock::now() - burst_start >= kBurstDuration) {
      break;
    }
    for (size_t i = 0; i < kBurstSize; ++i) {
      ASSERT_TRUE(SendRequest(kBatchedModelName, &batched_tracker_))
          << "Failed to send batched request " << i << " of burst " << burst;
    }
    std::this_thread::sleep_until(burst_start + (burst + 1) * kBurstPeriod);
  }

  WaitComplete(&filler_tracker_);
  WaitComplete(&batched_tracker_);

  const DetectorResult result = DisarmDetectorAndVerify();

  EXPECT_EQ(filler_tracker_.errored, 0);
  EXPECT_EQ(batched_tracker_.errored, 0);
  ASSERT_GT(result.quarantined, 0u)
      << "no sizeof(std::mutex) deletes seen: sized deallocation (the GCC "
         "default) is required for the interposed delete to run";
  EXPECT_EQ(result.written_after_free, 0)
      << "payload exec mutex written after free "
         "(BatcherThread unlocked a destroyed payload's exec mutex): "
      << result.written_after_free << " of " << result.quarantined
      << " quarantined block(s), first at " << result.first_written_ptr
      << ", first changed byte at offset " << result.first_written_offset;
}

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
