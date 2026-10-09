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

#ifdef TRITON_INSTANCE_QUEUE_TEST_BACKEND
#include "triton/core/tritonbackend.h"

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance*, TRITONBACKEND_Request**, const uint32_t)
{
  return nullptr;
}
#else

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <vector>

#include "dynamic_batch_scheduler.h"
#include "gtest/gtest.h"
#include "instance_queue.h"
#include "payload.h"
#include "server.h"

namespace triton { namespace core {

namespace {

// Builds an empty INFER_RUN payload. Empty payloads are sufficient here: they
// carry no requests, so they always satisfy the batch-size condition and merge
// unconditionally once the queue delay has elapsed (BatcherStartNs() == 0).
std::shared_ptr<Payload>
MakeInferPayload()
{
  auto payload = std::make_shared<Payload>();
  payload->Reset(Payload::Operation::INFER_RUN, nullptr /* instance */);
  return payload;
}

// Reproduces the waiting-consumer bookkeeping that RateLimiter performs around
// a single per-model InstanceQueue and asserts it does not drift when Dequeue
// merges payloads.
//
// RateLimiter drives the counter with exactly two moves on this queue:
//   * EnqueuePayload  -> DecrementConsumerCount(), once per enqueued payload.
//   * DequeuePayload  -> IncrementConsumerCount(), once per dequeue call.
// When Dequeue merges k payloads into one, the k payloads were each decremented
// at enqueue but only a single increment is issued for the dequeue call, so the
// count leaks -(k-1) per merge. Over many merge-heavy rounds this drives
// waiting_consumer_count_ negative, which throttles the dynamic batcher onto
// its slow 500 ms poll fallback (the dispatch gates require
// WaitingConsumerCount() > 0), degrading throughput.
//
// With the fix (Dequeue credits back one count per merged payload) the counter
// returns to the idle-instance count after every round.
TEST(InstanceQueueTest, ConsumerCountStableAcrossMerges)
{
  constexpr size_t kMaxBatchSize = 8;
  // Small, non-zero delay: pending payloads are always "old enough" to merge.
  constexpr uint64_t kMaxQueueDelayNs = 1000;
  constexpr int kNumInstances = 4;
  constexpr int kRounds = 50;
  // 1 primary payload + (kBurst - 1) merged payloads per dequeue.
  constexpr int kBurst = 3;

  InstanceQueue queue(kMaxBatchSize, kMaxQueueDelayNs);

  // Model start-up: every instance parks in DequeuePayload once, announcing
  // itself as a waiting consumer.
  for (int i = 0; i < kNumInstances; ++i) {
    queue.IncrementConsumerCount();
  }
  ASSERT_EQ(queue.WaitingConsumerCount(), kNumInstances);

  for (int round = 0; round < kRounds; ++round) {
    // Producer enqueues a burst; each enqueue claims one waiting consumer.
    for (int b = 0; b < kBurst; ++b) {
      auto payload = MakeInferPayload();
      queue.Enqueue(payload);
      queue.DecrementConsumerCount();
    }

    // One consumer dequeues the whole burst as a single merged batch.
    std::shared_ptr<Payload> payload;
    std::vector<std::shared_ptr<Payload>> merged_payloads;
    queue.Dequeue(&payload, &merged_payloads);

    ASSERT_NE(payload, nullptr);
    ASSERT_EQ(merged_payloads.size(), static_cast<size_t>(kBurst - 1))
        << "test setup expects all burst payloads to merge into one dequeue";

    // Consumer finishes and re-parks for the next round.
    queue.IncrementConsumerCount();
  }

  // Once all payloads have been consumed and every consumer is idle again, the
  // waiting-consumer count must equal the true idle-instance count. Without the
  // fix it ends deeply negative (kNumInstances - kRounds * (kBurst - 1)).
  EXPECT_EQ(queue.WaitingConsumerCount(), kNumInstances)
      << "waiting_consumer_count_ drifted across merges; the dynamic batcher "
         "would eventually be throttled onto its slow poll fallback";
}

// A dequeue that merges nothing (single queued payload) must not change the
// waiting-consumer count beyond the single increment the caller issues.
TEST(InstanceQueueTest, ConsumerCountUnchangedWithoutMerge)
{
  constexpr size_t kMaxBatchSize = 8;
  constexpr uint64_t kMaxQueueDelayNs = 1000;

  InstanceQueue queue(kMaxBatchSize, kMaxQueueDelayNs);
  queue.IncrementConsumerCount();  // one idle instance
  ASSERT_EQ(queue.WaitingConsumerCount(), 1);

  auto payload_in = MakeInferPayload();
  queue.Enqueue(payload_in);
  queue.DecrementConsumerCount();

  std::shared_ptr<Payload> payload;
  std::vector<std::shared_ptr<Payload>> merged_payloads;
  queue.Dequeue(&payload, &merged_payloads);

  ASSERT_NE(payload, nullptr);
  EXPECT_TRUE(merged_payloads.empty());

  queue.IncrementConsumerCount();  // consumer re-parks
  EXPECT_EQ(queue.WaitingConsumerCount(), 1);
}

TEST(InstanceQueueTest, TimedWaitSeesConsumerBeforeWaiting)
{
  InstanceQueue queue(1, 0);
  queue.IncrementConsumerCount();
  EXPECT_TRUE(queue.WaitForConsumer(std::chrono::microseconds(0)));
  // Waiting observes availability without reserving the consumer.
  EXPECT_EQ(queue.WaitingConsumerCount(), 1);
}

TEST(InstanceQueueTest, TimedWaitExpiresWithoutConsumer)
{
  InstanceQueue queue(1, 0);
  EXPECT_FALSE(queue.WaitForConsumer(std::chrono::milliseconds(10)));
  EXPECT_EQ(queue.WaitingConsumerCount(), 0);
}

TEST(InstanceQueueTest, TimedWaitWakesWhenConsumerBecomesAvailable)
{
  InstanceQueue queue(1, 0);
  auto waiter = std::async(std::launch::async, [&queue]() {
    return queue.WaitForConsumer(std::chrono::seconds(2));
  });
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(50)),
      std::future_status::timeout);
  queue.IncrementConsumerCount();
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(250)),
      std::future_status::ready);
  EXPECT_TRUE(waiter.get());
}

TEST(InstanceQueueTest, TimedWaitRequiresPositiveConsumerCount)
{
  InstanceQueue queue(1, 0);
  queue.DecrementConsumerCount();
  auto waiter = std::async(std::launch::async, [&queue]() {
    return queue.WaitForConsumer(std::chrono::seconds(2));
  });
  queue.IncrementConsumerCount();
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(50)),
      std::future_status::timeout);
  queue.IncrementConsumerCount();
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(250)),
      std::future_status::ready);
  EXPECT_TRUE(waiter.get());
}

TEST(InstanceQueueTest, ConsumerNotificationRacingWithTimedWaitIsNotLost)
{
  for (int i = 0; i < 100; ++i) {
    InstanceQueue queue(1, 0);
    std::promise<void> start;
    auto waiter = std::async(std::launch::async, [&queue, &start]() {
      start.set_value();
      return queue.WaitForConsumer(std::chrono::seconds(2));
    });
    start.get_future().wait();
    queue.IncrementConsumerCount();
    EXPECT_EQ(
        waiter.wait_for(std::chrono::milliseconds(250)),
        std::future_status::ready)
        << "iteration " << i;
    EXPECT_TRUE(waiter.get());
  }
}

}  // namespace

using namespace std::chrono_literals;

// Public factories create passive instances so tests can control availability
// through RateLimiter::DequeuePayload without a backend thread.
class PayloadSlotWaitTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override
  {
    const auto models =
        std::filesystem::canonical("/proc/self/exe").parent_path() /
        "instance_queue_test_models";
    server_.SetModelRepositoryPaths({(models / "empty").string()});
    server_.SetModelControlMode(ModelControlMode::MODE_EXPLICIT);
    server_.SetPinnedMemoryPoolByteSize(0);
    server_.SetRepoAgentDir((models / "empty").string());
    server_.SetCacheDir((models / "empty").string());
    server_.SetRateLimiterMode(RateLimitMode::RL_OFF);
    server_.SetEnablePeerAccess(false);
    server_.SetResponseCacheEnabled(false);
    const triton::common::BackendCmdlineConfigMap backend_config{
        {"",
         {{"backend-directory", models.string()},
          {"auto-complete-config", "false"},
          {"min-compute-capability", "0"}}}};
    server_.SetBackendCmdlineConfig(backend_config);
    auto status = server_.Init();
    ASSERT_TRUE(status.IsOk()) << status.Message();
    inference::ModelConfig config;
    config.set_name("test");
    config.set_backend("test");
    config.set_max_batch_size(1);
    config.mutable_version_policy()->mutable_latest()->set_num_versions(1);
    auto* group = config.add_instance_group();
    group->set_kind(inference::ModelInstanceGroup::KIND_CPU);
    group->set_count(1);
    group->set_passive(true);
    batcher_config_.add_preferred_batch_size(1);
    batcher_config_.mutable_default_queue_policy()->set_max_queue_size(
        GetParam() ? 0 : 1);
    status = TritonModel::Create(
        &server_, (models / "test").string(), backend_config, {},
        ModelIdentifier("", "test"), 1, config, true, &model_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = TritonModelInstance::CreateInstance(
        model_.get(), "test", TritonModelInstance::Signature(*group, 0),
        TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {}, true, "", {}, {},
        &instance_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    ASSERT_TRUE(server_.GetRateLimiter()
                    ->RegisterModelInstance(instance_.get(), {})
                    .IsOk());
  }

  void TearDown() override
  {
    scheduler_.reset();
    instance_.reset();
    model_.reset();
  }

  void FillPrefetchQueue()
  {
    for (int i = 0; i < 2; ++i) {
      ASSERT_TRUE(server_.GetRateLimiter()
                      ->EnqueuePayload(
                          model_.get(), server_.GetRateLimiter()->GetPayload(
                                            Payload::Operation::INFER_RUN))
                      .IsOk());
    }
  }

  std::future<std::shared_ptr<Payload>> Dequeue()
  {
    return std::async(std::launch::async, [this]() {
      std::deque<TritonModelInstance*> instances{instance_.get()};
      std::shared_ptr<Payload> payload;
      server_.GetRateLimiter()->DequeuePayload(instances, &payload);
      return payload;
    });
  }

  void EnqueueRequest()
  {
    auto request = std::make_unique<InferenceRequest>(model_.get(), 1);
    ASSERT_TRUE(request->SetResponseCallback(nullptr, nullptr, nullptr, nullptr)
                    .IsOk());
    ASSERT_TRUE(request->PrepareForInference().IsOk());
    ASSERT_TRUE(scheduler_->Enqueue(request).IsOk());
  }

  void StartBatcher()
  {
    auto status = DynamicBatchScheduler::Create(
        model_.get(), nullptr, 0, true, 1, {}, batcher_config_, &scheduler_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    // Let the batcher enter its empty-queue wait before the first admission.
    std::this_thread::sleep_for(50ms);
  }

  void StartSlotWait()
  {
    StartBatcher();
    EnqueueRequest();
    // A full prefetch queue suppresses the admission notification. Allow the
    // initial 500 ms scheduler wait to finish before checking the slot wait.
    std::this_thread::sleep_for(600ms);
    EXPECT_EQ(scheduler_->InflightInferenceCount(), 1);
  }

  bool WaitForDispatch()
  {
    const auto deadline = std::chrono::steady_clock::now() + 250ms;
    while (scheduler_->InflightInferenceCount() != 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }

  InferenceServer server_;
  std::unique_ptr<TritonModel> model_;
  std::shared_ptr<TritonModelInstance> instance_;
  inference::ModelDynamicBatching batcher_config_;
  std::unique_ptr<Scheduler> scheduler_;
};

TEST_P(PayloadSlotWaitTest, BackendAvailabilityWakesSlotWait)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  StartSlotWait();
  auto consumer = Dequeue();
  // No later admission releases the blocked batcher.
  EXPECT_TRUE(WaitForDispatch());
  EXPECT_EQ(consumer.wait_for(250ms), std::future_status::ready);
  EXPECT_NE(consumer.get(), nullptr);
}

TEST_P(PayloadSlotWaitTest, ShutdownInterruptsSlotWait)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  StartSlotWait();
  auto start = std::chrono::steady_clock::now();
  scheduler_.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(PayloadSlotWaitTest, StopBeforeSlotWaitDoesNotBlock)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  std::stop_source stop;
  stop.request_stop();
  auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(server_.GetRateLimiter()->PayloadSlotAvailable(
      model_.get(), nullptr, GetParam(), false, 5000000, stop.get_token()));
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(PayloadSlotWaitTest, FirstAdmissionWhileBusyWakesBatcher)
{
  StartBatcher();
  EnqueueRequest();
  std::this_thread::sleep_for(50ms);

  auto consumer = Dequeue();
  EXPECT_EQ(consumer.wait_for(250ms), std::future_status::ready);
  auto payload = consumer.get();
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->RequestCount(), 1);
}

TEST_P(PayloadSlotWaitTest, ShutdownInterruptsEmptyQueueWait)
{
  StartBatcher();
  auto start = std::chrono::steady_clock::now();
  scheduler_.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(PayloadSlotWaitTest, StopRacingWithSlotWaitDoesNotBlock)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  for (int i = 0; i < 100; ++i) {
    std::stop_source stop;
    std::promise<void> start;
    auto waiter = std::async(std::launch::async, [&]() {
      start.set_value();
      return server_.GetRateLimiter()->PayloadSlotAvailable(
          model_.get(), nullptr, GetParam(), false, 5000000, stop.get_token());
    });
    start.get_future().wait();
    stop.request_stop();
    EXPECT_EQ(waiter.wait_for(250ms), std::future_status::ready) << i;
    EXPECT_FALSE(waiter.get());
  }
}

INSTANTIATE_TEST_SUITE_P(
    QueuePolicy, PayloadSlotWaitTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Prefetch" : "Finite";
    });

}}  // namespace triton::core
#endif
