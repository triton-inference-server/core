// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dynamic_batch_scheduler.h"

#include <chrono>
#include <future>

#include "gtest/gtest.h"
#include "server.h"

namespace triton { namespace core {

using namespace std::chrono_literals;

// Construct real model/instance objects without starting a backend thread.
// Each test controls availability through RateLimiter::DequeuePayload.
class DynamicBatchSchedulerTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override
  {
    std::unique_ptr<RateLimiter> rate_limiter;
    ASSERT_TRUE(RateLimiter::Create(true, {}, &rate_limiter).IsOk());
    server_.rate_limiter_ = std::move(rate_limiter);

    auto backend = std::shared_ptr<TritonBackend>(
        new TritonBackend("test", "", "", TritonServerMessage("{}")));
    inference::ModelConfig config;
    config.set_name("test");
    config.set_backend("test");
    config.set_max_batch_size(1);
    config.mutable_version_policy()->mutable_latest()->set_num_versions(1);
    config.mutable_dynamic_batching()
        ->mutable_default_queue_policy()
        ->set_max_queue_size(GetParam() ? 0 : 1);
    model_.reset(new TritonModel(
        &server_, std::make_shared<LocalizedPath>(""), backend, 0,
        ModelIdentifier("", "test"), 1, config, false, {}, {}));
    auto status = model_->Init(true);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    instance_.reset(new TritonModelInstance(
        model_.get(), "test", TritonModelInstance::Signature({}, 0),
        TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {}, false, {},
        TritonServerMessage("{}"), {}));
    ASSERT_TRUE(server_.GetRateLimiter()
                    ->RegisterModelInstance(instance_.get(), {})
                    .IsOk());
    scheduler_.reset(new DynamicBatchScheduler(
        model_.get(), nullptr, true, 1, {}, false, {1}, 0,
        config.dynamic_batching().default_queue_policy(), 0, {}));
    scheduler_->NewPayload();
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

  std::future<bool> StartSlotWait()
  {
    std::promise<bool> result;
    auto future = result.get_future();
    scheduler_->scheduler_thread_ = std::thread(
        [scheduler = scheduler_.get(), result = std::move(result)]() mutable {
          std::unique_lock<std::mutex> lock(scheduler->mu_);
          result.set_value(
              scheduler->WaitForPayloadSlotAvailable(&lock, 5000000));
        });
    return future;
  }

  void StartBatcher()
  {
    scheduler_->scheduler_thread_ = std::thread(
        [scheduler = scheduler_.get()]() { scheduler->BatcherThread(0); });
    // Let the batcher enter its empty-queue wait before the first admission.
    std::this_thread::sleep_for(50ms);
  }

  InferenceServer server_;
  std::unique_ptr<TritonModel> model_;
  std::unique_ptr<TritonModelInstance> instance_;
  std::unique_ptr<DynamicBatchScheduler> scheduler_;
};

TEST_P(DynamicBatchSchedulerTest, BackendAvailabilityWakesSlotWait)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  auto waiter = StartSlotWait();
  EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);

  auto consumer = Dequeue();
  // No new admission or scheduler-CV notification occurs here.
  EXPECT_EQ(waiter.wait_for(250ms), std::future_status::ready);
  if (!GetParam()) {
    // Unblock the consumer after checking the producer's wake-up.
    EXPECT_TRUE(server_.GetRateLimiter()
                    ->EnqueuePayload(
                        model_.get(), server_.GetRateLimiter()->GetPayload(
                                          Payload::Operation::INFER_RUN))
                    .IsOk());
  }
  EXPECT_NE(consumer.get(), nullptr);
  scheduler_.reset();
  EXPECT_TRUE(waiter.get());
}

TEST_P(DynamicBatchSchedulerTest, ShutdownInterruptsSlotWait)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  auto waiter = StartSlotWait();
  EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);
  auto start = std::chrono::steady_clock::now();
  scheduler_.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
  EXPECT_FALSE(waiter.get());
}

TEST_P(DynamicBatchSchedulerTest, StopBeforeSlotWaitDoesNotBlock)
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

TEST_P(DynamicBatchSchedulerTest, FirstAdmissionWhileBusyWakesBatcher)
{
  StartBatcher();
  auto request = std::make_unique<InferenceRequest>(model_.get(), 1);
  ASSERT_TRUE(
      request->SetResponseCallback(nullptr, nullptr, nullptr, nullptr).IsOk());
  ASSERT_TRUE(request->PrepareForInference().IsOk());
  ASSERT_TRUE(scheduler_->Enqueue(request).IsOk());
  std::this_thread::sleep_for(50ms);

  auto consumer = Dequeue();
  EXPECT_EQ(consumer.wait_for(250ms), std::future_status::ready);
  auto payload = consumer.get();
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->RequestCount(), 1);
}

TEST_P(DynamicBatchSchedulerTest, ShutdownInterruptsEmptyQueueWait)
{
  StartBatcher();
  auto start = std::chrono::steady_clock::now();
  scheduler_.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(DynamicBatchSchedulerTest, StopRacingWithSlotWaitDoesNotBlock)
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
    QueuePolicy, DynamicBatchSchedulerTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Prefetch" : "Finite";
    });

}}  // namespace triton::core
