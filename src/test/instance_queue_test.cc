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

#include "instance_queue.h"

#include <iterator>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "payload.h"

namespace triton { namespace core {

// Keep this unit test scoped to InstanceQueue. The production Payload
// implementation pulls in backend/scheduler symbols that are unrelated to the
// queue accounting exercised here, so provide the minimal behavior Dequeue uses.
Payload::Payload()
    : op_type_(Operation::INFER_RUN),
      requests_(std::vector<std::unique_ptr<InferenceRequest>>()),
      OnCallback_([]() {}), instance_(nullptr), state_(State::UNINITIALIZED),
      batcher_start_ns_(0), saturated_(false), user_pointer_(nullptr)
{
  exec_mu_.reset(new std::mutex());
}

void
Payload::Reset(const Operation op_type, TritonModelInstance* instance)
{
  op_type_ = op_type;
  requests_.clear();
  OnCallback_ = []() {};
  release_callbacks_.clear();
  instance_ = instance;
  state_ = State::UNINITIALIZED;
  status_.reset(new std::promise<Status>());
  required_equal_inputs_ = RequiredEqualInputs();
  batcher_start_ns_ = 0;
  saturated_ = false;
  user_pointer_ = nullptr;
}

const Status&
Payload::MergePayload(std::shared_ptr<Payload>& payload)
{
  static const Status success(Status::Code::SUCCESS);
  requests_.insert(
      requests_.end(), std::make_move_iterator(payload->Requests().begin()),
      std::make_move_iterator(payload->Requests().end()));
  payload->Callback();
  return success;
}

size_t
Payload::BatchSize()
{
  return requests_.size();
}

void
Payload::Callback()
{
  OnCallback_();
}

void
Payload::SetState(Payload::State state)
{
  state_ = state;
}

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

// Reproduces the waiting-consumer bookkeeping that RateLimiter performs around a
// single per-model InstanceQueue and asserts it does not drift when Dequeue
// merges payloads.
//
// RateLimiter drives the counter with exactly two moves on this queue:
//   * EnqueuePayload  -> DecrementConsumerCount(), once per enqueued payload.
//   * DequeuePayload  -> IncrementConsumerCount(), once per dequeue call.
// When Dequeue merges k payloads into one, the k payloads were each decremented
// at enqueue but only a single increment is issued for the dequeue call, so the
// count leaks -(k-1) per merge. Over many merge-heavy rounds this drives
// waiting_consumer_count_ negative, which throttles the dynamic batcher onto its
// slow 500 ms poll fallback (the dispatch gates require
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

}  // namespace
}}  // namespace triton::core
