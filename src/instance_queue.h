// SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "payload.h"

namespace triton { namespace core {

//
// InstanceQueue
//
// A queue implementation holding Payloads ready to be scheduled on
// model instance.
class InstanceQueue {
 public:
  explicit InstanceQueue(size_t max_batch_size, uint64_t max_queue_delay_ns);

  size_t Size();
  bool Empty();
  void Enqueue(const std::shared_ptr<Payload>& payload);
  void Dequeue(
      std::shared_ptr<Payload>* payload,
      std::vector<std::shared_ptr<Payload>>* merged_payloads);

  void IncrementConsumerCount();
  void DecrementConsumerCount();
  void WaitForConsumer();
  int WaitingConsumerCount();

 private:
  size_t max_batch_size_;
  uint64_t max_queue_delay_ns_;

  std::deque<std::shared_ptr<Payload>> payload_queue_;
  std::shared_ptr<Payload> staged_payload_;
  std::mutex mu_;

  int waiting_consumer_count_;
  std::mutex waiting_consumer_mu_;
  std::condition_variable waiting_consumer_cv_;
};

}}  // namespace triton::core
