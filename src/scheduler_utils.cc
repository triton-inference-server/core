// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "scheduler_utils.h"

#include <cassert>

#include "constants.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

uint64_t
CaptureTimeNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool
CacheLookUpUtil(
    std::unique_ptr<InferenceRequest>& request,
    std::unique_ptr<InferenceResponse>& cached_response,
    std::shared_ptr<TritonCache> cache)
{
  Status status;
  std::unique_ptr<InferenceResponse> local_response;
  request->ResponseFactory()->CreateResponse(&local_response);
  std::string key = "";
  if (!request->CacheKeyIsSet()) {
    status = cache->Hash(*request, &key);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to hash request: " << status.Message();
      return false;
    }
    request->SetCacheKey(key);
  } else {
    key = request->CacheKey();
  }
  request->CaptureCacheLookupStartNs();
  status = cache->Lookup(local_response.get(), key);
  request->CaptureCacheLookupEndNs();
  if (status.IsOk() && (local_response != nullptr)) {
    cached_response = std::move(local_response);
    return true;
  }
  return false;
}

Status
RequiredEqualInputs::Initialize(
    const std::unique_ptr<InferenceRequest>& request,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool has_optional_input)
{
  has_optional_input_ = has_optional_input;
  required_inputs_.clear();

  for (const auto& pr : request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const auto itr = enforce_equal_shape_tensors.find(input->Name());
    if (itr != enforce_equal_shape_tensors.end()) {
      required_inputs_.emplace(
          std::piecewise_construct, std::forward_as_tuple(input->Name()),
          std::forward_as_tuple(input, itr->second));
    }
    // When the model has optional inputs, overload 'required_inputs_'
    // to track the inputs involved in the batch
    else if (has_optional_input) {
      required_inputs_.emplace(
          std::piecewise_construct, std::forward_as_tuple(input->Name()),
          std::forward_as_tuple(nullptr, false));
    }
  }

  init_ = true;
  return Status::Success;
}

bool
RequiredEqualInputs::HasEqualInputs(
    const std::unique_ptr<InferenceRequest>& request)
{
  // If current request has different number of inputs, then dynamic batching
  // shouldn't be applied.
  if (has_optional_input_ &&
      (request->ImmutableInputs().size() != required_inputs_.size())) {
    return false;
  }
  for (const auto& pr : request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const auto itr = required_inputs_.find(input->Name());
    if (itr != required_inputs_.end()) {
      if (itr->second.first != nullptr) {
        // Make sure shape of input tensors is equal.
        if (!triton::common::CompareDims(
                itr->second.first->Shape(), input->Shape())) {
          return false;
        }

        // If necessary compare the contents as well...
        if (itr->second.second) {
          const auto& d1 = itr->second.first->Data();
          const auto& d2 = input->Data();

          // For now being conservative and assuming that content
          // comparison is for shape tensors which are likely to always
          // be in a single buffer.
          if ((d1->BufferCount() != 1) || (d2->BufferCount() != 1)) {
            return false;
          }

          size_t d1_byte_size, d2_byte_size;
          TRITONSERVER_MemoryType d1_memory_type, d2_memory_type;
          int64_t d1_memory_id, d2_memory_id;
          const char* d1_buffer = d1->BufferAt(
              0 /* idx */, &d1_byte_size, &d1_memory_type, &d1_memory_id);
          const char* d2_buffer = d2->BufferAt(
              0 /* idx */, &d2_byte_size, &d2_memory_type, &d2_memory_id);

          // Tensor must be same size and in in CPU memory so that it
          // can be easily compared. If not return false conservatively.
          if ((d1_byte_size != d2_byte_size) || (d1_buffer == nullptr) ||
              (d2_buffer == nullptr) ||
              (d1_memory_type == TRITONSERVER_MEMORY_GPU) ||
              (d2_memory_type == TRITONSERVER_MEMORY_GPU)) {
            return false;
          }

          if (strncmp(d1_buffer, d2_buffer, d1_byte_size) != 0) {
            return false;
          }
        }
      }
    } else if (has_optional_input_) {
      // If the model has optional inputs, the current request must contains all
      // inputs that in the first request (tracked in 'required_inputs_').
      return false;
    }
  }

  return true;
}

Status
PriorityQueue::PolicyQueue::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  if ((max_queue_size_ != 0) && (Size() >= max_queue_size_)) {
    return Status(
        Status::Code::UNAVAILABLE,
        request->LogRequest() + "Exceeds maximum queue size");
  }

  queue_.emplace_back(std::move(request));
  auto timeout_us = default_timeout_us_;
  if (allow_timeout_override_) {
    auto override_timeout_us = queue_.back()->TimeoutMicroseconds();
    if (override_timeout_us != 0 && override_timeout_us < timeout_us) {
      timeout_us = override_timeout_us;
    }
  }
  if (timeout_us != 0) {
    timeout_timestamp_ns_.emplace_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count() +
        timeout_us * 1000);
  } else {
    timeout_timestamp_ns_.emplace_back(0);
  }

  return Status::Success;
}

Status
PriorityQueue::PolicyQueue::Dequeue(std::unique_ptr<InferenceRequest>* request)
{
  if (!queue_.empty()) {
    *request = std::move(queue_.front());
    queue_.pop_front();
    timeout_timestamp_ns_.pop_front();
  } else {
    *request = std::move(delayed_queue_.front());
    delayed_queue_.pop_front();
  }

  return Status::Success;
}

bool
PriorityQueue::PolicyQueue::ApplyPolicy(
    size_t idx, size_t* rejected_count, size_t* rejected_batch_size,
    size_t* cancelled_count, size_t* cancelled_batch_size)
{
  uint64_t now_nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  if (idx < queue_.size()) {
    size_t curr_idx = idx;

    // Advance curr_idx until a request that goes into a batch, if not already.
    while (curr_idx < queue_.size()) {
      // Cancel request at curr_idx if it is marked as cancelled.
      if (queue_[curr_idx]->IsCancelled()) {
        cancelled_queue_.emplace_back(std::move(queue_[curr_idx]));
        *cancelled_count += 1;
        *cancelled_batch_size +=
            std::max(1U, cancelled_queue_.back()->BatchSize());
        curr_idx++;
      }
      // Delay or reject request at curr_idx if it is expired.
      else if (
          (timeout_timestamp_ns_[curr_idx] != 0) &&
          (now_nanoseconds > timeout_timestamp_ns_[curr_idx])) {
        if (timeout_action_ == inference::ModelQueuePolicy::DELAY) {
          delayed_queue_.emplace_back(std::move(queue_[curr_idx]));
        } else {
          rejected_queue_.emplace_back(std::move(queue_[curr_idx]));
          *rejected_count += 1;
          *rejected_batch_size +=
              std::max(1U, rejected_queue_.back()->BatchSize());
        }
        curr_idx++;
      }
      // Request at curr_idx is unexpired and non-cancelled.
      else {
        break;
      }
    }

    // Erase requests that are not going into a batch from queue_.
    // Use range erasure on deque as all erasure functions are linear,
    // this implies in the edge case where this function is always called on
    // 'bad' index can be O(n^2). However, for data structures that are O(1)
    // erasure, the traversal may not be as efficient due to cache miss
    // (elements not stored contiguously).
    queue_.erase(queue_.begin() + idx, queue_.begin() + curr_idx);
    timeout_timestamp_ns_.erase(
        timeout_timestamp_ns_.begin() + idx,
        timeout_timestamp_ns_.begin() + curr_idx);

    // Check if idx is still in range after erasing requests.
    if (idx < queue_.size()) {
      return true;
    }
  }
  // At this point, idx >= queue_.size().
  // If the item is in delayed queue, then return true. Otherwise, false
  // meaning the queue has no item with this 'idx'.
  return ((idx - queue_.size()) < delayed_queue_.size());
}

size_t
PriorityQueue::PolicyQueue::RejectTimeoutRequests()
{
  if (timeout_action_ != inference::ModelQueuePolicy::REJECT) {
    return 0;
  }

  size_t rejected_count = 0;
  uint64_t now_nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  size_t idx = 0;
  while (idx < queue_.size()) {
    if (timeout_timestamp_ns_[idx] != 0 &&
        now_nanoseconds > timeout_timestamp_ns_[idx]) {
      rejected_count++;
      rejected_queue_.emplace_back(std::move(queue_[idx]));
      queue_.erase(queue_.begin() + idx);
      timeout_timestamp_ns_.erase(timeout_timestamp_ns_.begin() + idx);
    } else {
      idx++;
    }
  }
  return rejected_count;
}

void
PriorityQueue::PolicyQueue::ReleaseRejectedQueue(
    std::deque<std::unique_ptr<InferenceRequest>>* requests)
{
  rejected_queue_.swap(*requests);
}

void
PriorityQueue::PolicyQueue::ReleaseCancelledQueue(
    std::deque<std::unique_ptr<InferenceRequest>>* requests)
{
  cancelled_queue_.swap(*requests);
}

const std::unique_ptr<InferenceRequest>&
PriorityQueue::PolicyQueue::At(size_t idx) const
{
  if (idx < queue_.size()) {
    return queue_[idx];
  } else {
    return delayed_queue_[idx - queue_.size()];
  }
}

uint64_t
PriorityQueue::PolicyQueue::TimeoutAt(size_t idx)
{
  if (idx < queue_.size()) {
    return timeout_timestamp_ns_[idx];
  } else {
    return 0;
  }
}

bool
PriorityQueue::PolicyQueue::ReadyForErasure()
{
  size_t total_size = Size() + rejected_queue_.size() + cancelled_queue_.size();
  return !keep_instantiated_ && total_size == 0;
}

PriorityQueue::PriorityQueue()
    : size_(0), front_priority_level_(0), default_policy_()
{
  queues_.emplace(0, PolicyQueue(default_policy_, true));
  front_priority_level_ = queues_.begin()->first;
  ResetCursor();
}

PriorityQueue::PriorityQueue(
    const inference::ModelQueuePolicy& default_queue_policy,
    uint64_t priority_levels, const ModelQueuePolicyMap queue_policy_map)
    : size_(0), default_policy_(default_queue_policy)
{
  // Permanently instantiate PolicyQueue with keep_instantiate=true
  // to prevent them from being erased & created during scheduling
  if (priority_levels == 0) {
    // Only default policy is instantiated
    queues_.emplace(0, PolicyQueue(default_policy_, true));
    support_prefetching_ =
        (default_policy_.default_timeout_microseconds() == 0) &&
        (!default_policy_.allow_timeout_override()) &&
        (default_policy_.max_queue_size() == 0);
  } else {
    // All priorities with user-given policy are instantiated. We do not
    // permanently add default PolicyQueue because those will be dynamically
    // created and erased to keep memory footprint low
    for (const auto& qp : queue_policy_map) {
      queues_.emplace(qp.first, PolicyQueue(qp.second, true));
    }
    support_prefetching_ = false;
  }
  front_priority_level_ = queues_.empty() ? 0 : queues_.begin()->first;
  ResetCursor();
}

Status
PriorityQueue::Enqueue(
    uint64_t priority_level, std::unique_ptr<InferenceRequest>& request)
{
  // Get corresponding PolicyQueue if it exists, otherwise insert it
  // via emplace with the default policy
  auto it = queues_.insert(std::make_pair(priority_level, default_policy_));
  auto status = it.first->second.Enqueue(request);
  if (status.IsOk()) {
    size_++;
    front_priority_level_ = std::min(front_priority_level_, priority_level);
    // Invalidate the pending batch cursor if the enqueued item is placed
    // within the pending batch. At the same priority level the request is
    // guaranteed to be after pending batch if the batch hasn't reached
    // delayed queue.
    if (pending_cursor_.valid_ &&
        ((priority_level < pending_cursor_.curr_it_->first) ||
         ((priority_level == pending_cursor_.curr_it_->first) &&
          (pending_cursor_.at_delayed_queue_)))) {
      pending_cursor_.valid_ = false;
    }
  }

  return status;
}

Status
PriorityQueue::Dequeue(std::unique_ptr<InferenceRequest>* request)
{
  pending_cursor_.valid_ = false;
  auto it_start = queues_.lower_bound(front_priority_level_);
  for (auto it = it_start; it != queues_.end(); ++it) {
    if (!it->second.Empty()) {
      front_priority_level_ = it->first;
      RETURN_IF_ERROR(it->second.Dequeue(request));
      size_--;
      if (it->second.ReadyForErasure()) {
        queues_.erase(it);
      }
      return Status::Success;
    }
  }
  return Status(Status::Code::UNAVAILABLE, "dequeue on empty queue");
}

void
PriorityQueue::RejectTimeoutRequests()
{
  for (auto it = queues_.begin(); it != queues_.end(); it++) {
    size_t rejected_count = it->second.RejectTimeoutRequests();
    size_ -= rejected_count;
    if (rejected_count > 0 && it->first == pending_cursor_.curr_it_->first) {
      pending_cursor_.valid_ = false;
    }
  }
}

void
PriorityQueue::ReleaseSkippedRequests(
    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>*
        rejected_requests,
    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>*
        cancelled_requests)
{
  std::vector<std::deque<std::unique_ptr<InferenceRequest>>> reject_req(
      queues_.size());
  std::vector<std::deque<std::unique_ptr<InferenceRequest>>> cancel_req(
      queues_.size());

  size_t idx = 0;
  for (auto it = queues_.begin(); it != queues_.end();) {
    it->second.ReleaseRejectedQueue(&reject_req[idx]);
    it->second.ReleaseCancelledQueue(&cancel_req[idx]);
    idx++;
    if (it->second.ReadyForErasure()) {
      // Invalidate the pending batch cursor if it points to
      // the item to be erased
      if (pending_cursor_.valid_ &&
          it->first == pending_cursor_.curr_it_->first) {
        pending_cursor_.valid_ = false;
      }
      it = queues_.erase(it);  // returns iterator following removed element
    } else {
      ++it;
    }
  }

  rejected_requests->swap(reject_req);
  cancelled_requests->swap(cancel_req);
}

bool
PriorityQueue::IsCursorValid()
{
  if (pending_cursor_.valid_) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
               .count() < pending_cursor_.pending_batch_closest_timeout_ns_;
  }
  return false;
}

PriorityQueue::Cursor::Cursor(PriorityQueues::iterator start_it)
    : curr_it_(start_it), queue_idx_(0), at_delayed_queue_(false),
      pending_batch_closest_timeout_ns_(0),
      pending_batch_oldest_enqueue_time_ns_(0), pending_batch_count_(0),
      valid_(true)
{
}

size_t
PriorityQueue::ApplyPolicyAtCursor()
{
  size_t rejected_batch_size = 0;
  size_t rejected_count = 0;
  size_t cancelled_batch_size = 0;
  size_t cancelled_count = 0;
  while (pending_cursor_.curr_it_ != queues_.end()) {
    if (!(pending_cursor_.curr_it_->second.ApplyPolicy(
            pending_cursor_.queue_idx_, &rejected_count, &rejected_batch_size,
            &cancelled_count, &cancelled_batch_size))) {
      if (size_ > pending_cursor_.pending_batch_count_ + rejected_count +
                      cancelled_count) {
        pending_cursor_.curr_it_++;
        pending_cursor_.queue_idx_ = 0;
        continue;
      }
    }
    // Control reach here if the cursor points to a request that is candidate
    // for pending batch, or if all requests are in pending batch.
    break;
  }
  size_ -= rejected_count + cancelled_count;
  return rejected_batch_size + cancelled_batch_size;
}

void
PriorityQueue::AdvanceCursor()
{
  if (pending_cursor_.pending_batch_count_ >= size_) {
    return;
  }

  const auto& timeout_ns =
      pending_cursor_.curr_it_->second.TimeoutAt(pending_cursor_.queue_idx_);
  if (timeout_ns != 0) {
    if (pending_cursor_.pending_batch_closest_timeout_ns_ != 0) {
      pending_cursor_.pending_batch_closest_timeout_ns_ = std::min(
          pending_cursor_.pending_batch_closest_timeout_ns_, timeout_ns);
    } else {
      pending_cursor_.pending_batch_closest_timeout_ns_ = timeout_ns;
    }
  }

  uint64_t curr_enqueue_time_ns =
      pending_cursor_.curr_it_->second.At(pending_cursor_.queue_idx_)
          ->BatcherStartNs();
  if (pending_cursor_.pending_batch_oldest_enqueue_time_ns_ != 0) {
    pending_cursor_.pending_batch_oldest_enqueue_time_ns_ = std::min(
        pending_cursor_.pending_batch_oldest_enqueue_time_ns_,
        curr_enqueue_time_ns);
  } else {
    pending_cursor_.pending_batch_oldest_enqueue_time_ns_ =
        curr_enqueue_time_ns;
  }
  ++pending_cursor_.queue_idx_;
  ++pending_cursor_.pending_batch_count_;
  // pending batch includes delayed request if (queue_idx_ - 1) points to
  // delayed queue.
  pending_cursor_.at_delayed_queue_ =
      (pending_cursor_.queue_idx_ >
       pending_cursor_.curr_it_->second.UnexpiredSize());
}

}}  // namespace triton::core
