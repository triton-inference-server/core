// Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "backend_model.h"
#include "backend_model_instance.h"
#include "model_config.pb.h"
#include "rate_limiter.h"
#include "scheduler.h"
#include "scheduler_utils.h"
#include "sequence_state.h"
#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

class SequenceBatch;

// Scheduler that implements batching across sequences of correlated
// inferences.
class SequenceBatchScheduler : public Scheduler {
 public:
  using ControlInputs = std::vector<std::shared_ptr<InferenceRequest::Input>>;

  ~SequenceBatchScheduler();

  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  static Status Create(
      TritonModel* model,
      const std::vector<std::shared_ptr<TritonModelInstance>>& new_instances,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      std::unique_ptr<Scheduler>* scheduler);

  // \see Scheduler::Enqueue()
  Status Enqueue(std::unique_ptr<InferenceRequest>& request) override;

  // \see Scheduler::InflightInferenceCount()
  size_t InflightInferenceCount() override
  {
    std::unique_lock<std::mutex> lock(mu_);
    return sequence_to_batcherseqslot_map_.size();
  }

  // \see Scheduler::Stop()
  void Stop() override { stop_ = true; }

  // Update the scheduler to the new set of model instances. This function
  // cannot be called concurrently.
  Status Update(
      const std::vector<std::shared_ptr<TritonModelInstance>>& added_instances,
      const std::vector<std::shared_ptr<TritonModelInstance>>&
          removed_instances);

  // A batcher-sequence_slot combination. The batcher is represented
  // by the index into 'batchers_'.
  struct BatcherSequenceSlot {
    BatcherSequenceSlot() = default;
    BatcherSequenceSlot(const BatcherSequenceSlot&) = default;
    BatcherSequenceSlot(TritonModelInstance* i, uint32_t s)
        : model_instance_(i), seq_slot_(s)
    {
    }
    TritonModelInstance* model_instance_;
    uint32_t seq_slot_;
  };

  // Fill a sequence slot with a sequence from the backlog or show
  // that the sequence slot is no longer being used.
  InferenceRequest::SequenceId ReleaseSequenceSlot(
      const BatcherSequenceSlot& seq_slot,
      std::deque<std::unique_ptr<InferenceRequest>>* requests);

  // For debugging/testing, batcher reports how many waiting requests
  // and returns true if the batcher should continue waiting.
  bool DelayScheduler(
      const TritonModelInstance* model_instance, const size_t cnt,
      const size_t total);

  const std::unordered_map<
      std::string, const inference::ModelSequenceBatching_State&>&
  StateOutputConfigMap()
  {
    return state_output_config_map_;
  }

  size_t MaxBatchSize() { return max_batch_size_; }
  const std::unordered_map<std::string, SequenceStates::InitialStateData>&
  InitialState()
  {
    return initial_state_;
  }

 private:
  SequenceBatchScheduler(
      TritonModel* model,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors)
      : model_(model),
        enforce_equal_shape_tensors_(enforce_equal_shape_tensors), stop_(false)
  {
  }

  void StartBackgroundThreads();
  void StopBackgroundThreads();

  Status CreateBooleanControlTensors(
      const inference::ModelConfig& config,
      std::shared_ptr<ControlInputs>* start_input_overrides,
      std::shared_ptr<ControlInputs>* end_input_overrides,
      std::shared_ptr<ControlInputs>* startend_input_overrides,
      std::shared_ptr<ControlInputs>* continue_input_overrides,
      std::shared_ptr<ControlInputs>* notready_input_overrides);

  Status GenerateInitialStateData(
      const inference::ModelSequenceBatching_InitialState& initial_state,
      const inference::ModelSequenceBatching_State& state, TritonModel* model);

  struct BatcherSequenceSlotCompare {
    bool operator()(
        const BatcherSequenceSlot& a, const BatcherSequenceSlot& b) const
    {
      return a.seq_slot_ > b.seq_slot_;
    }
  };

  // Create a batcher for each of the provided instances.
  Status CreateBatchers(
      const std::vector<std::shared_ptr<TritonModelInstance>>& instances);

  // Move requests so they will be cancelled by the CleanUpThread.
  void MarkRequestsCancelled(
      std::deque<std::unique_ptr<InferenceRequest>>* requests);

  // Erase the sequence slot from 'pending_removal_seq_slots_'. The batcher
  // behind the sequence slot will be removed when all sequence slots of the
  // batcher are removed. Return true if the sequence slot is pending removal.
  // Otherwise, false is returned.
  bool EraseBatcherSequenceSlot(const BatcherSequenceSlot& seq_slot);

  // A thread that monitors idle sequences. This thread is time sensitive that
  // all operations should be completed as quickly as possible to avoid blocking
  // the thread from starting its next iteration.
  void ReaperThread(const int nice);

  // A thread that asynchronously erase removed resources. This thread is
  // intended to destruct resources that might take some time to complete,
  // without preventing the scheduler from scheduling requests.
  void CleanUpThread(const int nice);

  // The 'TritonModel' and 'enforce_equal_shape_tensors' when this scheduler is
  // created.
  TritonModel* model_;
  std::unordered_map<std::string, bool> enforce_equal_shape_tensors_;

  // The number of candidate sequence slots.
  size_t seq_slot_cnt_;

  // The max_sequence_idle_microseconds value for this scheduler.
  uint64_t max_sequence_idle_microseconds_;

  // Whether this scheduler has stopped accepting new inference requests.
  bool stop_;

  // Mutex
  std::mutex mu_;

  // The reaper thread
  std::unique_ptr<std::thread> reaper_thread_;
  std::condition_variable reaper_cv_;
  std::atomic<bool> reaper_thread_exit_;
  // Need to share between enqueue thread and reaper thread because
  // the timeout may be shorten by new request
  uint64_t timeout_timestamp_;

  // The clean-up thread
  std::unique_ptr<std::thread> clean_up_thread_;
  std::condition_variable clean_up_cv_;
  std::atomic<bool> clean_up_thread_exit_;
  // Removed objects to be cleaned up
  std::vector<std::shared_ptr<TritonModelInstance>> removed_instances_;
  std::vector<std::unique_ptr<SequenceBatch>> removed_batchers_;
  std::vector<std::unique_ptr<InferenceRequest>> cancelled_requests_;

  // Map from a model instance pointer that is pending to be removed from this
  // scheduler to a pair ["the number of sequence slots remaining for the
  // instance batcher", "the shared_ptr of the instance"]. The shared_ptr is
  // kept to ensure the instance is available until its sequence slots and
  // batcher are removed from this scheduler.
  std::unordered_map<
      const TritonModelInstance*,
      std::pair<size_t, std::shared_ptr<TritonModelInstance>>>
      pending_removal_seq_slots_;

  // The SequenceBatchs being managed by this scheduler.
  std::unordered_map<const TritonModelInstance*, std::unique_ptr<SequenceBatch>>
      batchers_;

  // Map from a request's correlation ID to the BatcherSequenceSlot
  // assigned to that correlation ID.
  using BatcherSequenceSlotMap =
      std::unordered_map<InferenceRequest::SequenceId, BatcherSequenceSlot>;
  BatcherSequenceSlotMap sequence_to_batcherseqslot_map_;

  // The ordered backlog of sequences waiting for a free sequence slot.
  // The backlog queue keep track of the closest expiration timestamp among
  // the request of the corresponding sequence. Reaper thread will
  // sweep the queues on wake up and clear all timed out sequence.
  // See ReaperThread() for detail implementation.
  struct BacklogQueue {
    // Default to max value so it is not possible to time out unless specified.
    uint64_t expiration_timestamp_{std::numeric_limits<uint64_t>::max()};
    std::shared_ptr<std::deque<std::unique_ptr<InferenceRequest>>> queue_{
        std::make_shared<std::deque<std::unique_ptr<InferenceRequest>>>()};
  };
  std::deque<std::shared_ptr<BacklogQueue>> backlog_queues_;

  // Map from a request's correlation ID to the backlog queue
  // collecting requests for that correlation ID.
  using BacklogMap = std::unordered_map<
      InferenceRequest::SequenceId, std::shared_ptr<BacklogQueue>>;
  BacklogMap sequence_to_backlog_map_;

  // The batcher/sequence-slot locations ready to accept a new
  // sequence. Ordered from lowest sequence-slot-number to highest so
  // that all batchers grow at the same rate and attempt to remain as
  // small as possible.
  std::priority_queue<
      BatcherSequenceSlot, std::vector<BatcherSequenceSlot>,
      BatcherSequenceSlotCompare>
      ready_batcher_seq_slots_;

  // For each correlation ID the most recently seen timestamp, in
  // microseconds, for a request using that correlation ID.
  std::unordered_map<InferenceRequest::SequenceId, uint64_t>
      correlation_id_timestamps_;

  // Used for debugging/testing.
  size_t backlog_delay_cnt_;
  std::unordered_map<const TritonModelInstance*, size_t> queue_request_cnts_;

  // IO mapping between the output state name and the state configuration.
  std::unordered_map<std::string, const inference::ModelSequenceBatching_State&>
      state_output_config_map_;
  size_t max_batch_size_;

  // Initial state used for implicit state.
  std::unordered_map<std::string, SequenceStates::InitialStateData>
      initial_state_;
};

// Base class for a scheduler that implements a particular scheduling
// strategy for a model instance.
class SequenceBatch {
 public:
  SequenceBatch(
      SequenceBatchScheduler* base, TritonModelInstance* model_instance,
      const size_t seq_slot_cnt,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool has_optional_input,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          start_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          end_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          startend_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          continue_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          notready_input_overrides);
  virtual ~SequenceBatch() = default;

  // Enqueue a request into the appropriate queue for the requested
  // sequence slot. This function takes ownership of 'request' so on
  // request 'request' will be nullptr.
  virtual void Enqueue(
      const uint32_t seq_slot,
      const InferenceRequest::SequenceId& correlation_id,
      std::unique_ptr<InferenceRequest>& request) = 0;

  size_t SeqSlotCnt() { return seq_slot_cnt_; }

 protected:
  bool CreateCorrelationIDControl(const inference::ModelConfig& config);
  void SetControlTensors(
      std::unique_ptr<InferenceRequest>& irequest, const int32_t seq_slot,
      const InferenceRequest::SequenceId& corr_id,
      const bool not_ready = false);

  // Update the implicit state and set the required input states.
  void UpdateImplicitState(
      std::unique_ptr<InferenceRequest>& irequest, const int32_t seq_slot);

  // The controlling scheduler.
  SequenceBatchScheduler* const base_;

  // The identifier of this batcher within the controlling scheduler.
  TritonModelInstance* const model_instance_;

  // The number of candidate sequence slots.
  const size_t seq_slot_cnt_;

  // The input tensors that require shape checking before being
  // allowed in a batch. As a map from the tensor name to a bool. If
  // tensor is in map then its shape must match shape of same tensor
  // in requests already in the batch. If value is "true" then
  // additional tensor is treated as a shape tensor and the values
  // contained in the shape tensor must match same tensor already in
  // the batch.
  const std::unordered_map<std::string, bool> enforce_equal_shape_tensors_;

  // Store information on whether the model contains optional inputs.
  bool has_optional_input_;

  // The control values, delivered as input tensors, that should be
  // used when starting a sequence, continuing a sequence, ending a
  // sequence, and showing that a sequence has not input available.
  std::shared_ptr<SequenceBatchScheduler::ControlInputs> start_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs> end_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs>
      startend_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs>
      continue_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs>
      notready_input_overrides_;

  // The correlation ID override. Empty if model does not specify the
  // CONTROL_SEQUENCE_CORRID control.
  std::shared_ptr<InferenceRequest::Input> seq_slot_corrid_override_;

  // For each sequence slot store the optional state i/o tensors.
  std::vector<std::shared_ptr<SequenceStates>> sequence_states_;
};

// Scheduler that implements the Direct sequence scheduling strategy
// for a model instance.
class DirectSequenceBatch : public SequenceBatch {
 public:
  DirectSequenceBatch(
      SequenceBatchScheduler* base, TritonModelInstance* model_instance,
      const size_t seq_slot_cnt,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool has_optional_input,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          start_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          end_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          startend_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          continue_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          notready_input_overrides,
      bool* is_initialized);
  ~DirectSequenceBatch();

  void Enqueue(
      const uint32_t seq_slot,
      const InferenceRequest::SequenceId& correlation_id,
      std::unique_ptr<InferenceRequest>& request) override;

 private:
  void BatcherThread(const int nice);
  void NewPayload();

  std::shared_ptr<Payload> curr_payload_;

  // The thread scheduling requests that are queued in this batch.
  std::unique_ptr<std::thread> scheduler_thread_;
  std::atomic<bool> scheduler_thread_exit_;
  bool scheduler_idle_;

  // Mutex protecting correlation queues, etc.
  std::mutex mu_;
  std::condition_variable cv_;

  // Execution state of the last enqueued payload
  bool exec_complete_;

  // Mutex protecting execution state of payload
  std::mutex payload_mu_;
  std::condition_variable payload_cv_;

  // Queues holding inference requests. There are 'seq_slot_cnt'
  // queues, one for each sequence slot where requests assigned to
  // that slot are enqueued to wait for inferencing.
  std::vector<std::deque<std::unique_ptr<InferenceRequest>>> queues_;
  // Notify when requests are removed from the queue.
  std::condition_variable queues_cv_;

  // Is each sequence slot active or not? A zero or empty value indicates
  // inactive, a non-zero/non-empty value indicates active and is the
  // correlation ID of the sequence active in the slot. An empty
  // queue for a sequence slot does not mean it's inactive... it
  // could just not have any requests pending at the moment.
  std::vector<InferenceRequest::SequenceId> seq_slot_correlation_ids_;

  // The maximum active sequence slot. A value of -1 indicates that
  // no slots are active in the model.
  int32_t max_active_seq_slot_;

  size_t max_batch_size_;
  float minimum_slot_utilization_;
  uint64_t pending_batch_delay_ns_;
};

// Scheduler that implements the oldest-first sequence scheduling
// strategy for a model instance.
class OldestSequenceBatch : public SequenceBatch {
 public:
  OldestSequenceBatch(
      SequenceBatchScheduler* base, TritonModelInstance* model_instance,
      const size_t seq_slot_cnt,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool has_optional_input,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          start_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          end_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          startend_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          continue_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          notready_input_overrides,
      bool* is_initialized);
  ~OldestSequenceBatch();

  void Enqueue(
      const uint32_t seq_slot,
      const InferenceRequest::SequenceId& correlation_id,
      std::unique_ptr<InferenceRequest>& request) override;

 private:
  void CompleteAndNext(const uint32_t seq_slot);

  // The dynamic batcher for this scheduler
  std::unique_ptr<Scheduler> dynamic_batcher_;

  // Mutex protecting queues, etc.
  std::mutex mu_;

  // Condition variable notifying request queueing and/or in-flight completion.
  std::condition_variable cv_;

  // For each sequence slot, true if there is a request for that
  // sequence in-flight in the dynamic batcher. Used to ensure that at
  // most one request from each sequence can be scheduled at a time.
  std::vector<bool> in_flight_;

  // Queues holding inference requests. There are 'seq_slot_cnt'
  // queues, one for each sequence slot where requests assigned to
  // that slot are enqueued to wait for inferencing.
  std::vector<std::deque<std::unique_ptr<InferenceRequest>>> queues_;
};

}}  // namespace triton::core
