// Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sequence_batch_scheduler.h"

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <algorithm>

#include "constants.h"
#include "dynamic_batch_scheduler.h"
#include "model_config_utils.h"
#include "server.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

namespace {

template <typename TimeUnit>
inline uint64_t
Now()
{
  return std::chrono::duration_cast<TimeUnit>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void
SetThreadPriority(const int nice, const char* thread_name)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting " << thread_name << " thread at nice " << nice
                   << "...";
  } else {
    LOG_VERBOSE(1) << "Starting " << thread_name
                   << " thread at default nice (requested nice " << nice
                   << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting " << thread_name << " thread at default nice...";
#endif
}

bool
IsAnyRequestCancelled(std::deque<std::unique_ptr<InferenceRequest>>& requests)
{
  for (auto& req : requests) {
    if (req->IsCancelled()) {
      return true;
    }
  }
  return false;
}

void
CancelRequests(std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  const static Status cancelled_status = Status(Status::Code::CANCELLED);
  for (auto& req : requests) {
    // Mark the request as cancelled before responding.
    Status status = req->Cancel();
    if (!status.IsOk()) {
      LOG_ERROR << status.Message();
    }
    // Respond the request as cancelled.
    InferenceRequest::RespondIfError(req, cancelled_status, true);
  }
}

}  // namespace

Status
SequenceBatchScheduler::Create(
    TritonModel* model,
    const std::vector<std::shared_ptr<TritonModelInstance>>& new_instances,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    std::unique_ptr<Scheduler>* scheduler)
{
  std::unique_ptr<SequenceBatchScheduler> sched(
      new SequenceBatchScheduler(model, enforce_equal_shape_tensors));

  // For debugging and testing,
  const char* dstr = getenv("TRITONSERVER_BACKLOG_DELAY_SCHEDULER");
  sched->backlog_delay_cnt_ = 0;
  if (dstr != nullptr) {
    sched->backlog_delay_cnt_ = atoi(dstr);
    LOG_INFO << "Delaying scheduler until " << sched->backlog_delay_cnt_
             << " backlog queued requests...";
  }

  auto& config = model->Config();

  // Sequencer
  if (config.sequence_batching().iterative_sequence()) {
    sched->sequencer_.reset(new IterativeSequencer(sched.get()));
  } else {
    sched->sequencer_.reset(new Sequencer());
  }

  // Max sequence idle...
  sched->max_sequence_idle_microseconds_ =
      config.sequence_batching().max_sequence_idle_microseconds();

  sched->max_batch_size_ = config.max_batch_size();

  // Implicit States
  auto& states = config.sequence_batching().state();

  for (const inference::ModelSequenceBatching_State& state : states) {
    sched->state_output_config_map_.insert({state.output_name(), state});

    if (state.initial_state_size() > 1) {
      return Status(
          Status::Code::INVALID_ARG,
          std::string(
              std::string("initial_state field for state input '") +
              state.input_name() +
              "' must contain exactly one or zero element. Found '" +
              std::to_string(state.initial_state_size()) + "' elements."));
    }

    // If the model configuration has initial_state field.
    if (state.initial_state_size() == 1) {
      auto& initial_state = state.initial_state(0);
      RETURN_IF_ERROR(
          sched->GenerateInitialStateData(initial_state, state, model));
    }
  }

  // Get the number of candidate sequence slots to allow for each
  // runner. This is at least 1 even if the model doesn't support
  // batching.
  const size_t model_batch_size = std::max(1, config.max_batch_size());
  sched->seq_slot_cnt_ = model_batch_size;
  if (config.sequence_batching().has_oldest()) {
    const auto& max_candidate_seqs =
        config.sequence_batching().oldest().max_candidate_sequences();
    if (max_candidate_seqs > 0) {
      sched->seq_slot_cnt_ = max_candidate_seqs;
    }
  }

  // Create a batcher for each instance.
  RETURN_IF_ERROR(sched->CreateBatchers(new_instances));

  // Create background threads that watch for different sequence states.
  sched->StartBackgroundThreads();

  scheduler->reset(sched.release());

  return Status::Success;
}

Status
SequenceBatchScheduler::Update(
    const std::vector<std::shared_ptr<TritonModelInstance>>& added_instances,
    const std::vector<std::shared_ptr<TritonModelInstance>>& removed_instances)
{
  std::lock_guard<std::mutex> lk(mu_);

  // Add the new batchers.
  RETURN_IF_ERROR(CreateBatchers(added_instances));

  // All sequence slots of the removed instances are pending to be removed.
  // Track them in 'pending_removal_seq_slots_' and they will be erased once
  // they become ready.
  for (auto& instance : removed_instances) {
    pending_removal_seq_slots_.emplace(
        instance.get(),
        std::make_pair(batchers_[instance.get()]->SeqSlotCnt(), instance));
  }

  // Erase pending removal sequence slots that are already at ready.
  std::priority_queue<
      BatcherSequenceSlot, std::vector<BatcherSequenceSlot>,
      BatcherSequenceSlotCompare>
      new_ready_batcher_seq_slots;
  while (!ready_batcher_seq_slots_.empty()) {
    auto& ready_seq_slot = ready_batcher_seq_slots_.top();
    if (pending_removal_seq_slots_.find(ready_seq_slot.model_instance_) ==
        pending_removal_seq_slots_.end()) {
      // The ready slot is not being removed.
      new_ready_batcher_seq_slots.push(ready_seq_slot);
    } else {
      // The ready slot is being removed, erase the slot.
      EraseBatcherSequenceSlot(ready_seq_slot);
    }
    ready_batcher_seq_slots_.pop();
  }
  ready_batcher_seq_slots_.swap(new_ready_batcher_seq_slots);

  return Status::Success;
}

bool
SequenceBatchScheduler::EraseBatcherSequenceSlot(
    const BatcherSequenceSlot& seq_slot)
{
  // Find the sequence slot from pending removal sequence slots.
  auto it = pending_removal_seq_slots_.find(seq_slot.model_instance_);
  if (it == pending_removal_seq_slots_.end()) {
    // The sequence slot is not pending removal.
    return false;
  }

  LOG_VERBOSE(1) << "Removing slot for batcher "
                 << seq_slot.model_instance_->Name() << ", slot "
                 << seq_slot.seq_slot_;

  // Subtract the number of slots, and erase the batcher if no more slots left.
  size_t& remaining_slots = it->second.first;
  if (--remaining_slots == 0) {
    LOG_VERBOSE(1) << "Removing batcher " << seq_slot.model_instance_->Name();

    // Erase batcher.
    auto batcher_it = batchers_.find(seq_slot.model_instance_);
    auto& pending_removal_batcher = batcher_it->second;
    removed_batchers_.emplace_back(std::move(pending_removal_batcher));
    batchers_.erase(batcher_it);

    // Stop tracking the removed instance.
    auto& pending_removal_instance = it->second.second;
    removed_instances_.emplace_back(std::move(pending_removal_instance));

    // Erase from debugging/testing info.
    queue_request_cnts_.erase(seq_slot.model_instance_);

    // Erase sequence slot from pending.
    pending_removal_seq_slots_.erase(it);

    // Notify the clean-up thread.
    clean_up_cv_.notify_one();
  }

  // The sequence slot was pending removal.
  return true;
}

Status
SequenceBatchScheduler::CreateBatchers(
    const std::vector<std::shared_ptr<TritonModelInstance>>& instances)
{
  auto& config = model_->Config();

  // Based on the model configuration create input tensors for control
  // signals indicating sequence start, sequence continue, and
  // sequence not ready.
  std::shared_ptr<ControlInputs> start;
  std::shared_ptr<ControlInputs> end;
  std::shared_ptr<ControlInputs> startend;
  std::shared_ptr<ControlInputs> cont;
  std::shared_ptr<ControlInputs> notready;
  RETURN_IF_ERROR(CreateBooleanControlTensors(
      config, &start, &end, &startend, &cont, &notready));

  bool has_optional_input = false;
  for (const auto& input : config.input()) {
    if (input.optional()) {
      has_optional_input = true;
      break;
    }
  }

  // Create one SequenceBatch object for each requested runner. The
  // SequenceBatch object has a thread that manages the batch of
  // requests.
  for (const auto& instance : instances) {
    bool init_state;
    std::unique_ptr<SequenceBatch> sb;

    // Create the SequenceBatch derivative that handles the requested
    // scheduling strategy.
    if (config.sequence_batching().has_oldest()) {
      sb.reset(new OldestSequenceBatch(
          this, instance.get(), seq_slot_cnt_, enforce_equal_shape_tensors_,
          has_optional_input, start, end, startend, cont, notready,
          &init_state));
    } else {
      sb.reset(new DirectSequenceBatch(
          this, instance.get(), seq_slot_cnt_, enforce_equal_shape_tensors_,
          has_optional_input, start, end, startend, cont, notready,
          &init_state));
    }

    if (init_state) {
      batchers_.emplace(instance.get(), std::move(sb));
      // All sequence slots in the batcher are initially ready for a
      // new sequence.
      for (size_t b = 0; b < seq_slot_cnt_; ++b) {
        ready_batcher_seq_slots_.push(
            SequenceBatchScheduler::BatcherSequenceSlot(instance.get(), b));
      }
    }
  }
  if (batchers_.empty()) {
    return Status(
        Status::Code::INTERNAL,
        "Initialization failed for all sequence-batch scheduler threads");
  }

  return Status::Success;
}

Status
SequenceBatchScheduler::GenerateInitialStateData(
    const inference::ModelSequenceBatching_InitialState& initial_state,
    const inference::ModelSequenceBatching_State& state, TritonModel* model)
{
  if (initial_state.data_type() != state.data_type()) {
    return Status(
        Status::Code::INVALID_ARG,
        std::string("The data type used for 'initial_state' field of state '") +
            state.input_name() + "' does not match the state data type.");
  }

  if (initial_state.name().size() == 0) {
    return Status(
        Status::Code::INVALID_ARG,
        std::string("Field 'name' must be set when using initial_state for "
                    "state input '") +
            state.input_name() + "'.");
  }

  auto initial_state_itr = initial_state_.find(state.input_name());
  if (initial_state_itr != initial_state_.end()) {
    return Status(
        Status::Code::INVALID_ARG, std::string("State input name '") +
                                       state.input_name() +
                                       "' specified more than once.");
  }

  if (initial_state.dims().size() != state.dims().size()) {
    return Status(
        Status::Code::INVALID_ARG,
        std::string(
            "Number of dimensions in 'initial_state' doesn't match the size of"
            " 'state' dimensions for state input '") +
            state.input_name() + "'. " +
            std::to_string(initial_state.dims().size()) +
            " != " + std::to_string(state.dims().size()));
  }
  const auto& initial_state_pair = initial_state_.emplace(
      std::piecewise_construct, std::forward_as_tuple(state.input_name()),
      std::forward_as_tuple(initial_state.name()));
  auto& initial_state_data = initial_state_pair.first->second;

  if (max_batch_size_ > 0) {
    initial_state_data.shape_.emplace_back(1);
  }

  // Check the dimensions to make sure it doesn't have variable-sized dims and
  // matches the state description.
  auto initial_state_dim = initial_state.dims().begin();
  auto state_dim = state.dims().begin();
  for (; initial_state_dim != initial_state.dims().end();
       initial_state_dim++, state_dim++) {
    if (*initial_state_dim == -1) {
      return Status(
          Status::Code::INVALID_ARG,
          std::string("'initial_state' field for state input name '") +
              state.input_name() + "' contains variable dimensions.");
    } else {
      if (*state_dim != -1 && *initial_state_dim != *state_dim) {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("'initial_state' dim for input name '") +
                state.input_name() +
                "' doesn't match 'state' dim description. " +
                std::to_string(*initial_state_dim) +
                " != " + std::to_string(*state_dim));
      }
    }
    initial_state_data.shape_.emplace_back(*initial_state_dim);
  }

  // Calculate total memory byte size
  auto element_count = triton::common::GetElementCount(initial_state.dims());
  size_t dtype_byte_size =
      triton::common::GetDataTypeByteSize(initial_state.data_type());
  size_t total_byte_size = element_count * dtype_byte_size;

  // Custom handling for TYPE_BYTES
  if (dtype_byte_size == 0) {
    total_byte_size = sizeof(int32_t) * element_count;
  }

  switch (initial_state.state_data_case()) {
    case inference::ModelSequenceBatching_InitialState::StateDataCase::
        kZeroData: {
      initial_state_data.data_ = std::make_shared<AllocatedMemory>(
          total_byte_size, TRITONSERVER_MEMORY_CPU /* memory_type */,
          0 /* memory_type_id */);

      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      char* data_ptr = initial_state_data.data_->MutableBuffer(
          &memory_type, &memory_type_id);
      memset(data_ptr, 0, total_byte_size);
      break;
    }
    case inference::ModelSequenceBatching_InitialState::StateDataCase::
        kDataFile: {
      std::string file_input;
      RETURN_IF_ERROR(ReadTextFile(
          JoinPath(
              {model->LocalizedModelPath(), kInitialStateFolder,
               (initial_state.data_file())}),
          &file_input));
      if (initial_state.data_type() == inference::DataType::TYPE_STRING) {
        total_byte_size = file_input.size();
      } else if (total_byte_size > file_input.size()) {
        return Status(
            Status::Code::INVALID_ARG,
            "initial_state setting expects " + std::to_string(total_byte_size) +
                " bytes, but the data "
                "provided from " +
                initial_state.data_file() + "only has " +
                std::to_string(file_input.size()) + " bytes.");
      }

      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;

      initial_state_data.data_ = std::make_shared<AllocatedMemory>(
          total_byte_size, TRITONSERVER_MEMORY_CPU /* memory_type */,
          0 /* memory_type_id */);
      char* data_ptr = initial_state_data.data_->MutableBuffer(
          &memory_type, &memory_type_id);
      memcpy(data_ptr, file_input.data(), total_byte_size);

      break;
    }
    default:
      return Status(
          Status::Code::INVALID_ARG,
          std::string("initial_state setting expects state'") +
              state.input_name() + "' to have state_data set");
  }

  return Status::Success;
}

SequenceBatchScheduler::~SequenceBatchScheduler()
{
  StopBackgroundThreads();

  // Release 'batchers_' before other member variables because 'batchers_'
  // can access 'this' and we need to make sure the member variables live
  // longer than 'batchers_'
  batchers_.clear();
  removed_batchers_.clear();
}


namespace {

Status
GetBooleanOverrideInputs(
    const std::string& tensor_name, const bool support_batching,
    const inference::DataType tensor_datatype, const float fp32_false_value,
    const float fp32_true_value, const int32_t int32_false_value,
    const int32_t int32_true_value, const bool bool_false_value,
    const bool bool_true_value,
    std::shared_ptr<InferenceRequest::Input>* true_override,
    std::shared_ptr<InferenceRequest::Input>* false_override)
{
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  const std::vector<int64_t> tensor_shape{1};
  std::vector<int64_t> tensor_shape_with_batch_dim{1};
  if (support_batching) {
    tensor_shape_with_batch_dim.push_back(1);
  }
  const size_t size_p = triton::common::GetDataTypeByteSize(tensor_datatype);

  auto true_p =
      std::make_shared<AllocatedMemory>(size_p, TRITONSERVER_MEMORY_CPU, 0);
  char* true_p_ptr = true_p->MutableBuffer(&memory_type, &memory_type_id);
  if ((true_p_ptr == nullptr) ||
      ((memory_type != TRITONSERVER_MEMORY_CPU) &&
       (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) ||
      (memory_type_id != 0)) {
    return Status(
        Status::Code::INTERNAL,
        "failed to allocate sequence control signal in CPU memory");
  }

  auto false_p =
      std::make_shared<AllocatedMemory>(size_p, TRITONSERVER_MEMORY_CPU, 0);
  char* false_p_ptr = false_p->MutableBuffer(&memory_type, &memory_type_id);
  if ((false_p_ptr == nullptr) ||
      ((memory_type != TRITONSERVER_MEMORY_CPU) &&
       (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) ||
      (memory_type_id != 0)) {
    return Status(
        Status::Code::INTERNAL,
        "failed to allocate sequence control signal in CPU memory");
  }

  if (tensor_datatype == inference::DataType::TYPE_INT32) {
    *(reinterpret_cast<int32_t*>(true_p_ptr)) = int32_true_value;
    *(reinterpret_cast<int32_t*>(false_p_ptr)) = int32_false_value;
  } else if (tensor_datatype == inference::DataType::TYPE_FP32) {
    *(reinterpret_cast<float*>(true_p_ptr)) = fp32_true_value;
    *(reinterpret_cast<float*>(false_p_ptr)) = fp32_false_value;
  } else {
    *(reinterpret_cast<bool*>(true_p_ptr)) = bool_true_value;
    *(reinterpret_cast<bool*>(false_p_ptr)) = bool_false_value;
  }

  auto ltrue_override = std::make_shared<InferenceRequest::Input>(
      tensor_name, tensor_datatype, tensor_shape);
  *ltrue_override->MutableShape() = ltrue_override->OriginalShape();
  *ltrue_override->MutableShapeWithBatchDim() = tensor_shape_with_batch_dim;
  RETURN_IF_ERROR(ltrue_override->SetData(true_p));

  auto lfalse_override = std::make_shared<InferenceRequest::Input>(
      tensor_name, tensor_datatype, tensor_shape);
  *lfalse_override->MutableShape() = lfalse_override->OriginalShape();
  *lfalse_override->MutableShapeWithBatchDim() = tensor_shape_with_batch_dim;
  RETURN_IF_ERROR(lfalse_override->SetData(false_p));

  *true_override = std::move(ltrue_override);
  *false_override = std::move(lfalse_override);

  return Status::Success;
}

}  // namespace

Status
SequenceBatchScheduler::CreateBooleanControlTensors(
    const inference::ModelConfig& config,
    std::shared_ptr<ControlInputs>* start_input_overrides,
    std::shared_ptr<ControlInputs>* end_input_overrides,
    std::shared_ptr<ControlInputs>* startend_input_overrides,
    std::shared_ptr<ControlInputs>* continue_input_overrides,
    std::shared_ptr<ControlInputs>* notready_input_overrides)
{
  // Currently only batch-size 1 requests are supported so only need
  // to provide control vectors of that size.
  *start_input_overrides = std::make_shared<ControlInputs>();
  *end_input_overrides = std::make_shared<ControlInputs>();
  *startend_input_overrides = std::make_shared<ControlInputs>();
  *continue_input_overrides = std::make_shared<ControlInputs>();
  *notready_input_overrides = std::make_shared<ControlInputs>();

  std::string tensor_name;
  inference::DataType tensor_datatype;
  int32_t int32_false_value, int32_true_value;
  float fp32_false_value, fp32_true_value;
  bool bool_false_value, bool_true_value;

  // START, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value,
        &bool_false_value, &bool_true_value));
    if (!tensor_name.empty()) {
      std::shared_ptr<InferenceRequest::Input> true_override;
      std::shared_ptr<InferenceRequest::Input> false_override;

      RETURN_IF_ERROR(GetBooleanOverrideInputs(
          tensor_name, config.max_batch_size() != 0, tensor_datatype,
          fp32_false_value, fp32_true_value, int32_false_value,
          int32_true_value, bool_false_value, bool_true_value, &true_override,
          &false_override));

      (*start_input_overrides)->emplace_back(true_override);
      (*end_input_overrides)->emplace_back(false_override);
      (*startend_input_overrides)->emplace_back(true_override);
      (*continue_input_overrides)->emplace_back(false_override);
      (*notready_input_overrides)->emplace_back(false_override);
    }
  }

  // END, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value,
        &bool_false_value, &bool_true_value));
    if (!tensor_name.empty()) {
      std::shared_ptr<InferenceRequest::Input> true_override;
      std::shared_ptr<InferenceRequest::Input> false_override;

      RETURN_IF_ERROR(GetBooleanOverrideInputs(
          tensor_name, config.max_batch_size() != 0, tensor_datatype,
          fp32_false_value, fp32_true_value, int32_false_value,
          int32_true_value, bool_false_value, bool_true_value, &true_override,
          &false_override));

      (*start_input_overrides)->emplace_back(false_override);
      (*end_input_overrides)->emplace_back(true_override);
      (*startend_input_overrides)->emplace_back(true_override);
      (*continue_input_overrides)->emplace_back(false_override);
      (*notready_input_overrides)->emplace_back(false_override);
    }
  }

  // READY, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value,
        &bool_false_value, &bool_true_value));
    if (!tensor_name.empty()) {
      std::shared_ptr<InferenceRequest::Input> true_override;
      std::shared_ptr<InferenceRequest::Input> false_override;

      RETURN_IF_ERROR(GetBooleanOverrideInputs(
          tensor_name, config.max_batch_size() != 0, tensor_datatype,
          fp32_false_value, fp32_true_value, int32_false_value,
          int32_true_value, bool_false_value, bool_true_value, &true_override,
          &false_override));

      (*start_input_overrides)->emplace_back(true_override);
      (*end_input_overrides)->emplace_back(true_override);
      (*startend_input_overrides)->emplace_back(true_override);
      (*continue_input_overrides)->emplace_back(true_override);
      (*notready_input_overrides)->emplace_back(false_override);
    }
  }

  return Status::Success;
}

Status
SequenceBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& irequest)
{
  // Queue timer starts at the beginning of the queueing and
  // scheduling process
  irequest->CaptureQueueStartNs();
  INFER_TRACE_ACTIVITY(
      irequest->TraceProxy(), TRITONSERVER_TRACE_QUEUE_START,
      irequest->QueueStartNs());

  // Record time at the beginning of the batcher queueing
  irequest->CaptureBatcherStartNs();

  // For now the request must have batch-size 1 since the sequence
  // batcher does not yet support requests that are statically
  // batched.
  if (irequest->BatchSize() > 1) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request to model '" + irequest->ModelName() +
            "' must specify batch-size 1 due to requirements of sequence "
            "batcher");
  }

  RETURN_IF_ERROR(sequencer_->SetupSequenceRequest(irequest));
  const auto& correlation_id = irequest->CorrelationId();

  BatcherSequenceSlot* target = nullptr;

  const bool seq_start =
      ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) != 0);
  const bool seq_end =
      ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0);

  bool wake_reaper_thread = false;

  std::unique_lock<std::mutex> lock(mu_);

  // Check if the request is one of the in-flight sequence (not starting new
  // sequence), we consider sequences in backlog as also in-flight.
  if (stop_ && seq_start) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Server is stopping, scheduler for model has stopped accepting new "
        "inference requests");
  }

  auto sb_itr = sequence_to_batcherseqslot_map_.find(correlation_id);
  auto bl_itr = sequence_to_backlog_map_.find(correlation_id);

  sequencer_->AddReleaseCallback(
      irequest,
      [this](std::unique_ptr<InferenceRequest>& request, const uint32_t flags)
          -> Status { return sequencer_->RescheduleRequest(request, flags); });

  // If this request is not starting a new sequence its correlation ID
  // should already be known with a target in either a sequence slot
  // or in the backlog. If it doesn't then the sequence wasn't started
  // correctly or there has been a correlation ID conflict. In either
  // case fail this request.
  if (!seq_start && (sb_itr == sequence_to_batcherseqslot_map_.end()) &&
      (bl_itr == sequence_to_backlog_map_.end())) {
    std::string correlation_id_str{""};
    if (correlation_id.Type() ==
        InferenceRequest::SequenceId::DataType::STRING) {
      correlation_id_str = correlation_id.StringValue();
    } else if (
        correlation_id.Type() ==
        InferenceRequest::SequenceId::DataType::UINT64) {
      correlation_id_str = std::to_string(correlation_id.UnsignedIntValue());
    }
    return Status(
        Status::Code::INVALID_ARG,
        "inference request for sequence " + correlation_id_str + " to model '" +
            irequest->ModelName() +
            "' must specify the START flag on the first request of the "
            "sequence");
  }

  // Record the timestamp of this request for the correlation ID. The
  // reaper thread will check to make sure that
  // max_sequence_idle_microseconds value is not exceed for any
  // sequence, and if it is it will release the sequence slot (if any)
  // allocated to that sequence.
  uint64_t now_us = Now<std::chrono::microseconds>();
  correlation_id_timestamps_[correlation_id] = now_us;

  // If this request starts a new sequence but the correlation ID
  // already has an in-progress sequence then that previous sequence
  // did not end correctly, or there is a correlation ID conflict. In
  // this case we continue the new sequence (in either backlog or
  // sequence slot). It is ok for a backlog/slot to have multiple
  // starts... as long as it has a single end. The previous sequence
  // that was not correctly ended will have its existing requests
  // handled and then the new sequence will start.
  if (seq_start && ((sb_itr != sequence_to_batcherseqslot_map_.end()) ||
                    (bl_itr != sequence_to_backlog_map_.end()))) {
    LOG_WARNING
        << "sequence " << correlation_id << " for model '"
        << irequest->ModelName()
        << "' has a conflict. The previous sequence did not end before this "
           "sequence start. Previous sequence will be terminated early.";
  }

  // This request already has an assigned slot...
  if (sb_itr != sequence_to_batcherseqslot_map_.end()) {
    target = &sb_itr->second;
  }
  // This request already has a queue in the backlog...
  else if (bl_itr != sequence_to_backlog_map_.end()) {
    LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id
                   << " into existing backlog: " << irequest->ModelName();

    auto& backlog = bl_itr->second;
    if (irequest->TimeoutMicroseconds() != 0) {
      backlog->expiration_timestamp_ = std::min(
          backlog->expiration_timestamp_,
          now_us + irequest->TimeoutMicroseconds());
      if (backlog->expiration_timestamp_ < timeout_timestamp_) {
        timeout_timestamp_ = backlog->expiration_timestamp_;
        wake_reaper_thread = true;
      }
    }
    backlog->queue_->emplace_back(std::move(irequest));

    // If the sequence is ending then forget correlation ID
    // connection to this backlog queue. If another sequence starts
    // with the same correlation ID it will be collected in another
    // backlog queue.
    if (seq_end) {
      sequence_to_backlog_map_.erase(bl_itr);
    }

    // Waking up reaper so it received latest timeout to be waited for,
    // shouldn't incur actual reaper work.
    if (wake_reaper_thread) {
      reaper_cv_.notify_all();
    }
    return Status::Success;
  }
  // This request does not have an assigned backlog or sequence
  // slot. By the above checks it must be starting. If there is a free
  // sequence slot available then assign this sequence to that slot...
  else if (!ready_batcher_seq_slots_.empty()) {
    target = &sequence_to_batcherseqslot_map_[correlation_id];
    *target = ready_batcher_seq_slots_.top();
    ready_batcher_seq_slots_.pop();
  }
  // Last option is to assign this request to the backlog...
  else {
    LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id
                   << " into new backlog: " << irequest->ModelName();

    auto backlog = std::make_shared<BacklogQueue>();
    if (irequest->TimeoutMicroseconds() != 0) {
      backlog->expiration_timestamp_ = now_us + irequest->TimeoutMicroseconds();
      if (backlog->expiration_timestamp_ < timeout_timestamp_) {
        timeout_timestamp_ = backlog->expiration_timestamp_;
        wake_reaper_thread = true;
      }
    }
    backlog_queues_.push_back(backlog);
    backlog->queue_->emplace_back(std::move(irequest));
    if (!seq_end) {
      sequence_to_backlog_map_[correlation_id] = std::move(backlog);
    }

    // Waking up reaper so it received latest timeout to be waited for,
    // shouldn't incur actual reaper work.
    if (wake_reaper_thread) {
      reaper_cv_.notify_all();
    }
    return Status::Success;
  }

  // Need to grab the target contents before the erase below since
  // that can free it.
  const TritonModelInstance* model_instance = target->model_instance_;
  const uint32_t seq_slot = target->seq_slot_;

  // At this point the request has been assigned to a sequence
  // slot. If the sequence is ending then stop tracking the
  // correlation.
  if (seq_end) {
    sequence_to_batcherseqslot_map_.erase(correlation_id);
  }

  // Enqueue request into batcher and sequence slot.  Don't hold the
  // lock while enqueuing in a specific batcher.
  lock.unlock();

  LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id << " into batcher "
                 << model_instance->Name() << ", sequence slot " << seq_slot
                 << ": " << irequest->ModelName();
  batchers_[model_instance]->Enqueue(seq_slot, correlation_id, irequest);
  return Status::Success;
}

void
SequenceBatchScheduler::MarkRequestsCancelled(
    std::deque<std::unique_ptr<InferenceRequest>>* requests)
{
  bool notify_clean_up = !requests->empty();
  while (!requests->empty()) {
    auto& cancelled_request = requests->front();
    if (cancelled_request) {
      cancelled_requests_.emplace_back(std::move(cancelled_request));
    }
    requests->pop_front();
  }
  if (notify_clean_up) {
    clean_up_cv_.notify_one();
  }
}

InferenceRequest::SequenceId
SequenceBatchScheduler::ReleaseSequenceSlot(
    const BatcherSequenceSlot& batcher_seq_slot,
    std::deque<std::unique_ptr<InferenceRequest>>* requests)
{
  std::unique_lock<std::mutex> lock(mu_);

  // If we are releasing the slot for a cancelled sequence,
  // we have to clean up the sequence
  // otherwise the reaper will try to clean it up again.
  if (!requests->empty() && requests->front()) {
    const InferenceRequest::SequenceId& corr_id =
        requests->front()->CorrelationId();
    LOG_VERBOSE(1) << "Releasing canceled sequence CORRID " << corr_id;

    // Clean up the correlation id to sequence slot mapping, to avoid the reaper
    // from trying to release the same slot again on this instance of the
    // correlation id.
    sequence_to_batcherseqslot_map_.erase(corr_id);
    // Clean up the correlation id to sequence timeout timestamp mapping, to
    // avoid removal of a newer sequence from the backlog upon previous timeout
    // if the same id is re-used by the newer sequence.
    correlation_id_timestamps_.erase(corr_id);
  }

  // If there are any remaining requests on the releasing sequence slot, those
  // requests will be cancelled.
  MarkRequestsCancelled(requests);

  // If the instance behind the slot is pending to be removed, do not add the
  // slot back to ready and erase it instead.
  if (EraseBatcherSequenceSlot(batcher_seq_slot)) {
    // The slot is erased.
    return InferenceRequest::SequenceId();
  }

  // If there is a backlogged sequence and it is requested, return it so that it
  // can use the newly available sequence slot.
  while (!backlog_queues_.empty()) {
    auto backlog = backlog_queues_.front()->queue_;
    backlog_queues_.pop_front();

    if (backlog->empty()) {
      LOG_ERROR << "Should not print this! Unexpected empty backlog.";
      continue;
    }

    const auto& irequest = backlog->back();
    const InferenceRequest::SequenceId& correlation_id =
        irequest->CorrelationId();
    const bool seq_cancelled = IsAnyRequestCancelled(*backlog);

    // If the last queue entry is not an END request then the entire sequence is
    // not contained in the backlog. In that case must update backlog and
    // batcherseqslot maps so that future requests get directed to the batcher
    // sequence-slot instead of the backlog.
    const bool seq_end =
        ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0);
    if (!seq_end) {
      // Since the correlation ID is being actively collected in the backlog,
      // there should not be any in-flight sequences with that same correlation
      // ID that have an assigned slot.
      if (sequence_to_batcherseqslot_map_.find(correlation_id) !=
          sequence_to_batcherseqslot_map_.end()) {
        LOG_ERROR << irequest->LogRequest() << "internal: backlog sequence "
                  << correlation_id
                  << " conflicts with in-flight sequence for model '"
                  << irequest->ModelName() << "'";
      }

      sequence_to_backlog_map_.erase(correlation_id);
      if (!seq_cancelled) {
        sequence_to_batcherseqslot_map_[correlation_id] = batcher_seq_slot;
      }
    }

    if (seq_cancelled) {
      LOG_VERBOSE(1) << irequest->LogRequest() << "CORRID " << correlation_id
                     << " sequence cancelled: " << irequest->ModelName();
      MarkRequestsCancelled(backlog.get());
      continue;
    }
    *requests = std::move(*backlog);

    LOG_VERBOSE(1) << irequest->LogRequest() << "CORRID " << correlation_id
                   << " reusing batcher "
                   << batcher_seq_slot.model_instance_->Name() << ", slot "
                   << batcher_seq_slot.seq_slot_ << ": "
                   << irequest->ModelName();
    return correlation_id;
  }

  // There is no backlogged sequence so just release the batch slot
  LOG_VERBOSE(1) << "Freeing slot in batcher "
                 << batcher_seq_slot.model_instance_->Name() << ", slot "
                 << batcher_seq_slot.seq_slot_;

  ready_batcher_seq_slots_.push(batcher_seq_slot);
  return InferenceRequest::SequenceId();
}

bool
SequenceBatchScheduler::DelayScheduler(
    const TritonModelInstance* model_instance, const size_t cnt,
    const size_t total)
{
  std::unique_lock<std::mutex> lock(mu_);
  queue_request_cnts_[model_instance] = cnt;

  size_t seen = 0;
  for (auto c : queue_request_cnts_) {
    seen += c.second;
  }

  if (seen < total) {
    return true;
  }

  if (backlog_delay_cnt_ > 0) {
    size_t backlog_seen = 0;
    for (const auto& q : backlog_queues_) {
      backlog_seen += q->queue_->size();
    }

    if (backlog_seen < backlog_delay_cnt_) {
      return true;
    }
  }

  return false;
}

void
SequenceBatchScheduler::StartBackgroundThreads()
{
  // Create a reaper thread that watches for idle sequences.
  reaper_thread_exit_ = false;
  reaper_thread_.reset(
      new std::thread([this]() { ReaperThread(10 /* nice */); }));

  // Create a clean-up thread that asynchronously erase removed resources.
  clean_up_thread_exit_ = false;
  clean_up_thread_.reset(
      new std::thread([this]() { CleanUpThread(20 /* nice */); }));
}

void
SequenceBatchScheduler::StopBackgroundThreads()
{
  // Exit the clean-up thread.
  clean_up_thread_exit_ = true;
  clean_up_cv_.notify_one();
  if (clean_up_thread_ && clean_up_thread_->joinable()) {
    clean_up_thread_->join();
  }

  // Exit the reaper thread.
  reaper_thread_exit_ = true;
  reaper_cv_.notify_one();
  if (reaper_thread_ && reaper_thread_->joinable()) {
    reaper_thread_->join();
  }
}

void
SequenceBatchScheduler::ReaperThread(const int nice)
{
  SetThreadPriority(nice, "sequence-batch reaper" /* thread_name */);

  const uint64_t backlog_idle_wait_microseconds = 50 * 1000;

  uint64_t idle_timestamp =
      Now<std::chrono::microseconds>() + max_sequence_idle_microseconds_;
  timeout_timestamp_ = std::numeric_limits<uint64_t>::max();

  while (!reaper_thread_exit_) {
    uint64_t now_us = Now<std::chrono::microseconds>();

    // Reap idle assigned sequence
    if (now_us >= idle_timestamp) {
      uint64_t wait_microseconds = max_sequence_idle_microseconds_;
      BatcherSequenceSlotMap force_end_sequences;
      {
        std::unique_lock<std::mutex> lock(mu_);
        for (auto cid_itr = correlation_id_timestamps_.cbegin();
             cid_itr != correlation_id_timestamps_.cend();) {
          int64_t remaining_microseconds =
              (int64_t)max_sequence_idle_microseconds_ -
              (now_us - cid_itr->second);
          if (remaining_microseconds > 0) {
            wait_microseconds = std::min(
                wait_microseconds, (uint64_t)remaining_microseconds + 1);
            ++cid_itr;
            continue;
          }

          const InferenceRequest::SequenceId& idle_correlation_id =
              cid_itr->first;
          LOG_VERBOSE(1) << "Reaper: CORRID " << idle_correlation_id
                         << ": max sequence idle exceeded";

          auto idle_sb_itr =
              sequence_to_batcherseqslot_map_.find(idle_correlation_id);

          // If the idle correlation ID has an assigned sequence slot,
          // then release that assignment so it becomes available for
          // another sequence. Release is done by enqueuing and must be
          // done outside the lock, so just collect needed info here.
          if (idle_sb_itr != sequence_to_batcherseqslot_map_.end()) {
            force_end_sequences[idle_correlation_id] = idle_sb_itr->second;

            sequence_to_batcherseqslot_map_.erase(idle_correlation_id);
            cid_itr = correlation_id_timestamps_.erase(cid_itr);
          } else {
            // If the idle correlation ID is in the backlog, then just
            // need to increase the timeout so that we revisit it again in
            // the future to check if it is assigned to a sequence slot.
            auto idle_bl_itr =
                sequence_to_backlog_map_.find(idle_correlation_id);
            if (idle_bl_itr != sequence_to_backlog_map_.end()) {
              LOG_VERBOSE(1)
                  << "Reaper: found idle CORRID " << idle_correlation_id;
              wait_microseconds =
                  std::min(wait_microseconds, backlog_idle_wait_microseconds);
              ++cid_itr;
            } else {
              LOG_VERBOSE(1) << "Reaper: ignoring stale idle CORRID "
                             << idle_correlation_id;
              cid_itr = correlation_id_timestamps_.erase(cid_itr);
            }
          }
        }
      }

      // Enqueue force-ends outside of the lock.
      for (const auto& pr : force_end_sequences) {
        const InferenceRequest::SequenceId& idle_correlation_id = pr.first;
        const TritonModelInstance* model_instance = pr.second.model_instance_;
        const uint32_t seq_slot = pr.second.seq_slot_;

        LOG_VERBOSE(1) << "Reaper: force-ending CORRID " << idle_correlation_id
                       << " in batcher " << model_instance->Name() << ", slot "
                       << seq_slot;

        // A slot assignment is released by enqueuing a request with a
        // null request. The scheduler thread will interpret the null
        // request as meaning it should release the sequence slot but
        // otherwise do nothing with the request.
        std::unique_ptr<InferenceRequest> null_request;
        batchers_[model_instance]->Enqueue(
            seq_slot, idle_correlation_id, null_request);
      }

      // Update timestamp for next idle check
      idle_timestamp = now_us + wait_microseconds;
    }

    // Reap timed out backlog sequence
    if (now_us >= timeout_timestamp_) {
      timeout_timestamp_ = std::numeric_limits<uint64_t>::max();
      std::deque<std::shared_ptr<BacklogQueue>> expired_backlogs;
      {
        std::unique_lock<std::mutex> lock(mu_);
        // Remove expired backlog from 'backlog_queues_'
        auto it = backlog_queues_.begin();
        while (it != backlog_queues_.end()) {
          const auto queue_timestamp = (*it)->expiration_timestamp_;
          if (queue_timestamp > now_us) {
            timeout_timestamp_ = std::min(timeout_timestamp_, queue_timestamp);
            ++it;
          } else {
            // The queue expired, clear the records and reject the request
            // outside lock
            const auto& correlation_id =
                (*it)->queue_->front()->CorrelationId();
            expired_backlogs.emplace_back(std::move(*it));

            // Need to double check on 'sequence_to_backlog_map_', it may
            // be tracking a new sequence with the same ID which may not be
            // timing out.
            const auto& mit = sequence_to_backlog_map_.find(correlation_id);
            if ((mit != sequence_to_backlog_map_.end()) &&
                (mit->second->expiration_timestamp_ <= now_us)) {
              sequence_to_backlog_map_.erase(mit);
            }

            it = backlog_queues_.erase(it);
          }
        }
      }

      // Reject timeout requests
      const static Status rejected_status = Status(
          Status::Code::UNAVAILABLE,
          "timeout of the corresponding sequence has been expired");
      for (auto& backlog : expired_backlogs) {
        for (auto& req : *backlog->queue_) {
          InferenceRequest::RespondIfError(req, rejected_status, true);
        }
      }
    }

    const auto wait_microseconds =
        std::min(idle_timestamp, timeout_timestamp_) - now_us;
    // Wait until the next timeout needs to be checked
    if (wait_microseconds > 0) {
      std::unique_lock<std::mutex> lock(mu_);
      LOG_VERBOSE(2) << "Reaper: sleeping for " << wait_microseconds << "us...";
      std::chrono::microseconds wait_timeout(wait_microseconds);
      reaper_cv_.wait_for(lock, wait_timeout);
    }
  }

  LOG_VERBOSE(1) << "Stopping sequence-batch reaper thread...";
}

void
SequenceBatchScheduler::CleanUpThread(const int nice)
{
  SetThreadPriority(nice, "sequence-batch clean-up" /* thread_name */);

  while (!clean_up_thread_exit_) {
    // Removed resources should be destructed outside the lock.
    std::vector<std::shared_ptr<TritonModelInstance>> removed_instances;
    std::vector<std::unique_ptr<SequenceBatch>> removed_batchers;
    std::vector<std::unique_ptr<InferenceRequest>> cancelled_requests;

    {
      std::unique_lock<std::mutex> lk(mu_);

      clean_up_cv_.wait(lk, [this] {
        return clean_up_thread_exit_ || !removed_instances_.empty() ||
               !removed_batchers_.empty() || !cancelled_requests_.empty();
      });

      removed_instances_.swap(removed_instances);
      removed_batchers_.swap(removed_batchers);
      cancelled_requests_.swap(cancelled_requests);
    }

    LOG_VERBOSE(2)
        << "Cleaning-up resources on sequence-batch clean-up thread...";

    CancelRequests(std::move(cancelled_requests));
  }

  LOG_VERBOSE(1) << "Stopping sequence-batch clean-up thread...";
}

SequenceBatch::SequenceBatch(
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
        notready_input_overrides)
    : base_(base), model_instance_(model_instance), seq_slot_cnt_(seq_slot_cnt),
      enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
      has_optional_input_(has_optional_input),
      start_input_overrides_(start_input_overrides),
      end_input_overrides_(end_input_overrides),
      startend_input_overrides_(startend_input_overrides),
      continue_input_overrides_(continue_input_overrides),
      notready_input_overrides_(notready_input_overrides),
      sequence_states_(seq_slot_cnt)
{
}

bool
SequenceBatch::CreateCorrelationIDControl(const inference::ModelConfig& config)
{
  // If model wants CORRID control then get the name of the input
  // tensor and initialize the override structure for each sequence
  // slot that is used to communicate the correlation ID.
  std::string correlation_id_tensor_name;
  inference::DataType correlation_id_datatype;
  Status corrid_status = GetTypedSequenceControlProperties(
      config.sequence_batching(), config.name(),
      inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID,
      false /* required */, &correlation_id_tensor_name,
      &correlation_id_datatype);
  if (!corrid_status.IsOk()) {
    LOG_ERROR << "failed validating CORRID control for sequence-batch "
                 "scheduler thread "
              << model_instance_->Name() << ": " << corrid_status.Message();
    return false;
  }

  if (!correlation_id_tensor_name.empty()) {
    if ((correlation_id_datatype != inference::DataType::TYPE_UINT64) &&
        (correlation_id_datatype != inference::DataType::TYPE_INT64) &&
        (correlation_id_datatype != inference::DataType::TYPE_UINT32) &&
        (correlation_id_datatype != inference::DataType::TYPE_INT32) &&
        (correlation_id_datatype != inference::DataType::TYPE_STRING)) {
      LOG_ERROR << "unexpected control data type, expected TYPE_UINT64, "
                   "TYPE_INT64, TYPE_UINT32, TYPE_INT32, or TYPE_STRING for "
                << inference::ModelSequenceBatching_Control_Kind_Name(
                       inference::ModelSequenceBatching::Control::
                           CONTROL_SEQUENCE_CORRID)
                << " for " << config.name();
      return false;
    }

    const std::vector<int64_t> tensor_shape{1};
    std::vector<int64_t> tensor_shape_with_batch_dim{1};
    if (config.max_batch_size() != 0) {
      tensor_shape_with_batch_dim.push_back(1);
    }

    auto override = std::make_shared<InferenceRequest::Input>(
        correlation_id_tensor_name, correlation_id_datatype, tensor_shape);
    *override->MutableShape() = override->OriginalShape();
    *override->MutableShapeWithBatchDim() = tensor_shape_with_batch_dim;

    seq_slot_corrid_override_ = std::move(override);
  }

  return true;
}

void
SequenceBatch::SetControlTensors(
    std::unique_ptr<InferenceRequest>& irequest, const int32_t seq_slot,
    const InferenceRequest::SequenceId& corrid, const bool not_ready)
{
  const SequenceBatchScheduler::ControlInputs* controls;

  // Set the start, end, and ready control tensors appropriately...
  if (not_ready) {
    controls = notready_input_overrides_.get();
  } else if (
      (irequest->Flags() & (TRITONSERVER_REQUEST_FLAG_SEQUENCE_START |
                            TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)) ==
      (TRITONSERVER_REQUEST_FLAG_SEQUENCE_START |
       TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)) {
    controls = startend_input_overrides_.get();
  } else if (
      (irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) != 0) {
    controls = start_input_overrides_.get();
  } else if (
      (irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0) {
    controls = end_input_overrides_.get();
  } else {
    controls = continue_input_overrides_.get();
  }

  for (const auto& control : *controls) {
    irequest->AddOverrideInput(control);
  }

  // Set correlation ID control tensor if requested by the model.
  if (seq_slot_corrid_override_ != nullptr) {
    auto& seq_corr_id = seq_slot_corrid_override_;
    size_t size_p = triton::common::GetDataTypeByteSize(seq_corr_id->DType());
    if (seq_corr_id->DType() == inference::DataType::TYPE_STRING) {
      // 4 bytes for length of string plus length of the corrid string
      // length in bytes
      std::string correlation_id = corrid.StringValue();
      uint32_t correlation_id_length = correlation_id.length();
      size_p = 4 + correlation_id_length;
    }

    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    auto corrid_p =
        std::make_shared<AllocatedMemory>(size_p, TRITONSERVER_MEMORY_CPU, 0);
    char* corrid_p_ptr = corrid_p->MutableBuffer(&memory_type, &memory_type_id);
    if ((corrid_p_ptr == nullptr) ||
        ((memory_type != TRITONSERVER_MEMORY_CPU) &&
         (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) ||
        (memory_type_id != 0)) {
      LOG_ERROR << "failed to allocate sequence CORRID control signal in CPU "
                   "memory";
      return;
    }

    auto override = std::make_shared<InferenceRequest::Input>(
        seq_corr_id->Name(), seq_corr_id->DType(), seq_corr_id->Shape());
    *override->MutableShape() = override->OriginalShape();
    *override->MutableShapeWithBatchDim() = seq_corr_id->ShapeWithBatchDim();
    Status corrid_status = override->SetData(corrid_p);
    if (!corrid_status.IsOk()) {
      LOG_ERROR << "failed creating CORRID control for sequence-batch "
                   "scheduler thread "
                << model_instance_->Name() << " for " << seq_corr_id->Name();
      return;
    }

    if (corrid.Type() == InferenceRequest::SequenceId::DataType::STRING) {
      std::string correlation_id = corrid.StringValue();
      uint32_t correlation_id_length = correlation_id.length();
      memcpy(corrid_p_ptr, &correlation_id_length, sizeof(uint32_t));
      memcpy(
          corrid_p_ptr + sizeof(uint32_t), correlation_id.c_str(),
          correlation_id_length);
    } else if (
        corrid.Type() == InferenceRequest::SequenceId::DataType::UINT64) {
      uint64_t correlation_id = corrid.UnsignedIntValue();
      const char* corrid_ptr = reinterpret_cast<const char*>(&correlation_id);
      memcpy(corrid_p_ptr, corrid_ptr, size_p);
    }
    irequest->AddOverrideInput(override);
  }
}

void
SequenceBatch::UpdateImplicitState(
    std::unique_ptr<InferenceRequest>& irequest, const int32_t seq_slot)
{
  // This should be executed only if the model has a states section.
  if (!base_->StateOutputConfigMap().empty()) {
    auto& sequence_states = sequence_states_[seq_slot];

    // Initialize the input state if the sequence is starting.
    if ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) != 0) {
      sequence_states = nullptr;
    }

    // Create the state for the first request in the sequence.
    if (sequence_states == nullptr) {
      sequence_states.reset(new SequenceStates);
      Status status = sequence_states->Initialize(
          base_->StateOutputConfigMap(), base_->MaxBatchSize(),
          base_->InitialState(), model_instance_->Kind(),
          model_instance_->DeviceId(),
          model_instance_->Model()->Server()->CudaVirtualAddressSpaceSize());

      if (!status.IsOk()) {
        LOG_ERROR << "Failed to initialize sequence state: "
                  << status.Message();
      }
    }

    irequest->SetSequenceStates(sequence_states);
  }
}

DirectSequenceBatch::DirectSequenceBatch(
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
    bool* is_initialized)
    : SequenceBatch(
          base, model_instance, seq_slot_cnt, enforce_equal_shape_tensors,
          has_optional_input, start_input_overrides, end_input_overrides,
          startend_input_overrides, continue_input_overrides,
          notready_input_overrides),
      scheduler_thread_exit_(false), scheduler_idle_(false),
      queues_(seq_slot_cnt), seq_slot_correlation_ids_(seq_slot_cnt, 0),
      max_active_seq_slot_(-1)
{
  // Initialize to handle CORRID control. If error just exit
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  const auto& config = model_instance_->Model()->Config();
  if (!CreateCorrelationIDControl(config)) {
    *is_initialized = false;
    return;
  }

  max_batch_size_ = ((size_t)std::max(1, config.max_batch_size()));
  minimum_slot_utilization_ =
      config.sequence_batching().direct().minimum_slot_utilization();
  pending_batch_delay_ns_ =
      config.sequence_batching().direct().max_queue_delay_microseconds() * 1000;

  // Create a scheduler thread associated with 'batcher_idx' that
  // executes the queued requests.
  const int nice = 0;
  NewPayload();
  scheduler_thread_.reset(
      new std::thread([this, nice]() { BatcherThread(nice); }));

  *is_initialized = true;
}

DirectSequenceBatch::~DirectSequenceBatch()
{
  // Wait until all queued requests begin execution.
  {
    std::unique_lock<std::mutex> lk(mu_);
    queues_cv_.wait(lk, [this] {
      for (uint32_t seq_slot = 0; seq_slot < queues_.size(); seq_slot++) {
        if (!queues_[seq_slot].empty()) {
          LOG_VERBOSE(1) << "Waiting for slot " << seq_slot
                         << " to begin execution before exiting";
          return false;
        }
      }
      return true;
    });
  }

  // Wait until the last enqueued payload completes execution.
  {
    std::unique_lock<std::mutex> lk(payload_mu_);
    payload_cv_.wait(lk, [this] {
      if (!exec_complete_ || curr_payload_->RequestCount() > 0) {
        LOG_VERBOSE(1) << "Waiting for current payload to complete execution "
                          "before exiting";
        return false;
      }
      return true;
    });
  }

  // Signal the scheduler thread to exit.
  scheduler_thread_exit_ = true;
  cv_.notify_one();

  // It is possible for the scheduler thread to be the last holder of
  // a model object, and when that scheduler thread releases the
  // object the scheduler thread itself will destroy this
  // SequenceBatch object. So we need to check to make sure the
  // scheduler thread does not join it against itself and instead
  // detach it so there is not a problem when its thread object is
  // destroyed.
  if (scheduler_thread_->joinable()) {
    scheduler_thread_->join();
  }
}

void
DirectSequenceBatch::Enqueue(
    const uint32_t seq_slot, const InferenceRequest::SequenceId& correlation_id,
    std::unique_ptr<InferenceRequest>& request)
{
  bool wake_runner = false;

  {
    std::lock_guard<std::mutex> lock(mu_);

    queues_[seq_slot].emplace_back(std::move(request));

    seq_slot_correlation_ids_[seq_slot] = correlation_id;
    max_active_seq_slot_ =
        std::max(max_active_seq_slot_, static_cast<int32_t>(seq_slot));

    // If runner is idle then wake it to service this request. We do
    // the actual wake outside of the lock to avoid having the woken
    // thread immediately block on the lock
    wake_runner = scheduler_idle_;
  }

  if (wake_runner) {
    cv_.notify_one();
  }
}

void
DirectSequenceBatch::NewPayload()
{
  curr_payload_ =
      model_instance_->Model()->Server()->GetRateLimiter()->GetPayload(
          Payload::Operation::INFER_RUN, model_instance_);
}

void
DirectSequenceBatch::BatcherThread(const int nice)
{
  SetThreadPriority(nice, "Direct sequence-batch scheduler" /* thread_name */);

  // For debugging and testing, delay start of thread until queues
  // contain the specified number of entries (across all
  // SequenceBatchs in the scheduler).
  const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_VERBOSE(1) << "Delaying scheduler thread " << model_instance_->Name()
                   << " until " << delay_cnt << " queued requests...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;
  exec_complete_ = true;

  // When there is optional input or input shape must be enforced,
  // the inputs in the requests must be examined for forming a batch
  const bool check_input =
      !enforce_equal_shape_tensors_.empty() || has_optional_input_;
  while (!scheduler_thread_exit_) {
    uint64_t wait_microseconds = default_wait_microseconds;

    // Wait till execution of the last enqueued payload is
    // complete.
    {
      std::unique_lock<std::mutex> lk(payload_mu_);
      payload_cv_.wait(lk, [this] { return exec_complete_; });
    }

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);

      if (delay_cnt > 0) {
        wait_microseconds = 10 * 1000;
        // Debugging/testing... wait until queues together contain at
        // least 'delay_cnt' items...
        size_t total_size = 0;
        for (const auto& q : queues_) {
          total_size += q.size();
        }
        if (!base_->DelayScheduler(model_instance_, total_size, delay_cnt)) {
          delay_cnt = 0;
        }
        LOG_VERBOSE(1) << "Delaying scheduler thread "
                       << model_instance_->Name() << " until " << delay_cnt
                       << " queued requests, current total = " << total_size;
      } else {
        RequiredEqualInputs required_equal_inputs;
        InferenceRequest* null_irequest = nullptr;

        // Make one pass through the active slots to:
        //
        //   1) release any slots that have cancelled or forcibly ended
        //      sequences
        //
        //   2) find a representative request that will provide:
        //
        //      a) the shape, type, etc. information for null requests
        //
        //      b) the required tensor shapes for the batch for the
        //      case where ragged batching is not allowed
        //
        //   3) Determine the earliest enqueue time and number of ready
        //      sequences if queue delay is enabled
        //
        int32_t max_seq_slot = -1;
        uint64_t earliest_enqueue_time_ns = UINT64_MAX;
        size_t ready_cnt = 0;
        for (int32_t seq_slot = 0; seq_slot <= max_active_seq_slot_;
             ++seq_slot) {
          std::deque<std::unique_ptr<InferenceRequest>>& queue =
              queues_[seq_slot];
          if (!queue.empty()) {
            bool release_seq_slot = false;

            // If the request is nullptr then the sequence in the slot
            // has timed-out so release the slot for another sequence
            // from the backlog.
            if (queue.front() == nullptr) {
              queue.pop_front();
              release_seq_slot = true;
            }
            // If the request is cancelled, then the sequence in the slot is
            // cancelled, so release the slot and cancel all queued requests of
            // the sequence.
            else if (queue.front()->IsCancelled()) {
              release_seq_slot = true;
            }

            if (release_seq_slot) {
              SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
                  model_instance_, seq_slot);
              seq_slot_correlation_ids_[seq_slot] =
                  base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
            }
          }

          // Need to check queue again for contents since if released
          // above it may now be empty...
          if (!queue.empty()) {
            // For NULL requests need an InferenceRequest that can be
            // batched but has controls set to "not ready". Any
            // request can serve this purpose so grab a copy of the
            // first one. This first request is also used to
            // initialize 'required_equal_inputs' so we are sure that
            // this null request will have the correct shape for any
            // created batch.
            if (null_irequest == nullptr) {
              null_irequest = queue.front().get();
              UpdateImplicitState(queue.front(), seq_slot);
            }

            // If this is the first non-null request capture the shape
            // of the tensors that don't support ragged so we can
            // compare them to later requests.
            if (!required_equal_inputs.Initialized() && check_input) {
              Status status = required_equal_inputs.Initialize(
                  queue.front(), enforce_equal_shape_tensors_,
                  has_optional_input_);
              if (!status.IsOk()) {
                LOG_ERROR
                    << "internal: unexpecting failure initializing shape: "
                    << status.Message();
              }
            }

            earliest_enqueue_time_ns = std::min(
                earliest_enqueue_time_ns, queue.front()->BatcherStartNs());
            ready_cnt++;
            max_seq_slot = seq_slot;
          }
        }

        if (max_seq_slot != -1) {
          if ((pending_batch_delay_ns_ == 0) ||
              (minimum_slot_utilization_ == 0.0)) {
            wait_microseconds = 0;
          } else {
            // Compare the age of the oldest pending request to the maximum
            // batch queuing delay, and the size of the ready requests in the
            // batch, execute now if queuing delay is exceeded or the batch
            // size is large enough. Otherwise create a timer to wakeup a
            // thread to check again at the maximum allowed delay.
            uint64_t now_ns = Now<std::chrono::nanoseconds>();
            uint64_t current_batch_delay_ns =
                (now_ns - earliest_enqueue_time_ns);
            if ((current_batch_delay_ns > pending_batch_delay_ns_) ||
                (((float)ready_cnt) / max_batch_size_ >=
                 minimum_slot_utilization_)) {
              wait_microseconds = 0;
              LOG_VERBOSE(1)
                  << "start sequence batch execution. "
                  << "current batch delay: " << current_batch_delay_ns
                  << "; maximum delay allowed: " << pending_batch_delay_ns_
                  << "slot utilization: " << ready_cnt << "/" << max_batch_size_
                  << "; utilization threshold: " << minimum_slot_utilization_;
            } else {
              wait_microseconds =
                  (pending_batch_delay_ns_ - current_batch_delay_ns) / 1000;
              // reset 'max_seq_slot' so that not request is pulled from the
              // queues
              max_seq_slot = -1;
              LOG_VERBOSE(1)
                  << "defer sequence batch execution. "
                  << "current batch delay: " << current_batch_delay_ns
                  << "; maximum delay allowed: " << pending_batch_delay_ns_
                  << "slot utilization: " << ready_cnt << "/" << max_batch_size_
                  << "; utilization threshold: " << minimum_slot_utilization_;
            }
          }
        }

        // Collect requests from slot 0 to max_seq_slot.
        for (int32_t seq_slot = 0; seq_slot <= max_seq_slot; ++seq_slot) {
          bool end_of_sequence = false;
          bool use_null_request = false;
          std::deque<std::unique_ptr<InferenceRequest>>& queue =
              queues_[seq_slot];

          // If 'seq_slot' doesn't have any requests then change the
          // request to send dummy/null input tensors for this
          // slot. We need this so that other requests stay in the
          // correct slot.
          if (queue.empty()) {
            use_null_request = true;
          }
          // If there are one or more tensors that don't support
          // ragged batch, then don't allow a request into an existing
          // batch if shape differs.
          else if (required_equal_inputs.Initialized() && check_input) {
            if (!required_equal_inputs.HasEqualInputs(queue.front())) {
              use_null_request = true;
            }
          }

          // Use null-request if necessary otherwise use the next
          // request in the queue...
          if (use_null_request) {
            std::unique_ptr<InferenceRequest> ni(
                InferenceRequest::CopyAsNull(*null_irequest));
            // Note that when the not-ready control input of the
            // request is "true" the model can't assume that any
            // other inputs are meaningful, including CORRID. So we
            // just use zero for that.
            SetControlTensors(
                ni, seq_slot, 0 /* corrid */, true /* not_ready */);

            // This should be executed only if the model has a states section.
            if (!base_->StateOutputConfigMap().empty()) {
              // For NULL requests we will be using a dummy state instead of the
              // real state stored in Triton. When the model is using variable
              // dimensions and batching, the null request's input state shapes
              // may be different from the actual shapes of the state for that
              // sequence. We create a dummy state in order to avoid corrupting
              // the actual state of the sequence.
              std::shared_ptr<SequenceStates> sequence_states(
                  new SequenceStates);
              sequence_states->SetNullSequenceStates(
                  null_irequest->GetSequenceStates());
              ni->SetSequenceStates(sequence_states);
            }

            curr_payload_->AddRequest(std::move(ni));
          } else {
            std::unique_ptr<InferenceRequest>& irequest = queue.front();

            // Set the control tensor values in the request.
            SetControlTensors(irequest, seq_slot, irequest->CorrelationId());

            // Update the implicit state and set the input state tensors.
            UpdateImplicitState(irequest, seq_slot);

            if ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) !=
                0) {
              end_of_sequence = true;
            }
            curr_payload_->AddRequest(std::move(irequest));

            queue.pop_front();
          }

          if (curr_payload_->GetState() == Payload::State::UNINITIALIZED) {
            curr_payload_->SetState(Payload::State::READY);
          }

          // If the sequence has ended then attempt to refill the
          // sequence slot with a sequence from the backlog. If
          // there is no backlog show that the slot is no longer
          // active.
          if (end_of_sequence) {
            LOG_VERBOSE(1) << "End sequence CORRID "
                           << seq_slot_correlation_ids_[seq_slot]
                           << " in batcher " << model_instance_->Name()
                           << ", slot " << seq_slot;

            // Should never be anything in a queue after the END
            // marker. If it happens that means we will clobber
            // that request if/when we swap in a backlog sequence
            // in ReleaseSequenceSlot below.
            if (!queue.empty()) {
              LOG_ERROR << "internal: unexpected requests after sequence "
                           "end in slot "
                        << seq_slot;
            }

            SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
                model_instance_, seq_slot);
            seq_slot_correlation_ids_[seq_slot] =
                base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
          }
        }
      }

      // One or more sequences may have ended... find the new
      // 'max_active_seq_slot_'.
      while ((max_active_seq_slot_ >= 0) &&
             (!seq_slot_correlation_ids_[max_active_seq_slot_].InSequence())) {
        max_active_seq_slot_--;
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queues again.
      if (wait_microseconds > 0) {
        scheduler_idle_ = true;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        scheduler_idle_ = false;
      }
    }

    // Requests could be removed from the queue, so notify a thread waiting for
    // requests removal from the queue.
    queues_cv_.notify_one();

    if (curr_payload_->GetState() == Payload::State::READY) {
      // Add callback to signal the execution completion
      exec_complete_ = false;
      auto callback = [this]() {
        {
          std::unique_lock<std::mutex> lk(payload_mu_);
          exec_complete_ = true;
        }
        payload_cv_.notify_all();
      };
      curr_payload_->AddInternalReleaseCallback(callback);
      curr_payload_->MarkSaturated();

      // Enqueue the payload to RateLimiter
      model_instance_->Model()->Server()->GetRateLimiter()->EnqueuePayload(
          model_instance_->Model(), curr_payload_);
      NewPayload();
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping Direct sequence-batch scheduler thread "
                 << model_instance_->Name() << "...";
}

OldestSequenceBatch::OldestSequenceBatch(
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
    bool* is_initialized)
    : SequenceBatch(
          base, model_instance, seq_slot_cnt, enforce_equal_shape_tensors,
          has_optional_input, start_input_overrides, end_input_overrides,
          startend_input_overrides, continue_input_overrides,
          notready_input_overrides),
      in_flight_(seq_slot_cnt, false), queues_(seq_slot_cnt)
{
  // Initialize to handle CORRID control. If error just exit
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  const auto& config = model_instance->Model()->Config();
  if (!CreateCorrelationIDControl(config)) {
    *is_initialized = false;
    return;
  }

  // Create a dynamic batcher use to batch together sequences for
  // inference.
  std::set<int32_t> preferred_batch_sizes;
  for (const auto size :
       config.sequence_batching().oldest().preferred_batch_size()) {
    preferred_batch_sizes.insert(size);
  }

  // TODO: Provide appropriate request_cache_enable flag when caching
  // is enabled for sequence models.
  Status status = DynamicBatchScheduler::Create(
      model_instance->Model(), model_instance,
      triton::common::GetCpuNiceLevel(config),
      true /* dynamic_batching_enabled */, config.max_batch_size(),
      enforce_equal_shape_tensors_,
      config.sequence_batching().oldest().preserve_ordering(),
      preferred_batch_sizes,
      config.sequence_batching().oldest().max_queue_delay_microseconds(),
      &dynamic_batcher_);
  if (!status.IsOk()) {
    LOG_ERROR << "failed creating dynamic sequence batcher for OldestFirst "
              << model_instance->Name() << ": " << status.Message();
    *is_initialized = false;
    return;
  }

  *is_initialized = true;
}

OldestSequenceBatch::~OldestSequenceBatch()
{
  std::unique_lock<std::mutex> lock(mu_);

  // Wait until all pending requests are completed.
  for (uint32_t seq_slot = 0; seq_slot < queues_.size(); seq_slot++) {
    while (in_flight_[seq_slot] || !queues_[seq_slot].empty()) {
      LOG_VERBOSE(1) << "Waiting for slot " << seq_slot << " with "
                     << (in_flight_[seq_slot] ? "an" : "no")
                     << " in-flight request and " << queues_[seq_slot].size()
                     << " pending requests before exiting";
      cv_.wait(lock);
    }
  }
}

void
OldestSequenceBatch::CompleteAndNext(const uint32_t seq_slot)
{
  {
    std::lock_guard<std::mutex> lock(mu_);

    // We may enqueue 1 or more pending inferences triggered by the
    // completion. If the sequence has a pending inference then it needs
    // to be send to dynamic batcher since the "previous" inference just
    // completed. If this next inference ends up being the end of the
    // sequence (either from the END flag or because the sequence is
    // being force-ended) then we try to fill the now-free sequence slot
    // from the backlog and then send the first inference from that
    // sequence to the dynamic batcher...
    std::deque<std::unique_ptr<InferenceRequest>>& queue = queues_[seq_slot];
    bool retry = true;
    while (retry) {
      retry = false;

      bool release_seq_slot = false;
      in_flight_[seq_slot] = false;

      // If the next sequence inference is ready in the queue then enqueue
      // it in the dynamic batcher now.
      if (!queue.empty()) {
        auto& irequest = queue.front();
        bool retain_queue_front = false;

        // If the request is null then this inference request is from
        // the reaper thread indicating a timed-out sequence. Mark that
        // the sequence slot should be released but otherwise do
        // nothing.
        if (irequest == nullptr) {
          LOG_VERBOSE(1) << "force-end timed-out sequence in batcher "
                         << model_instance_->Name() << ", slot " << seq_slot;
          release_seq_slot = true;
        } else if (irequest->IsCancelled()) {
          const InferenceRequest::SequenceId& correlation_id =
              irequest->CorrelationId();
          LOG_VERBOSE(1) << irequest->LogRequest()
                         << "force-end cancelled sequence CORRID "
                         << correlation_id << " in batcher "
                         << model_instance_->Name() << ", slot " << seq_slot;
          release_seq_slot = true;
          retain_queue_front = true;
        } else {
          const InferenceRequest::SequenceId& correlation_id =
              irequest->CorrelationId();

          // After handling the last inference in a sequence we must
          // release the sequence slot to make it available to another
          // sequence.
          if ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) !=
              0) {
            LOG_VERBOSE(1) << irequest->LogRequest() << "end sequence CORRID "
                           << correlation_id << " in batcher "
                           << model_instance_->Name() << ", slot " << seq_slot;
            release_seq_slot = true;
          }

          // Add the appropriate control tensor values to the request.
          SetControlTensors(irequest, seq_slot, correlation_id);

          // Update the implicit state and set the input state tensors.
          UpdateImplicitState(irequest, seq_slot);

          LOG_VERBOSE(1) << irequest->LogRequest()
                         << "issue to dynamic batcher CORRID " << correlation_id
                         << " in batcher " << model_instance_->Name()
                         << ", slot " << seq_slot;
          in_flight_[seq_slot] = true;

          base_->SequencerPtr()->AddReleaseCallback(
              irequest,
              [this, seq_slot](
                  std::unique_ptr<InferenceRequest>& request,
                  const uint32_t flags) -> Status {
                CompleteAndNext(seq_slot);
                return Status::Success;
              });

          dynamic_batcher_->Enqueue(irequest);
        }

        if (!retain_queue_front) {
          queue.pop_front();
        }
      }

      // If releasing the sequence slot then the sequence queue should be
      // empty and we can now assign a new sequence to the queue (from the
      // backlog).
      if (release_seq_slot) {
        // Unless the sequence is cancelled, there should never be anything in a
        // queue after the END marker. Any requests remaining in the queue will
        // be cancelled.
        if (!queue.empty()) {
          LOG_VERBOSE(2) << "requests remaining when releasing sequence slot "
                         << seq_slot;
        }

        SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
            model_instance_, seq_slot);
        const InferenceRequest::SequenceId& released_cid =
            base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);

        if (released_cid.InSequence()) {
          LOG_VERBOSE(1) << "Enqueued new sequence containing " << queue.size()
                         << " requests into OldestFirst batcher "
                         << model_instance_->Name() << ", slot " << seq_slot;

          // If an inference is already in-flight in the dynamic batcher
          // in this sequence slot then can't process the new queue
          // inferences right now, because the in-flight request is
          // using slot resources like the CORRID override map.
          if (!in_flight_[seq_slot]) {
            retry = true;
          }
        }
      }
    }
  }

  // Opportunity for checking queueing and in-flight status.
  cv_.notify_all();
}

void
OldestSequenceBatch::Enqueue(
    const uint32_t seq_slot, const InferenceRequest::SequenceId& correlation_id,
    std::unique_ptr<InferenceRequest>& request)
{
  // Queue the new request... if there isn't already a request in
  // flight for this sequence then send one to the dynamic batcher
  // immediately.
  bool in_flight;
  {
    std::lock_guard<std::mutex> lock(mu_);

    std::deque<std::unique_ptr<InferenceRequest>>& queue = queues_[seq_slot];
    queue.emplace_back(std::move(request));
    in_flight = in_flight_[seq_slot];
  }

  if (!in_flight) {
    CompleteAndNext(seq_slot);
  }
}
}}  // namespace triton::core
