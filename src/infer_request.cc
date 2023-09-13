// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "infer_request.h"

#include <algorithm>
#include <deque>

#include "constants.h"
#include "model.h"
#include "model_config_utils.h"
#include "server.h"
#include "triton/common/logging.h"
#ifdef TRITON_ENABLE_TRACING
#include "cuda_utils.h"
#endif  // TRITON_ENABLE_TRACING

namespace triton { namespace core {

namespace {

// Utilities for Null request feature.
TRITONSERVER_Error*
NullResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "unexpected allocation for null request, no output should be requested.");
}

TRITONSERVER_Error*
NullResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "unexpected release for null request, no output should be requested.");
}

ResponseAllocator null_allocator = ResponseAllocator(
    NullResponseAlloc, NullResponseRelease, nullptr /* start_fn */);

void
NullResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  if (iresponse != nullptr) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting null response");
  }
}

void
NullRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request), "deleting null request");
  }
}

}  // namespace

InferenceRequest::InferenceRequest(
    const std::shared_ptr<Model>& model, const int64_t requested_model_version)
    : InferenceRequest(model.get(), requested_model_version)
{
  model_shared_ = model;
}

InferenceRequest::InferenceRequest(
    Model* model, const int64_t requested_model_version)
    : needs_normalization_(true), model_raw_(model),
      requested_model_version_(requested_model_version), flags_(0),
      correlation_id_(0), batch_size_(0), timeout_us_(0), collect_stats_(true),
      state_(InferenceRequest::State::INITIALIZED), null_request_(false)
{
  SetPriority(0);
}

Status
InferenceRequest::SetState(InferenceRequest::State new_state)
{
  LOG_VERBOSE(1) << LogRequest() << "Setting state from " << state_ << " to "
                 << new_state;
  // No-op if this is already the current state, or if this is a null request.
  if (new_state == state_ || null_request_) {
    return Status::Success;
  }

  // Generate error when called rather than copying it into every case below.
  const auto generate_error = [&]() {
    std::stringstream ss;
    ss << LogRequest() << "Invalid request state transition from " << state_
       << " to " << new_state;
    return Status(Status::Code::INTERNAL, ss.str());
  };

  // Define state transitions
  switch (state_) {
    case InferenceRequest::State::INITIALIZED: {
      if (new_state == InferenceRequest::State::PENDING) {
        IncrementPendingRequestCount();
      } else if (new_state == InferenceRequest::State::RELEASED) {
        // No-op when moving from initialized to released, just releasing early.
      } else {
        return generate_error();
      }
      break;
    }
    case InferenceRequest::State::PENDING: {
      // Request may move from pending to either execution when scheduled to
      // backend, or released early due to some error.
      if (new_state == InferenceRequest::State::EXECUTING ||
          new_state == InferenceRequest::State::RELEASED) {
        DecrementPendingRequestCount();
      } else {
        // Unexpected state transition
        return generate_error();
      }
      break;
    }
    case InferenceRequest::State::EXECUTING: {
      if (new_state != InferenceRequest::State::RELEASED) {
        return generate_error();
      }
      break;
    }
    case InferenceRequest::State::RELEASED: {
      if (new_state != InferenceRequest::State::INITIALIZED) {
        // Only transition currently supported after release is to start over
        // again, such as re-using request objects for multiple inferences.
        return generate_error();
      }
      break;
    }
  }
  state_ = new_state;
  return Status::Success;
}

void
InferenceRequest::IncrementPendingRequestCount()
{
#ifdef TRITON_ENABLE_METRICS
  // Pending request count should always be 0 or 1 per-request. If a request
  // increments the count, it should not be incremented again until decremented.
  auto reporter = model_raw_->MetricReporter();
  if (reporter) {
    reporter->IncrementGauge(kPendingRequestMetric, 1);
  }
#endif  // TRITON_ENABLE_METRICS
}

void
InferenceRequest::DecrementPendingRequestCount()
{
#ifdef TRITON_ENABLE_METRICS
  // Pending request count should always be 0 or 1 per-request. A request should
  // not decrement the count unless it has already been incremented.
  auto reporter = model_raw_->MetricReporter();
  if (reporter) {
    reporter->DecrementGauge(kPendingRequestMetric, 1);
  }
#endif  // TRITON_ENABLE_METRICS
}

const std::string&
InferenceRequest::ModelName() const
{
  return model_raw_->Name();
}

int64_t
InferenceRequest::ActualModelVersion() const
{
  return model_raw_->Version();
}

void
InferenceRequest::SetPriority(uint64_t p)
{
  if ((p == 0) || (p > model_raw_->MaxPriorityLevel())) {
    priority_ = model_raw_->DefaultPriorityLevel();
  } else {
    priority_ = p;
  }
}

Status
InferenceRequest::AddParameter(const char* name, const char* value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceRequest::AddParameter(const char* name, const int64_t value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceRequest::AddParameter(const char* name, const bool value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceRequest::SetParameters(
    const std::deque<InferenceParameter>& parameters)
{
  // NOTE: For BYTES parameters, this will shallow copy the pointer for now.
  parameters_ = parameters;
  return Status::Success;
}

#ifdef TRITON_ENABLE_TRACING
Status
InferenceRequest::TraceInputTensors(
    TRITONSERVER_InferenceTraceActivity activity, const std::string& msg)
{
  const auto& inputs = this->ImmutableInputs();
  TRITONSERVER_MemoryType dst_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t dst_memory_type_id = 0;

  for (const auto& pr : inputs) {
    InferenceRequest::Input* ti = pr.second;

    // input data
    const std::string& name = ti->Name();
    TRITONSERVER_DataType datatype = DataTypeToTriton(ti->DType());
    uint64_t byte_size = ti->Data()->TotalByteSize();
    const int64_t* shape = ti->ShapeWithBatchDim().data();
    uint32_t dim_count = ti->ShapeWithBatchDim().size();
    uint32_t buffer_count = ti->DataBufferCount();
    // chunk buffer
    Status status;
    const void* buffer;
    uint64_t buffer_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    bool cuda_used;

    if (buffer_count == 0) {
      LOG_STATUS_ERROR(
          status, LogRequest() +
                      TRITONSERVER_InferenceTraceActivityString(activity) +
                      ": " + msg + ": tensor: " + name + ": no buffer chunk");
      continue;
    }

    if (buffer_count == 1) {
      status = ti->DataBuffer(
          0, &buffer, &buffer_size, &src_memory_type, &src_memory_type_id);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            status, LogRequest() +
                        TRITONSERVER_InferenceTraceActivityString(activity) +
                        ": " + msg + ": tensor: " + name +
                        ": fail to get data buffer: " + status.Message());
        return status;
      }

      if (buffer_size != byte_size) {
        LOG_STATUS_ERROR(
            status,
            LogRequest() + TRITONSERVER_InferenceTraceActivityString(activity) +
                ": " + msg + ": tensor: " + name + ": truncated buffer");
        continue;
      }

      INFER_TRACE_TENSOR_ACTIVITY(
          this->trace_, activity, name.c_str(), datatype,
          const_cast<void*>(buffer), buffer_size, shape, dim_count,
          src_memory_type, src_memory_type_id);

      continue;
    }

    // input buffer
    std::vector<char> in_buffer(byte_size);
    char* base = in_buffer.data();
    size_t offset = 0;
    for (uint32_t b = 0; b < buffer_count; ++b) {
      status = ti->DataBuffer(
          b, &buffer, &buffer_size, &src_memory_type, &src_memory_type_id);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            status, LogRequest() +
                        TRITONSERVER_InferenceTraceActivityString(activity) +
                        ": " + msg + ": tensor: " + name +
                        ": fail to get data buffer: " + status.Message());
        return status;
      }

      status = CopyBuffer(
          "InferenceRequest TraceInputTensors", src_memory_type,
          src_memory_type_id, dst_memory_type, dst_memory_type_id, buffer_size,
          buffer, base + offset, nullptr, &cuda_used);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            status, LogRequest() +
                        TRITONSERVER_InferenceTraceActivityString(activity) +
                        ": " + msg + ": tensor: " + name +
                        ": fail to copy buffer: " + status.Message());
        return status;
      }

      offset += buffer_size;
    }

    INFER_TRACE_TENSOR_ACTIVITY(
        this->trace_, activity, name.c_str(), datatype,
        static_cast<void*>(base), byte_size, shape, dim_count, dst_memory_type,
        dst_memory_type_id);
  }

  return Status::Success;
}
#endif  // TRITON_ENABLE_TRACING

Status
InferenceRequest::OutputBufferProperties(
    const char* name, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id)
{
  const auto allocator = response_factory_->Allocator();
  if ((allocator == nullptr) || (allocator->QueryFn() == nullptr)) {
    return Status(
        Status::Code::UNAVAILABLE,
        (LogRequest() + "Output properties are not available").c_str());
  } else {
    RETURN_IF_TRITONSERVER_ERROR(allocator->QueryFn()(
        reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
            const_cast<ResponseAllocator*>(allocator)),
        response_factory_->AllocatorUserp(), name, byte_size, memory_type,
        memory_type_id));
  }
  return Status::Success;
}

Status
InferenceRequest::Run(std::unique_ptr<InferenceRequest>& request)
{
  RETURN_IF_ERROR(request->SetState(InferenceRequest::State::PENDING));
  return request->model_raw_->Enqueue(request);
}

void
InferenceRequest::RespondIfError(
    std::unique_ptr<InferenceRequest>& request, const Status& status,
    const bool release_request)
{
  if (status.IsOk()) {
    return;
  }

  // Use the response factory to create a response, set the status,
  // and send it. If something goes wrong all we can do is log the
  // error. Because this is sending an error we assume that this is
  // the last response for the request and so set the FINAL flag.
  std::unique_ptr<InferenceResponse> response;
  LOG_STATUS_ERROR(
      request->response_factory_->CreateResponse(&response),
      (request->LogRequest() + "failed to create error response").c_str());
  LOG_STATUS_ERROR(
      InferenceResponse::SendWithStatus(
          std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status),
      (request->LogRequest() + "failed to send error response").c_str());

  // If releasing the request then invoke the release callback which
  // gives ownership to the callback. So can't access 'request' after
  // this point.
  if (release_request) {
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);
  }
}

void
InferenceRequest::RespondIfError(
    std::vector<std::unique_ptr<InferenceRequest>>& requests,
    const Status& status, const bool release_requests)
{
  if (status.IsOk()) {
    return;
  }

  for (auto& request : requests) {
    RespondIfError(request, status, release_requests);
  }
}

void
InferenceRequest::Release(
    std::unique_ptr<InferenceRequest>&& request, const uint32_t release_flags)
{
  // Invoke the release callbacks added internally before releasing the
  // request to user provided callback.
  for (auto it = request->release_callbacks_.rbegin();
       it != request->release_callbacks_.rend(); it++) {
    (*it)();
  }
  request->release_callbacks_.clear();

#ifdef TRITON_ENABLE_TRACING
  // If tracing then record request end and release the trace.
  // This must be before the request callback to ensure the trace
  // is properly layered, as the request may be nested in an ensemble
  // and the callback may interact with upper level trace.
  if (request->trace_ != nullptr) {
    request->trace_->ReportNow(TRITONSERVER_TRACE_REQUEST_END);
    request->ReleaseTrace();
  }
#endif  // TRITON_ENABLE_TRACING

  LOG_STATUS_ERROR(
      request->SetState(InferenceRequest::State::RELEASED),
      "Failed to set released state");
  void* userp = request->release_userp_;
  auto& release_fn = request->release_fn_;
  release_fn(
      reinterpret_cast<TRITONSERVER_InferenceRequest*>(request.release()),
      release_flags, userp);
}

InferenceRequest*
InferenceRequest::CopyAsNull(const InferenceRequest& from)
{
  // Create a copy of 'from' request with artificial inputs and no requested
  // outputs. Maybe more efficient to share inputs and other metadata,
  // but that binds the Null request with 'from' request's lifecycle.
  std::unique_ptr<InferenceRequest> lrequest(
      new InferenceRequest(from.model_raw_, from.requested_model_version_));
  lrequest->null_request_ = true;
  lrequest->needs_normalization_ = false;
  lrequest->batch_size_ = from.batch_size_;
  lrequest->collect_stats_ = false;

  // Three passes: first to construct input for the shape tensors inputs, second
  // to obtain the max input byte size for allocating a large enough buffer for
  // all non shape tensor inputs; third to construct the inputs for these
  // tensors.
  //  First pass
  for (const auto& input : from.OriginalInputs()) {
    // Handle only shape tensors in this pass
    if (!input.second.IsShapeTensor()) {
      continue;
    }

    // Prepare the memory to hold input data
    size_t byte_size = input.second.Data()->TotalByteSize();
    auto mem_type = TRITONSERVER_MEMORY_CPU;
    int64_t mem_id = 0;
    std::shared_ptr<MutableMemory> data =
        std::make_shared<AllocatedMemory>(byte_size, mem_type, mem_id);

    // Get the source buffer. Assumes shape tensors be in a single buffer on the
    // CPU
    const auto& from_data = input.second.Data();
    size_t from_data_byte_size;
    TRITONSERVER_MemoryType from_data_memory_type;
    int64_t from_data_memory_id;
    const char* from_data_buffer = from_data->BufferAt(
        0 /* idx */, &from_data_byte_size, &from_data_memory_type,
        &from_data_memory_id);

    if (from_data_byte_size != byte_size) {
      LOG_WARNING
          << lrequest->LogRequest()
          << "The byte size of shape tensor to be copied does not match";
    }

    // Copy the shape values to the input buffer
    std::memcpy(data->MutableBuffer(), from_data_buffer, from_data_byte_size);

    Input* new_input;
    lrequest->AddOriginalInput(
        input.first, input.second.DType(), input.second.Shape(), &new_input);

    // Must normalize shape here...
    *new_input->MutableShape() = input.second.Shape();
    *new_input->MutableShapeWithBatchDim() = input.second.ShapeWithBatchDim();

    new_input->SetData(data);
  }

  // Second pass
  size_t max_byte_size = 0;
  size_t max_str_byte_size = 0;
  const std::string* max_input_name = nullptr;
  for (const auto& input : from.OriginalInputs()) {
    // Skip shape tensors in this pass
    if (input.second.IsShapeTensor()) {
      continue;
    }

    if (input.second.DType() == inference::DataType::TYPE_STRING) {
      int64_t element_count =
          triton::common::GetElementCount(input.second.Shape());

      size_t str_byte_size = static_cast<size_t>(4 * element_count);
      max_str_byte_size = std::max(str_byte_size, max_str_byte_size);
      if (str_byte_size > max_byte_size) {
        max_byte_size = str_byte_size;
        max_input_name = &(input.first);
      }
    } else {
      if (input.second.Data()->TotalByteSize() >= max_byte_size) {
        max_byte_size = input.second.Data()->TotalByteSize();
        max_input_name = &(input.first);
      }
    }
  }

  // Third pass
  // [DLIS-1268] should use one growable static buffer for all null requests
  auto mem_type = TRITONSERVER_MEMORY_CPU;
  int64_t mem_id = 0;
  std::shared_ptr<MutableMemory> data =
      std::make_shared<AllocatedMemory>(max_byte_size, mem_type, mem_id);
  auto data_base = data->BufferAt(0, &max_byte_size, &mem_type, &mem_id);

  // Zero initialization is only required when there is a TYPE_BYTES tensor in
  // the request. Only set the required number of bytes to zero.
  if (max_str_byte_size > 0) {
    std::fill(
        data->MutableBuffer(), data->MutableBuffer() + max_str_byte_size, 0);
  }

  for (const auto& input : from.OriginalInputs()) {
    // skip shape tensors in this pass
    if (input.second.IsShapeTensor()) {
      continue;
    }
    Input* new_input;
    lrequest->AddOriginalInput(
        input.first, input.second.DType(), input.second.Shape(), &new_input);

    // Must normalize shape here...
    *new_input->MutableShape() = input.second.Shape();
    *new_input->MutableShapeWithBatchDim() = input.second.ShapeWithBatchDim();

    // Note that the input that have max byte size will be responsible for
    // holding the artificial data, while other inputs will hold a reference to
    // it with byte size that matches 'from'
    if (input.first == *max_input_name) {
      new_input->SetData(data);
    } else {
      if (inference::DataType::TYPE_STRING == input.second.DType()) {
        new_input->AppendData(
            data_base,
            triton::common::GetElementCount(input.second.Shape()) * 4, mem_type,
            mem_id);
      } else {
        new_input->AppendData(
            data_base, input.second.Data()->TotalByteSize(), mem_type, mem_id);
      }
    }
  }

  // No outputs were requested and thus there should be no allocations.
  lrequest->SetResponseCallback(
      &null_allocator, nullptr, NullResponseComplete, nullptr);
  lrequest->SetReleaseCallback(NullRequestComplete, nullptr);
  lrequest->SetResponseFactory();

  // Must normalize inputs here...
  for (auto& pr : lrequest->original_inputs_) {
    lrequest->inputs_.emplace(
        std::make_pair(pr.second.Name(), std::addressof(pr.second)));
  }

  return lrequest.release();
}

Status
InferenceRequest::MutableOriginalInput(
    const std::string& name, InferenceRequest::Input** input)
{
  auto itr = original_inputs_.find(name);
  if (itr == original_inputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "input '" + name + "' does not exist in request");
  }

  *input = &(itr->second);

  return Status::Success;
}

Status
InferenceRequest::ImmutableInput(
    const std::string& name, const InferenceRequest::Input** input) const
{
  auto itr = inputs_.find(name);
  if (itr == inputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "input '" + name + "' does not exist in request");
  }

  *input = itr->second;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count,
    InferenceRequest::Input** input)
{
  const auto& pr = original_inputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple(name, datatype, shape, dim_count));
  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "input '" + name + "' already exists in request");
  }

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, InferenceRequest::Input** input)
{
  return AddOriginalInput(name, datatype, &shape[0], shape.size(), input);
}

Status
InferenceRequest::AddRawInput(
    const std::string& name, InferenceRequest::Input** input)
{
  if (original_inputs_.size() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "raw input '" + name +
            "' can't be added to request with other inputs");
  }
  const auto& pr = original_inputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple());
  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "input '" + name + "' already exists in request");
  }

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  raw_input_name_ = name;
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveOriginalInput(const std::string& name)
{
  if (original_inputs_.erase(name) != 1) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "input '" + name + "' does not exist in request");
  }

  if (name == raw_input_name_) {
    raw_input_name_.clear();
  }
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveAllOriginalInputs()
{
  original_inputs_.clear();
  raw_input_name_.clear();
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOverrideInput(
    const std::string& name, const inference::DataType datatype,
    const int64_t batch_size, const std::vector<int64_t>& shape,
    std::shared_ptr<InferenceRequest::Input>* input)
{
  std::shared_ptr<Input> i = std::make_shared<Input>(name, datatype, shape);
  *(i->MutableShape()) = i->OriginalShape();
  if (batch_size > 0) {
    *(i->MutableShapeWithBatchDim()) = {batch_size};
    i->MutableShapeWithBatchDim()->insert(
        i->MutableShapeWithBatchDim()->end(), i->OriginalShape().begin(),
        i->OriginalShape().end());
  } else {
    *(i->MutableShapeWithBatchDim()) = i->OriginalShape();
  }

  RETURN_IF_ERROR(AddOverrideInput(i));
  if (input != nullptr) {
    *input = std::move(i);
  }

  return Status::Success;
}

Status
InferenceRequest::AddOverrideInput(
    const std::shared_ptr<InferenceRequest::Input>& input)
{
  LOG_VERBOSE(1) << LogRequest() << "adding input override for "
                 << input->Name() << ": " << *this;

  const auto& pr =
      override_inputs_.emplace(std::make_pair(input->Name(), input));
  if (!pr.second) {
    pr.first->second = input;
  }

  // Add or replace this override in the inputs...
  const auto res = inputs_.emplace(std::make_pair(input->Name(), input.get()));
  if (!res.second) {
    res.first->second = input.get();
  }

  LOG_VERBOSE(1) << LogRequest() << "added input override for " << input->Name()
                 << ": " << *this;

  return Status::Success;
}

Status
InferenceRequest::AddOriginalRequestedOutput(const std::string& name)
{
  original_requested_outputs_.insert(name);
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::LoadInputStates()
{
  // Add the input states to the inference request.
  if (sequence_states_ != nullptr) {
    if (sequence_states_->IsNullRequest()) {
      sequence_states_ =
          SequenceStates::CopyAsNull(sequence_states_->NullSequenceStates());
    }
    for (auto& input_state_pair : sequence_states_->InputStates()) {
      auto& input_state = input_state_pair.second;
      std::shared_ptr<InferenceRequest::Input> input =
          std::make_shared<InferenceRequest::Input>(
              input_state->Name(), input_state->DType(), input_state->Shape());
      *input->MutableShapeWithBatchDim() = input_state->Shape();
      input->SetData(input_state->Data());
      AddOverrideInput(input);
    }
  }

  return Status::Success;
}

Status
InferenceRequest::RemoveOriginalRequestedOutput(const std::string& name)
{
  original_requested_outputs_.erase(name);
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveAllOriginalRequestedOutputs()
{
  original_requested_outputs_.clear();
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::PrepareForInference()
{
  // Remove override inputs as those are added during any previous
  // inference execution.
  inputs_.clear();
  override_inputs_.clear();
  SetResponseFactory();

  // Renormalize if anything has changed in the inference request in a
  // way that could impact renormalization.
  if (needs_normalization_) {
    RETURN_IF_ERROR(Normalize());
    needs_normalization_ = false;
  }

  // Initially show the actual inputs to be only the original
  // inputs. If overrides are added later they will be added to
  // 'inputs_'.
  for (auto& pr : original_inputs_) {
    inputs_.emplace(
        std::make_pair(pr.second.Name(), std::addressof(pr.second)));
  }

  // Clear the timestamps
  queue_start_ns_ = 0;
  batcher_start_ns_ = 0;
#ifdef TRITON_ENABLE_STATS
  request_start_ns_ = 0;
#endif  // TRITON_ENABLE_STATS

  // Help enforce that PrepareForInference() is called prior to Run().
  RETURN_IF_ERROR(SetState(InferenceRequest::State::INITIALIZED));

  LOG_VERBOSE(1) << LogRequest() << "prepared: " << *this;
  return Status::Success;
}

Status
InferenceRequest::Normalize()
{
  const inference::ModelConfig& model_config = model_raw_->Config();

  // Fill metadata for raw input
  if (!raw_input_name_.empty()) {
    const bool has_multiple_inputs =
        (original_inputs_.size() != 1) || (model_config.input_size() != 1);
    if (has_multiple_inputs) {
      return Status(
          Status::Code::INVALID_ARG,
          LogRequest() + "Raw request must only have 1 input (found " +
              std::to_string(original_inputs_.size()) +
              ") to be deduced but got " +
              std::to_string(model_config.input_size()) + " inputs in '" +
              ModelName() + "' model configuration");
    }
    auto it = original_inputs_.begin();
    if (raw_input_name_ != it->first) {
      return Status(
          Status::Code::INVALID_ARG,
          LogRequest() + "Unexpected reference name for raw input '" +
              raw_input_name_ + "' got '" + it->first + "'");
    }
    const auto& config_input = model_config.input(0);
    auto& raw_input = it->second;
    std::vector<int64_t> shape;
    if (model_config.max_batch_size() != 0) {
      shape.emplace_back(1);
    }
    int64_t dynamic_axis = -1;
    size_t element_cnt = 1;
    for (const auto& dim : config_input.dims()) {
      if (dim == triton::common::WILDCARD_DIM) {
        if (dynamic_axis != -1) {
          return Status(
              Status::Code::INVALID_ARG,
              LogRequest() + "The shape of the raw input '" +
                  config_input.name() +
                  "' can not be deduced because there are more than one "
                  "variable-sized dimension");
        }
        dynamic_axis = shape.size();
      } else {
        element_cnt *= (size_t)dim;
      }
      shape.emplace_back(dim);
    }
    if ((config_input.data_type() == inference::DataType::TYPE_STRING)) {
      const bool has_one_element = (dynamic_axis == -1) && (element_cnt == 1);
      if (!has_one_element) {
        return Status(
            Status::Code::INVALID_ARG, LogRequest() +
                                           "For BYTE datatype raw input '" +
                                           config_input.name() +
                                           "', the "
                                           "model must have input shape [1]");
      }
      // In the case of BYTE data type, we will prepend the byte size to follow
      // the Triton convention.
      raw_input_size_ = raw_input.Data()->TotalByteSize();
      RETURN_IF_ERROR(raw_input.PrependData(
          &raw_input_size_, sizeof(uint32_t), TRITONSERVER_MEMORY_CPU, 0));
      // Limit the BYTE raw input not to have host policy specific input for
      // simplicity, such case won't happen given the current protocol spec.
      // Will need to extend Input::PrependData() if needed.
      if (!raw_input.HostPolicyData().empty()) {
        return Status(
            Status::Code::INVALID_ARG, LogRequest() +
                                           "Raw input with data associated "
                                           "with a host policy setting is not "
                                           "currently supported");
      }
    } else if (dynamic_axis != -1) {
      shape[dynamic_axis] =
          raw_input.Data()->TotalByteSize() / element_cnt /
          triton::common::GetDataTypeByteSize(config_input.data_type());
    }
    raw_input.SetMetadata(config_input.name(), config_input.data_type(), shape);
  }

  // Initialize the requested outputs to be used during inference. If
  // original_requested_outputs_ is empty assume all outputs specified
  // in model config are being requested.
  requested_outputs_.clear();
  if (original_requested_outputs_.size() == 0) {
    for (const auto& output : model_config.output()) {
      requested_outputs_.insert(output.name());
    }
  } else {
    // Validate if the original requested output name exists in the
    // model configuration.
    for (const auto& output_name : original_requested_outputs_) {
      const inference::ModelOutput* output_config;
      RETURN_IF_ERROR(model_raw_->GetOutput(output_name, &output_config));
    }
  }
  // Make sure that the request is providing the number of inputs
  // as is expected by the model.
  if ((original_inputs_.size() > (size_t)model_config.input_size()) ||
      (original_inputs_.size() < model_raw_->RequiredInputCount())) {
    // If no input is marked as optional, then use exact match error message
    // for consistency / backward compatibility
    if ((size_t)model_config.input_size() == model_raw_->RequiredInputCount()) {
      return Status(
          Status::Code::INVALID_ARG,
          LogRequest() + "expected " +
              std::to_string(model_config.input_size()) + " inputs but got " +
              std::to_string(original_inputs_.size()) + " inputs for model '" +
              ModelName() + "'");
    } else {
      return Status(
          Status::Code::INVALID_ARG,
          LogRequest() + "expected number of inputs between " +
              std::to_string(model_raw_->RequiredInputCount()) + " and " +
              std::to_string(model_config.input_size()) + " but got " +
              std::to_string(original_inputs_.size()) + " inputs for model '" +
              ModelName() + "'");
    }
  }

  // Determine the batch size and shape of each input.
  if (model_config.max_batch_size() == 0) {
    // Model does not support Triton-style batching so set as
    // batch-size 0 and leave the tensor shapes as they are.
    batch_size_ = 0;
    for (auto& pr : original_inputs_) {
      auto& input = pr.second;
      *input.MutableShape() = input.OriginalShape();
    }
  } else {
    // Model does support Triton-style batching so each input tensor
    // must have the same first dimension which is the batch
    // size. Adjust the shape of the input tensors to remove the batch
    // dimension.
    batch_size_ = 0;
    for (auto& pr : original_inputs_) {
      auto& input = pr.second;

      // For a shape tensor, keep the tensor's shape as it is and mark
      // that the input is a shape tensor.
      const inference::ModelInput* input_config;
      RETURN_IF_ERROR(model_raw_->GetInput(input.Name(), &input_config));
      if (input_config->is_shape_tensor()) {
        *input.MutableShape() = input.OriginalShape();
        input.SetIsShapeTensor(true);
        continue;
      }

      if (input.OriginalShape().size() == 0) {
        return Status(
            Status::Code::INVALID_ARG,
            LogRequest() + "input '" + input.Name() +
                "' has no shape but model requires batch dimension for '" +
                ModelName() + "'");
      }

      if (batch_size_ == 0) {
        batch_size_ = input.OriginalShape()[0];
      } else if (input.OriginalShape()[0] != batch_size_) {
        return Status(
            Status::Code::INVALID_ARG,
            LogRequest() + "input '" + input.Name() +
                "' batch size does not match other inputs for '" + ModelName() +
                "'");
      }

      input.MutableShape()->assign(
          input.OriginalShape().begin() + 1, input.OriginalShape().end());
    }
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model.
  if ((int)batch_size_ > model_config.max_batch_size()) {
    return Status(
        Status::Code::INVALID_ARG,
        LogRequest() + "inference request batch-size must be <= " +
            std::to_string(model_config.max_batch_size()) + " for '" +
            ModelName() + "'");
  }

  // Verify that each input shape is valid for the model, make
  // adjustments for reshapes and find the total tensor size.
  for (auto& pr : original_inputs_) {
    const inference::ModelInput* input_config;
    RETURN_IF_ERROR(model_raw_->GetInput(pr.second.Name(), &input_config));

    auto& input = pr.second;
    auto shape = input.MutableShape();

    if (input.DType() != input_config->data_type()) {
      return Status(
          Status::Code::INVALID_ARG,
          LogRequest() + "inference input '" + pr.first + "' data-type is '" +
              std::string(
                  triton::common::DataTypeToProtocolString(input.DType())) +
              "', but model '" + ModelName() + "' expects '" +
              std::string(triton::common::DataTypeToProtocolString(
                  input_config->data_type())) +
              "'");
    }

    // Validate input shape
    {
      bool match_config = true;
      const auto& config_dims = input_config->dims();
      const auto& input_dims = *shape;
      if (config_dims.size() != (int64_t)input_dims.size()) {
        match_config = false;
      } else {
        for (int i = 0; i < config_dims.size(); ++i) {
          if (input_dims[i] == triton::common::WILDCARD_DIM) {
            return Status(
                Status::Code::INVALID_ARG,
                LogRequest() +
                    "All input dimensions should be specified for input '" +
                    pr.first + "' for model '" + ModelName() + "', got " +
                    triton::common::DimsListToString(input.OriginalShape()));
          } else if (
              (config_dims[i] != triton::common::WILDCARD_DIM) &&
              (config_dims[i] != input_dims[i])) {
            match_config = false;
            break;
          }
        }
      }

      if (!match_config) {
        triton::common::DimsList full_dims;
        std::string implicit_batch_note = "";
        if (model_config.max_batch_size() > 0) {
          full_dims.Add(triton::common::WILDCARD_DIM);
          implicit_batch_note =
              "NOTE: Setting a non-zero max_batch_size in the model config "
              "requires a batch dimension to be prepended to each input shape. "
              "If you want to specify the full shape including the batch dim "
              "in your input dims config, try setting max_batch_size to zero. "
              "See the model configuration docs for more info on "
              "max_batch_size.";
        }
        for (int i = 0; i < input_config->dims_size(); ++i) {
          full_dims.Add(input_config->dims(i));
        }
        return Status(
            Status::Code::INVALID_ARG,
            LogRequest() + "unexpected shape for input '" + pr.first +
                "' for model '" + ModelName() + "'. Expected " +
                triton::common::DimsListToString(full_dims) + ", got " +
                triton::common::DimsListToString(input.OriginalShape()) + ". " +
                implicit_batch_note);
      }
    }

    // If there is a reshape for this input then adjust them to
    // match the reshape. As reshape may have variable-size
    // dimensions, we need to record corresponding value so that we
    // can set the value correctly for reshape.
    if (input_config->has_reshape()) {
      std::deque<int64_t> variable_size_values;
      for (int64_t idx = 0; idx < input_config->dims_size(); idx++) {
        if (input_config->dims(idx) == -1) {
          variable_size_values.push_back((*shape)[idx]);
        }
      }

      shape->clear();
      for (const auto& dim : input_config->reshape().shape()) {
        if (dim == -1) {
          shape->push_back(variable_size_values.front());
          variable_size_values.pop_front();
        } else {
          shape->push_back(dim);
        }
      }
    }

    // Create shape with batch dimension.
    // FIXME, should not need this!!
    if (batch_size_ == 0) {
      *input.MutableShapeWithBatchDim() = *shape;
    } else {
      input.MutableShapeWithBatchDim()->clear();
      input.MutableShapeWithBatchDim()->push_back(batch_size_);
      for (int64_t d : *shape) {
        input.MutableShapeWithBatchDim()->push_back(d);
      }
    }
  }

  return Status::Success;
}

#ifdef TRITON_ENABLE_STATS
void
InferenceRequest::ReportStatistics(
    MetricModelReporter* metric_reporter, bool success,
    const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
    const uint64_t compute_output_start_ns, const uint64_t compute_end_ns)
{
  if (!collect_stats_) {
    return;
  }

#ifdef TRITON_ENABLE_TRACING
  if (trace_ != nullptr) {
    trace_->Report(TRITONSERVER_TRACE_COMPUTE_START, compute_start_ns);
    trace_->Report(TRITONSERVER_TRACE_COMPUTE_INPUT_END, compute_input_end_ns);
    trace_->Report(
        TRITONSERVER_TRACE_COMPUTE_OUTPUT_START, compute_output_start_ns);
    trace_->Report(TRITONSERVER_TRACE_COMPUTE_END, compute_end_ns);
  }
#endif  // TRITON_ENABLE_TRACING

  INFER_STATS_DECL_TIMESTAMP(request_end_ns);

  if (success) {
    model_raw_->MutableStatsAggregator()->UpdateSuccess(
        metric_reporter, std::max(1U, batch_size_), request_start_ns_,
        queue_start_ns_, compute_start_ns, compute_input_end_ns,
        compute_output_start_ns, compute_end_ns, request_end_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateSuccess(
          nullptr /* metric_reporter */, std::max(1U, batch_size_),
          request_start_ns_, queue_start_ns_, compute_start_ns,
          compute_input_end_ns, compute_output_start_ns, compute_end_ns,
          request_end_ns);
    }
  } else {
    model_raw_->MutableStatsAggregator()->UpdateFailure(
        metric_reporter, request_start_ns_, request_end_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateFailure(
          nullptr /* metric_reporter */, request_start_ns_, request_end_ns);
    }
  }
}

void
InferenceRequest::ReportStatisticsWithDuration(
    MetricModelReporter* metric_reporter, bool success,
    const uint64_t compute_start_ns, const uint64_t compute_input_duration_ns,
    const uint64_t compute_infer_duration_ns,
    const uint64_t compute_output_duration_ns)
{
  if (!collect_stats_) {
    return;
  }

  INFER_STATS_DECL_TIMESTAMP(request_end_ns);

  if (success) {
    model_raw_->MutableStatsAggregator()->UpdateSuccessWithDuration(
        metric_reporter, std::max(1U, batch_size_), request_start_ns_,
        queue_start_ns_, compute_start_ns, request_end_ns,
        compute_input_duration_ns, compute_infer_duration_ns,
        compute_output_duration_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateSuccessWithDuration(
          nullptr /* metric_reporter */, std::max(1U, batch_size_),
          request_start_ns_, queue_start_ns_, compute_start_ns, request_end_ns,
          compute_input_duration_ns, compute_infer_duration_ns,
          compute_output_duration_ns);
    }
  } else {
    model_raw_->MutableStatsAggregator()->UpdateFailure(
        metric_reporter, request_start_ns_, request_end_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateFailure(
          nullptr /* metric_reporter */, request_start_ns_, request_end_ns);
    }
  }
}

void
InferenceRequest::ReportStatisticsCacheHit(MetricModelReporter* metric_reporter)
{
  // Capture end of request time
  INFER_STATS_DECL_TIMESTAMP(request_end_ns);

  if (cache_lookup_start_ns_ >= cache_lookup_end_ns_) {
    LOG_WARNING << LogRequest()
                << "Cache lookup timestamps were not set correctly. Cache "
                   "lookup duration stats may be incorrect.";
  }
  const uint64_t cache_lookup_duration_ns =
      cache_lookup_end_ns_ - cache_lookup_start_ns_;

  // Cache hit is always success
  model_raw_->MutableStatsAggregator()->UpdateSuccessCacheHit(
      metric_reporter, std::max(1U, batch_size_), request_start_ns_,
      queue_start_ns_, cache_lookup_start_ns_, request_end_ns,
      cache_lookup_duration_ns);
  if (secondary_stats_aggregator_ != nullptr) {
    secondary_stats_aggregator_->UpdateSuccessCacheHit(
        nullptr /* metric_reporter */, std::max(1U, batch_size_),
        request_start_ns_, queue_start_ns_, cache_lookup_start_ns_,
        request_end_ns, cache_lookup_duration_ns);
  }
}
#endif  // TRITON_ENABLE_STATS

//
// Input
//
InferenceRequest::Input::Input()
    : is_shape_tensor_(false), data_(new MemoryReference),
      has_host_policy_specific_data_(false)
{
}

InferenceRequest::Input::Input(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count)
    : name_(name), datatype_(datatype),
      original_shape_(shape, shape + dim_count), is_shape_tensor_(false),
      data_(new MemoryReference), has_host_policy_specific_data_(false)
{
}

InferenceRequest::Input::Input(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape)
    : name_(name), datatype_(datatype), original_shape_(shape),
      is_shape_tensor_(false), data_(new MemoryReference),
      has_host_policy_specific_data_(false)
{
}

void
InferenceRequest::Input::SetMetadata(
    const std::string& name, const inference::DataType& dt,
    const std::vector<int64_t>& shape)
{
  name_ = name;
  datatype_ = dt;
  original_shape_ = shape;
}

Status
InferenceRequest::Input::SetIsShapeTensor(const bool is_shape_tensor)
{
  is_shape_tensor_ = is_shape_tensor;
  return Status::Success;
}

const std::shared_ptr<Memory>&
InferenceRequest::Input::Data(const std::string& host_policy_name) const
{
  auto device_data = host_policy_data_map_.find(host_policy_name);
  if (device_data == host_policy_data_map_.end()) {
    // Fall back on default data if there is no data that has been added for
    // this host policy
    return data_;
  }
  return device_data->second;
}

Status
InferenceRequest::Input::AppendData(
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBuffer(
        static_cast<const char*>(base), byte_size, memory_type, memory_type_id);
  }

  return Status::Success;
}

Status
InferenceRequest::Input::AppendDataWithBufferAttributes(
    const void* base, BufferAttributes* buffer_attributes)
{
  if (buffer_attributes->ByteSize() > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBuffer(
        static_cast<const char*>(base), buffer_attributes);
  }
  return Status::Success;
}

Status
InferenceRequest::Input::AppendDataWithHostPolicy(
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, const char* host_policy_name)
{
  auto device_data = host_policy_data_map_.find(host_policy_name);
  has_host_policy_specific_data_ = true;
  if (device_data == host_policy_data_map_.end()) {
    auto insert_pair = host_policy_data_map_.insert(
        std::make_pair(std::string(host_policy_name), new MemoryReference));
    device_data = insert_pair.first;
  }
  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(device_data->second)
        ->AddBuffer(
            static_cast<const char*>(base), byte_size, memory_type,
            memory_type_id);
  }

  return Status::Success;
}

Status
InferenceRequest::Input::PrependData(
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBufferFront(
        static_cast<const char*>(base), byte_size, memory_type, memory_type_id);
  }

  return Status::Success;
}

Status
InferenceRequest::Input::SetData(const std::shared_ptr<Memory>& data)
{
  if (data_->TotalByteSize() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name_ + "' already has data, can't overwrite");
  }

  data_ = data;

  return Status::Success;
}

Status
InferenceRequest::Input::SetData(
    const std::string& host_policy_name, const std::shared_ptr<Memory>& data)
{
  if (host_policy_data_map_.find(host_policy_name) !=
      host_policy_data_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "input '" + name_ +
                                       "' already has data for host policy '" +
                                       host_policy_name + "', can't overwrite");
  }

  host_policy_data_map_.emplace(host_policy_name, data);

  return Status::Success;
}

Status
InferenceRequest::Input::RemoveAllData()
{
  data_ = std::make_shared<MemoryReference>();
  host_policy_data_map_.clear();
  has_host_policy_specific_data_ = false;
  return Status::Success;
}

Status
InferenceRequest::Input::DataBuffer(
    const size_t idx, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id) const
{
  *base = data_->BufferAt(idx, byte_size, memory_type, memory_type_id);

  return Status::Success;
}

Status
InferenceRequest::Input::DataBufferAttributes(
    const size_t idx, const void** base,
    BufferAttributes** buffer_attributes) const
{
  *base = data_->BufferAt(idx, buffer_attributes);

  return Status::Success;
}

Status
InferenceRequest::Input::DataBufferForHostPolicy(
    const size_t idx, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
    const std::string& host_policy_name) const
{
  auto device_data = host_policy_data_map_.find(host_policy_name);
  if (device_data == host_policy_data_map_.end()) {
    // Return data buffer if there is no host-policy specific buffer available
    *base = data_->BufferAt(idx, byte_size, memory_type, memory_type_id);
  } else {
    *base = device_data->second->BufferAt(
        idx, byte_size, memory_type, memory_type_id);
  }

  return Status::Success;
}

size_t
InferenceRequest::Input::DataBufferCountForHostPolicy(
    const std::string& host_policy_name) const
{
  auto policy_data = host_policy_data_map_.find(host_policy_name);
  if (policy_data != host_policy_data_map_.end()) {
    return policy_data->second->BufferCount();
  }
  return data_->BufferCount();
}

InferenceRequest::SequenceId::SequenceId()
    : sequence_label_(""), sequence_index_(0),
      id_type_(InferenceRequest::SequenceId::DataType::UINT64)
{
}

InferenceRequest::SequenceId::SequenceId(const std::string& sequence_label)
    : sequence_label_(sequence_label), sequence_index_(0),
      id_type_(InferenceRequest::SequenceId::DataType::STRING)
{
}

InferenceRequest::SequenceId::SequenceId(uint64_t sequence_index)
    : sequence_label_(""), sequence_index_(sequence_index),
      id_type_(InferenceRequest::SequenceId::DataType::UINT64)
{
}

InferenceRequest::SequenceId&
InferenceRequest::SequenceId::operator=(const std::string& rhs)
{
  sequence_label_ = rhs;
  sequence_index_ = 0;
  id_type_ = InferenceRequest::SequenceId::DataType::STRING;
  return *this;
}

InferenceRequest::SequenceId&
InferenceRequest::SequenceId::operator=(const uint64_t rhs)
{
  sequence_label_ = "";
  sequence_index_ = rhs;
  id_type_ = InferenceRequest::SequenceId::DataType::UINT64;
  return *this;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest& request)
{
  out << "[0x" << std::addressof(request) << "] "
      << "request id: " << request.Id() << ", model: " << request.ModelName()
      << ", requested version: " << request.RequestedModelVersion()
      << ", actual version: " << request.ActualModelVersion() << ", flags: 0x"
      << std::hex << request.Flags() << std::dec
      << ", correlation id: " << request.CorrelationId()
      << ", batch size: " << request.BatchSize()
      << ", priority: " << request.Priority()
      << ", timeout (us): " << request.TimeoutMicroseconds() << std::endl;

  out << "original inputs:" << std::endl;
  for (const auto& itr : request.OriginalInputs()) {
    out << "[0x" << std::addressof(itr.second) << "] " << itr.second
        << std::endl;
  }

  out << "override inputs:" << std::endl;
  for (const auto& itr : request.OverrideInputs()) {
    out << "[0x" << itr.second.get() << "] " << *itr.second << std::endl;
  }

  out << "inputs:" << std::endl;
  for (const auto& itr : request.ImmutableInputs()) {
    out << "[0x" << itr.second << "] " << *itr.second << std::endl;
  }

  out << "original requested outputs:" << std::endl;
  for (const auto& name : request.OriginalRequestedOutputs()) {
    out << name << std::endl;
  }

  out << "requested outputs:" << std::endl;
  for (const auto& name : request.ImmutableRequestedOutputs()) {
    out << name << std::endl;
  }

  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest::Input& input)
{
  out << "input: " << input.Name()
      << ", type: " << triton::common::DataTypeToProtocolString(input.DType())
      << ", original shape: "
      << triton::common::DimsListToString(input.OriginalShape())
      << ", batch + shape: "
      << triton::common::DimsListToString(input.ShapeWithBatchDim())
      << ", shape: " << triton::common::DimsListToString(input.Shape());
  if (input.IsShapeTensor()) {
    out << ", is_shape_tensor: True";
  }
  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest::SequenceId& sequence_id)
{
  switch (sequence_id.Type()) {
    case InferenceRequest::SequenceId::DataType::STRING:
      out << sequence_id.StringValue();
      break;
    case InferenceRequest::SequenceId::DataType::UINT64:
      out << sequence_id.UnsignedIntValue();
      break;
    default:
      out << sequence_id.UnsignedIntValue();
      break;
  }
  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest::State& state)
{
  switch (state) {
    case InferenceRequest::State::INITIALIZED: {
      out << "INITIALIZED";
      break;
    }
    case InferenceRequest::State::PENDING: {
      out << "PENDING";
      break;
    }
    case InferenceRequest::State::EXECUTING: {
      out << "EXECUTING";
      break;
    }
    case InferenceRequest::State::RELEASED: {
      out << "RELEASED";
      break;
    }
    default:
      out << "UNKNOWN";
  }
  return out;
}

bool
operator==(
    const InferenceRequest::SequenceId lhs,
    const InferenceRequest::SequenceId rhs)
{
  if (lhs.Type() == rhs.Type()) {
    switch (lhs.Type()) {
      case InferenceRequest::SequenceId::DataType::STRING:
        return lhs.StringValue() == rhs.StringValue();
      case InferenceRequest::SequenceId::DataType::UINT64:
        return lhs.UnsignedIntValue() == rhs.UnsignedIntValue();
      default:
        return lhs.UnsignedIntValue() == rhs.UnsignedIntValue();
    }
  } else {
    return false;
  }
}

bool
operator!=(
    const InferenceRequest::SequenceId lhs,
    const InferenceRequest::SequenceId rhs)
{
  return !(lhs == rhs);
}

}}  // namespace triton::core
