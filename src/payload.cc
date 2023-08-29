// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "payload.h"

namespace triton { namespace core {

Payload::Payload()
    : op_type_(Operation::INFER_RUN),
      requests_(std::vector<std::unique_ptr<InferenceRequest>>()),
      OnCallback_([]() {}), instance_(nullptr), state_(State::UNINITIALIZED),
      batcher_start_ns_(0), saturated_(false), user_pointer_(nullptr)
{
  exec_mu_.reset(new std::mutex());
}

const Status&
Payload::MergePayload(std::shared_ptr<Payload>& payload)
{
  if ((payload->GetOpType() != Operation::INFER_RUN) ||
      (op_type_ != Operation::INFER_RUN)) {
    static Status op_type_error(
        Status::Code::INTERNAL,
        "Attempted to merge payloads of type that are not INFER_RUN");
    return op_type_error;
  }
  if (payload->GetInstance() != instance_) {
    static Status instance_error(
        Status::Code::INTERNAL,
        "Attempted to merge payloads of mismatching instance");
    return instance_error;
  }
  if ((payload->GetState() != State::EXECUTING) ||
      (state_ != State::EXECUTING)) {
    static Status state_error(
        Status::Code::INTERNAL,
        "Attempted to merge payloads that are not in executing state");
    return state_error;
  }

  // Skip comparison if not initialized (required), here assume either all
  // payloads are initialized or otherwise.
  if (required_equal_inputs_.Initialized() &&
      !required_equal_inputs_.HasEqualInputs(*payload->Requests().begin())) {
    static Status shape_error(
        Status::Code::INVALID_ARG,
        "Attempted to merge payloads that has non-equal inputs");
    return shape_error;
  }

  requests_.insert(
      requests_.end(), std::make_move_iterator(payload->Requests().begin()),
      std::make_move_iterator(payload->Requests().end()));

  payload->Callback();

  return Status::Success;
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

void
Payload::Release()
{
  op_type_ = Operation::INFER_RUN;
  requests_.clear();
  OnCallback_ = []() {};
  release_callbacks_.clear();
  instance_ = nullptr;
  state_ = State::RELEASED;
  required_equal_inputs_ = RequiredEqualInputs();
  batcher_start_ns_ = 0;
  saturated_ = false;
  user_pointer_ = nullptr;
}

size_t
Payload::BatchSize()
{
  size_t batch_size = 0;
  for (const auto& request : requests_) {
    batch_size += std::max(1U, request->BatchSize());
  }
  return batch_size;
}

void
Payload::ReserveRequests(size_t size)
{
  requests_.reserve(size);
}

void
Payload::AddRequest(std::unique_ptr<InferenceRequest> request)
{
  if ((batcher_start_ns_ == 0) ||
      (batcher_start_ns_ > request->BatcherStartNs())) {
    batcher_start_ns_ = request->BatcherStartNs();
  }
  requests_.push_back(std::move(request));
}

void
Payload::SetCallback(std::function<void()> OnCallback)
{
  OnCallback_ = OnCallback;
}

void
Payload::SetInstance(TritonModelInstance* model_instance)
{
  instance_ = model_instance;
}

void
Payload::AddInternalReleaseCallback(std::function<void()>&& callback)
{
  release_callbacks_.emplace_back(std::move(callback));
}

void
Payload::MarkSaturated()
{
  saturated_ = true;
}

void
Payload::SetState(Payload::State state)
{
  state_ = state;
}

Status
Payload::Wait()
{
  return status_->get_future().get();
}

void
Payload::Callback()
{
  OnCallback_();
}

void
Payload::OnRelease()
{
  // Invoke the release callbacks added internally before releasing the
  // request to user provided callback.
  for (auto it = release_callbacks_.rbegin(); it != release_callbacks_.rend();
       it++) {
    (*it)();
  }
  release_callbacks_.clear();
}

void
Payload::Execute(bool* should_exit)
{
  *should_exit = false;

  Status status;
  switch (op_type_) {
    case Operation::INFER_RUN:
      status = instance_->Schedule(std::move(requests_));
      break;
    case Operation::INIT:
      status = instance_->Initialize();
      break;
    case Operation::WARM_UP:
      status = instance_->WarmUp();
      break;
    case Operation::EXIT:
      *should_exit = true;
  }

  status_->set_value(status);
  // Call specified callback, notifying that execution has completed.
  Callback();
}

}}  // namespace triton::core
