// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>

#include "infer_request.h"
#include "status.h"

namespace triton { namespace core {

class SequenceBatchScheduler;

class Sequencer {
 public:
  // Regular sequencer expects the request is well-prepared by the user when
  // sending inference request.
  virtual Status SetupSequenceRequest(
      std::unique_ptr<InferenceRequest>& irequest)
  {
    // A request must have a correlation ID to be processed correctly by
    // this scheduler. A value of 0 (zero) or "" (empty) indicates that the
    // request doesn't have a correlation ID.
    const auto& correlation_id = irequest->CorrelationId();
    if (!correlation_id.InSequence()) {
      return Status(
          Status::Code::INVALID_ARG,
          "inference request to model '" + irequest->ModelName() +
              "' must specify a non-zero or non-empty correlation ID");
    }
    return Status::Success;
  }

  virtual void AddReleaseCallback(
      std::unique_ptr<InferenceRequest>& irequest,
      InferenceRequest::InternalReleaseFn&& callback)
  {
    irequest->AddInternalReleaseCallback(std::move(callback));
  }

  virtual Status RescheduleRequest(
      std::unique_ptr<InferenceRequest>& request, const uint32_t flags)
  {
    // Sequencer will not reschedule requests
    return Status::Success;
  }
};

class IterativeSequencer : public Sequencer {
 public:
  IterativeSequencer(SequenceBatchScheduler* base) : base_(base) {}
  // Iterative sequencer will prepare the request for sequence batcher if it is
  // not associated with an sequence
  Status SetupSequenceRequest(
      std::unique_ptr<InferenceRequest>& irequest) override
  {
    // A request must have a correlation ID to be processed correctly by
    // this scheduler. A value of 0 (zero) or "" (empty) indicates that the
    // request doesn't have a correlation ID.
    const auto& correlation_id = irequest->CorrelationId();
    if (!correlation_id.InSequence()) {
      irequest->SetCorrelationId(InferenceRequest::SequenceId(sequence_id_++));
      irequest->SetFlags(TRITONSERVER_REQUEST_FLAG_SEQUENCE_START);
    }
    return Status::Success;
  }

  void AddReleaseCallback(
      std::unique_ptr<InferenceRequest>& irequest,
      InferenceRequest::InternalReleaseFn&& callback) override
  {
    if (irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) {
      irequest->AddInternalReleaseCallback(std::move(callback));
    }
  }

  Status RescheduleRequest(
      std::unique_ptr<InferenceRequest>& request,
      const uint32_t flags) override;

 private:
  std::atomic<uint64_t> sequence_id_{1};
  SequenceBatchScheduler* const base_;
};

}}  // namespace triton::core
