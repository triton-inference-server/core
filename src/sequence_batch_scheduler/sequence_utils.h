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

  virtual void RescheduleRequest(
      std::unique_ptr<InferenceRequest>& request, const uint32_t flags)
  {
    // Sequencer will not reschedule requests
  }
};

class GenerativeSequencer : public Sequencer {
 public:
  GenerativeSequencer(SequenceBatchScheduler* base) : base_(base) {}
  // Generative sequencer will prepare the request for sequence batcher if it is
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

  void RescheduleRequest(
      std::unique_ptr<InferenceRequest>& request, const uint32_t flags);

 private:
  std::atomic<uint64_t> sequence_id_{1};
  SequenceBatchScheduler* const base_;
};

}}  // namespace triton::core
