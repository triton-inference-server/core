// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "sequence_utils.h"

#include "sequence_batch_scheduler.h"

namespace triton { namespace core {
Status
IterativeSequencer::RescheduleRequest(
    std::unique_ptr<InferenceRequest>& request, const uint32_t flags)
{
  if (flags & TRITONSERVER_REQUEST_RELEASE_RESCHEDULE) {
    // set flags to be not START and not END
    request->SetFlags(0);
    return base_->Enqueue(request);
  }
  // If not reschedule, use request cancellation to release sequence
  // resources. Check IsCancelled() to break the recursive calls if
  // the callback is called by cancellation.
  else if (!request->IsCancelled()) {
    // Use a null request to trigger sequence batcher cancellation so
    // additional request manipulation won't affect the actual request.
    std::unique_ptr<InferenceRequest> ni = nullptr;
    RETURN_IF_ERROR(InferenceRequest::CopyAsNull(*request, &ni));
    ni->SetCorrelationId(request->CorrelationId());
    ni->SetFlags(TRITONSERVER_REQUEST_FLAG_SEQUENCE_END);
    ni->Cancel();
    // result of enqueuing null request is not related to the actual request
    auto status = base_->Enqueue(ni);
    if (!status.IsOk()) {
      LOG_ERROR << status.AsString();
    }
  }
  return Status::Success;
}

}}  // namespace triton::core
