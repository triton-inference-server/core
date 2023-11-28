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
    std::unique_ptr<InferenceRequest> ni(
        InferenceRequest::CopyAsNull(*request));
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
