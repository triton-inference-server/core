#include "sequence_utils.h"

#include "sequence_batch_scheduler.h"

namespace triton { namespace core {
void
GenerativeSequencer::RescheduleRequest(
    std::unique_ptr<InferenceRequest>& request, const uint32_t flags)
{
  if (flags & TRITONSERVER_REQUEST_RELEASE_RESCHEDULE) {
    // set flags to be not START and not END
    request->SetFlags(0);
    base_->Enqueue(request);
  }
  // If not reschedule, use request cancellation to release sequence
  // resources. Check RELEASE_ALL flag to break the recursive calls if
  // the callback is called by cancellation.
  else if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) == 0) {
    request->SetFlags(TRITONSERVER_REQUEST_FLAG_SEQUENCE_END);
    request->Cancel();
    base_->Enqueue(request);
  }
}

}}  // namespace triton::core
