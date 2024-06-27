// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "infer_trace.h"

#define TRITONJSON_STATUSTYPE triton::core::Status
#define TRITONJSON_STATUSRETURN(M) \
  return triton::core::Status(triton::core::Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS triton::core::Status::Success
#include "triton/common/logging.h"
#include "triton/common/triton_json.h"

namespace triton { namespace core {

#ifdef TRITON_ENABLE_TRACING

// Start the trace id at 1, because id 0 is reserved to indicate no
// parent.
std::atomic<uint64_t> InferenceTrace::next_id_(1);

InferenceTrace*
InferenceTrace::SpawnChildTrace()
{
  InferenceTrace* trace = new InferenceTrace(
      level_, id_, activity_fn_, tensor_activity_fn_, release_fn_, userp_);
  return trace;
}

void
InferenceTrace::Release()
{
  release_fn_(reinterpret_cast<TRITONSERVER_InferenceTrace*>(this), userp_);
}

void
InferenceTrace::RecordActivityName(
    uint64_t timestamp_ns, std::string activity_name)
{
  triton::common::TritonJson::Value context_json(
      triton::common::TritonJson::ValueType::OBJECT);
  if (!context_.empty()) {
    Status status = context_json.Parse(context_);
    if (!status.IsOk()) {
      LOG_ERROR << "Error parsing trace context";
    }
  }
  std::string key = std::to_string(timestamp_ns);
  context_json.AddString(key.c_str(), &activity_name);
  triton::common::TritonJson::WriteBuffer buffer;
  context_json.Write(&buffer);
  context_ = buffer.Contents();
}

std::shared_ptr<InferenceTraceProxy>
InferenceTraceProxy::SpawnChildTrace()
{
  std::shared_ptr<InferenceTraceProxy> strace_proxy =
      std::make_shared<InferenceTraceProxy>(trace_->SpawnChildTrace());
  return strace_proxy;
}

#endif  // TRITON_ENABLE_TRACING

}}  // namespace triton::core
