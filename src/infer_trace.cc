// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
  std::lock_guard<std::mutex> lock(mu_);
  triton::common::TritonJson::Value context_json(
      triton::common::TritonJson::ValueType::OBJECT);
  if (!context_.empty()) {
    Status status = context_json.Parse(context_);
    if (!status.IsOk()) {
      LOG_ERROR << "Error parsing trace context";
    }
  }
  std::string key = std::to_string(timestamp_ns);
  context_json.SetStringObject(key.c_str(), activity_name);
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
