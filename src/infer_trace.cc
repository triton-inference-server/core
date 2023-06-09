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

namespace triton { namespace core {

#ifdef TRITON_ENABLE_TRACING

// Start the trace id at 1, because id 0 is reserved to indicate no
// parent.
std::atomic<uint64_t> InferenceTrace::next_id_(1);

InferenceTrace*
InferenceTrace::SpawnChildTrace()
{
  InferenceTrace* trace = new InferenceTrace(
      level_, mode_, id_, activity_fn_, tensor_activity_fn_, release_fn_,
      userp_);
  return trace;
}

void
InferenceTrace::Release()
{
  release_fn_(reinterpret_cast<TRITONSERVER_InferenceTrace*>(this), userp_);
}

std::shared_ptr<InferenceTraceProxy>
InferenceTraceProxy::SpawnChildTrace()
{
  std::shared_ptr<InferenceTraceProxy> strace_proxy =
      std::make_shared<InferenceTraceProxy>(trace_->SpawnChildTrace());
  return strace_proxy;
}

#ifndef _WIN32

opentelemetry::nostd::shared_ptr<otel_trace_api::Span> 
InferenceTrace::InitSpan(
      std::string name, const otel_common::SystemTimestamp& timestamp_ns,
      const uint64_t& raw_timestamp_ns)
  {
    opentelemetry::nostd::shared_ptr<otel_trace_api::Span> span{nullptr};
    if (otel_tracer_ != nullptr) {
      otel_trace_api::StartSpanOptions options;
      options.start_system_time = timestamp_ns;
      options.start_steady_time = otel_common::SteadyTimestamp{
          std::chrono::nanoseconds{raw_timestamp_ns}};
      options.parent = otel_trace_api::GetSpan(otel_context_)->GetContext();
      span = otel_tracer_->StartSpan(name, options);
    }
    return span;
  }

void 
InferenceTrace::EndActiveSpan(const uint64_t& raw_timestamp_ns)
  {
    opentelemetry::nostd::shared_ptr<otel_trace_api::Span> span =
        otel_trace_api::GetSpan(otel_context_);
    if (span != nullptr) {
      otel_trace_api::EndSpanOptions end_options;
      end_options.end_steady_time = otel_common::SteadyTimestamp{
          std::chrono::nanoseconds{raw_timestamp_ns}};
      span->End(end_options);
    }
  }

void 
InferenceTrace::ReportToOpenTelemetry(
      const TRITONSERVER_InferenceTraceActivity activity,
      const uint64_t& raw_timestamp_ns)
  {
    otel_common::SystemTimestamp otel_timestamp{
        (TimeOffset() + std::chrono::nanoseconds{raw_timestamp_ns})};

    if (activity == TRITONSERVER_TRACE_REQUEST_START) {
      trace_span_ =
          InitSpan(model_name_, otel_timestamp, raw_timestamp_ns);
      if (trace_span_ != nullptr){
        trace_span_->SetAttribute("triton.model_name", model_name_);
        trace_span_->SetAttribute("triton.model_version", model_version_);
        trace_span_->SetAttribute("triton.trace_id", id_);
        trace_span_->SetAttribute("triton.trace_parent_id", parent_id_);
        trace_span_->SetAttribute("triton.request_id", request_id_);
        otel_context_ = otel_trace_api::SetSpan(otel_context_, trace_span_);
      }
    }

    otel_trace_api::GetSpan(otel_context_)
        ->AddEvent(
            TRITONSERVER_InferenceTraceActivityString(activity),
            otel_timestamp);

    if (activity == TRITONSERVER_TRACE_REQUEST_END) {
      EndActiveSpan(raw_timestamp_ns);
    }
  }
#endif // _WIN32

#endif  // TRITON_ENABLE_TRACING

}}  // namespace triton::core
