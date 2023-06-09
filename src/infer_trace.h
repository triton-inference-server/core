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
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include "constants.h"
#include "status.h"
#include "tritonserver_apis.h"
#if !defined(_WIN32) && defined(TRITON_ENABLE_TRACING)
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/sdk/trace/tracer.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/provider.h"
namespace otel_trace_api = opentelemetry::trace;
namespace otel_trace_sdk = opentelemetry::sdk::trace;
namespace otel_common = opentelemetry::common;
#endif

namespace triton { namespace core {

#ifdef TRITON_ENABLE_TRACING

//
// InferenceTrace
//
// Interface to TRITONSERVER_InferenceTrace to report trace events.
//
class InferenceTrace {
 public:
  InferenceTrace(
      const TRITONSERVER_InferenceTraceLevel level,
      const TRITONSERVER_InferenceTraceMode mode, const uint64_t parent_id,
      TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
      TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn,
      TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* userp)
      : level_(level), mode_(mode), id_(next_id_++), parent_id_(parent_id),
        activity_fn_(activity_fn), tensor_activity_fn_(tensor_activity_fn),
        release_fn_(release_fn), userp_(userp)
  {
  }

  InferenceTrace* SpawnChildTrace();

  int64_t Id() const { return id_; }
  int64_t ParentId() const { return parent_id_; }

  const std::string& ModelName() const { return model_name_; }
  int64_t ModelVersion() const { return model_version_; }
  const std::string& RequestId() const { return request_id_; }
  const TRITONSERVER_InferenceTraceMode& TraceMode() const { return mode_; }

  void SetModelName(const std::string& n) { model_name_ = n; }
  void SetModelVersion(int64_t v) { model_version_ = v; }
  void SetRequestId(const std::string& request_id) { request_id_ = request_id; }

  // Report trace activity.
  void Report(
      const TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
  {
    if ((level_ & TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) > 0) {
      activity_fn_(
          reinterpret_cast<TRITONSERVER_InferenceTrace*>(this), activity,
          timestamp_ns, userp_);
#ifndef _WIN32
      if (mode_ == TRITONSERVER_TRACE_MODE_OPENTELEMETRY && parent_id_ == 0) {
        ReportToOpenTelemetry(activity, timestamp_ns);
      }
#endif
    }
  }

#ifndef _WIN32

  otel_trace_api::Tracer* OpenTelemetryTracer() { return otel_tracer_; }
  opentelemetry::nostd::shared_ptr<otel_trace_api::Span> OpenTelemetrySpan()
  {
    return trace_span_;
  }
  opentelemetry::context::Context& GetOpenTelemetryContext()
  {
    return otel_context_;
  }
  std::chrono::nanoseconds TimeOffset()
  {
    return std::chrono::nanoseconds{opentelemetry::nostd::get<uint64_t>(
        opentelemetry::context::RuntimeContext::GetValue(
            "time_offset", &(otel_context_)))};
  };
  void SetOpenTelemetryTracer(otel_trace_api::Tracer* otel_tracer)
  {
    otel_tracer_ = otel_tracer;
  }
  void SetOpenTelemetySpan(
      opentelemetry::nostd::shared_ptr<otel_trace_api::Span> span)
  {
    trace_span_ = span;
  }
  void SetOpenTelemetryContext(opentelemetry::context::Context& ctxt)
  {
    otel_context_ = ctxt;
  }

  opentelemetry::nostd::shared_ptr<otel_trace_api::Span> InitSpan(
      std::string name, const otel_common::SystemTimestamp& timestamp_ns,
      const uint64_t& raw_timestamp_ns)
  {
    opentelemetry::nostd::shared_ptr<otel_trace_api::Span> span{nullptr};
    otel_trace_api::StartSpanOptions options;
    options.start_system_time = timestamp_ns;
    options.start_steady_time = otel_common::SteadyTimestamp{
        std::chrono::nanoseconds{raw_timestamp_ns}};
    options.parent = otel_trace_api::GetSpan(otel_context_)->GetContext();
    if (otel_tracer_ != nullptr) {
      span = otel_tracer_->StartSpan(name, options);
    }
    return span;
  }

  void EndActiveSpan(const uint64_t& raw_timestamp_ns)
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

  void ReportToOpenTelemetry(
      const TRITONSERVER_InferenceTraceActivity activity,
      const uint64_t& raw_timestamp_ns)
  {
    otel_common::SystemTimestamp otel_timestamp{
        (TimeOffset() + std::chrono::nanoseconds{raw_timestamp_ns})};

    if (activity == TRITONSERVER_TRACE_REQUEST_START) {
      trace_span_ =
          InitSpan("request: " + model_name_, otel_timestamp, raw_timestamp_ns);
      otel_context_ = otel_trace_api::SetSpan(otel_context_, trace_span_);
    }

    otel_trace_api::GetSpan(otel_context_)
        ->AddEvent(
            TRITONSERVER_InferenceTraceActivityString(activity),
            otel_timestamp);

    if (activity == TRITONSERVER_TRACE_REQUEST_END) {
      trace_span_->SetAttribute("triton.model_name", model_name_);
      trace_span_->SetAttribute("triton.model_version", model_version_);
      trace_span_->SetAttribute("triton.trace_id", id_);
      trace_span_->SetAttribute("triton.trace_parent_id", parent_id_);
      trace_span_->SetAttribute("triton.request_id", request_id_);
      EndActiveSpan(raw_timestamp_ns);
    }
  }
#endif

  // Report trace activity at the current time.
  void ReportNow(const TRITONSERVER_InferenceTraceActivity activity)
  {
    if ((level_ & TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) > 0) {
      Report(
          activity, std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
    }
  }

  // Report tensor trace activity.
  void ReportTensor(
      const TRITONSERVER_InferenceTraceActivity activity, const char* name,
      TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
      const int64_t* shape, uint64_t dim_count,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
  {
    if ((level_ & TRITONSERVER_TRACE_LEVEL_TENSORS) > 0) {
      tensor_activity_fn_(
          reinterpret_cast<TRITONSERVER_InferenceTrace*>(this), activity, name,
          datatype, base, byte_size, shape, dim_count, memory_type,
          memory_type_id, userp_);
    }
  }

  // Release the trace. Call the trace release callback.
  void Release();

 private:
  const TRITONSERVER_InferenceTraceLevel level_;
  const TRITONSERVER_InferenceTraceMode mode_;
  const uint64_t id_;
  const uint64_t parent_id_;

  TRITONSERVER_InferenceTraceActivityFn_t activity_fn_;
  TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn_;
  TRITONSERVER_InferenceTraceReleaseFn_t release_fn_;
  void* userp_;

  std::string model_name_;
  int64_t model_version_;
  std::string request_id_;

  // Maintain next id statically so that trace id is unique even
  // across traces
  static std::atomic<uint64_t> next_id_;

#ifndef _WIN32
  otel_trace_api::Tracer* otel_tracer_{nullptr};
  opentelemetry::nostd::shared_ptr<otel_trace_api::Span> trace_span_{nullptr};
  opentelemetry::context::Context otel_context_{
      opentelemetry::context::RuntimeContext::GetCurrent()};
#endif
};

//
// InferenceTraceProxy
//
// Object attached as shared_ptr to InferenceRequest and
// InferenceResponse(s) being traced as part of a single inference
// request.
//
class InferenceTraceProxy {
 public:
  InferenceTraceProxy(InferenceTrace* trace) : trace_(trace) {}
  ~InferenceTraceProxy() { trace_->Release(); }
  int64_t Id() const { return trace_->Id(); }
  int64_t ParentId() const { return trace_->ParentId(); }

  const std::string& ModelName() const { return trace_->ModelName(); }
  const std::string& RequestId() const { return trace_->RequestId(); }
  const TRITONSERVER_InferenceTraceMode& TraceMode() const
  {
    return trace_->TraceMode();
  }

  int64_t ModelVersion() const { return trace_->ModelVersion(); }
  void SetModelName(const std::string& n) { trace_->SetModelName(n); }
  void SetRequestId(const std::string& n) { trace_->SetRequestId(n); }
  void SetModelVersion(int64_t v) { trace_->SetModelVersion(v); }

#ifndef _WIN32
  otel_trace_api::Tracer* OpenTelemetryTracer()
  {
    return trace_->OpenTelemetryTracer();
  }
  opentelemetry::nostd::shared_ptr<otel_trace_api::Span> OpenTelemetrySpan()
  {
    return trace_->OpenTelemetrySpan();
  }
  const std::chrono::nanoseconds TimeOffset() { return trace_->TimeOffset(); }
  opentelemetry::context::Context& GetOpenTelemetryContext()
  {
    return trace_->GetOpenTelemetryContext();
  }
  void SetOpenTelemetryContext(opentelemetry::context::Context& ctxt)
  {
    trace_->SetOpenTelemetryContext(ctxt);
  }
  void SetOpenTelemetryTracer(otel_trace_api::Tracer* otel_tracer)
  {
    trace_->SetOpenTelemetryTracer(otel_tracer);
  }
  opentelemetry::nostd::shared_ptr<otel_trace_api::Span> InitSpan(
      std::string name, const otel_common::SystemTimestamp& timestamp_ns,
      const uint64_t& raw_timestamp_ns)
  {
    return trace_->InitSpan(name, timestamp_ns, raw_timestamp_ns);
  }
  void EndActiveSpan(const uint64_t& raw_timestamp_ns)
  {
    trace_->EndActiveSpan(raw_timestamp_ns);
  }
#endif

  void Report(
      const TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
  {
    trace_->Report(activity, timestamp_ns);
  }

  void ReportNow(const TRITONSERVER_InferenceTraceActivity activity)
  {
    trace_->ReportNow(activity);
  }

  void ReportTensor(
      const TRITONSERVER_InferenceTraceActivity activity, const char* name,
      TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
      const int64_t* shape, uint64_t dim_count,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
  {
    trace_->ReportTensor(
        activity, name, datatype, base, byte_size, shape, dim_count,
        memory_type, memory_type_id);
  }

  std::shared_ptr<InferenceTraceProxy> SpawnChildTrace();

 private:
  InferenceTrace* trace_;
};

#endif  // TRITON_ENABLE_TRACING

//
// Macros to generate trace activity
//
#ifdef TRITON_ENABLE_TRACING
#define INFER_TRACE_ACTIVITY(T, A, TS_NS) \
  {                                       \
    const auto& trace = (T);              \
    const auto ts_ns = (TS_NS);           \
    if (trace != nullptr) {               \
      trace->Report(A, ts_ns);            \
    }                                     \
  }
#define INFER_TRACE_ACTIVITY_NOW(T, A) \
  {                                    \
    const auto& trace = (T);           \
    if (trace != nullptr) {            \
      trace->ReportNow(A);             \
    }                                  \
  }
#define INFER_TRACE_TENSOR_ACTIVITY(T, A, N, D, BA, BY, S, DI, MT, MTI) \
  {                                                                     \
    const auto& trace = (T);                                            \
    if (trace != nullptr) {                                             \
      trace->ReportTensor(A, N, D, BA, BY, S, DI, MT, MTI);             \
    }                                                                   \
  }
#else
#define INFER_TRACE_ACTIVITY(T, A, TS_NS)
#define INFER_TRACE_ACTIVITY_NOW(T, A)
#define INFER_TRACE_TENSOR_ACTIVITY(T, A, N, D, BA, BY, S, DI, MT, MTI)
#endif  // TRITON_ENABLE_TRACING
}}      // namespace triton::core
