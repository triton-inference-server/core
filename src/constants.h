// Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdint.h>

namespace triton { namespace core {

constexpr char kInferHeaderContentLengthHTTPHeader[] =
    "Inference-Header-Content-Length";
constexpr char kAcceptEncodingHTTPHeader[] = "Accept-Encoding";
constexpr char kContentEncodingHTTPHeader[] = "Content-Encoding";
constexpr char kContentTypeHeader[] = "Content-Type";
constexpr char kContentLengthHeader[] = "Content-Length";

constexpr char kTensorFlowGraphDefPlatform[] = "tensorflow_graphdef";
constexpr char kTensorFlowSavedModelPlatform[] = "tensorflow_savedmodel";
constexpr char kTensorFlowGraphDefFilename[] = "model.graphdef";
constexpr char kTensorFlowSavedModelFilename[] = "model.savedmodel";
constexpr char kTensorFlowBackend[] = "tensorflow";

constexpr char kTensorRTPlanPlatform[] = "tensorrt_plan";
constexpr char kTensorRTPlanFilename[] = "model.plan";
constexpr char kTensorRTBackend[] = "tensorrt";

constexpr char kOnnxRuntimeOnnxPlatform[] = "onnxruntime_onnx";
constexpr char kOnnxRuntimeOnnxFilename[] = "model.onnx";
constexpr char kOnnxRuntimeBackend[] = "onnxruntime";

constexpr char kOpenVINORuntimeOpenVINOFilename[] = "model.xml";
constexpr char kOpenVINORuntimeBackend[] = "openvino";

constexpr char kPyTorchLibTorchPlatform[] = "pytorch_libtorch";
constexpr char kPyTorchLibTorchFilename[] = "model.pt";
constexpr char kPyTorchBackend[] = "pytorch";

constexpr char kPythonFilename[] = "model.py";
constexpr char kPythonBackend[] = "python";

#ifdef TRITON_ENABLE_ENSEMBLE
constexpr char kEnsemblePlatform[] = "ensemble";
#endif  // TRITON_ENABLE_ENSEMBLE

constexpr char kTensorRTExecutionAccelerator[] = "tensorrt";
constexpr char kOpenVINOExecutionAccelerator[] = "openvino";
constexpr char kGPUIOExecutionAccelerator[] = "gpu_io";
constexpr char kAutoMixedPrecisionExecutionAccelerator[] =
    "auto_mixed_precision";

constexpr char kModelConfigPbTxt[] = "config.pbtxt";

constexpr char kMetricsLabelModelName[] = "model";
constexpr char kMetricsLabelModelVersion[] = "version";
constexpr char kMetricsLabelGpuUuid[] = "gpu_uuid";

constexpr char kWarmupDataFolder[] = "warmup";
constexpr char kInitialStateFolder[] = "initial_state";

// Metric names
constexpr char kPendingRequestMetric[] = "inf_pending_request_count";

constexpr uint64_t NANOS_PER_SECOND = 1000000000;
constexpr uint64_t NANOS_PER_MILLIS = 1000000;
constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
constexpr uint64_t SEQUENCE_IDLE_DEFAULT_MICROSECONDS = 1000 * 1000;
constexpr size_t STRING_CORRELATION_ID_MAX_LENGTH_BYTES = 128;
constexpr size_t CUDA_IPC_STRUCT_SIZE = 64;

#ifdef TRITON_ENABLE_METRICS
// MetricModelReporter expects a device ID for GPUs, but we reuse this device
// ID for other metrics as well such as for CPU and Response Cache metrics
constexpr int METRIC_REPORTER_ID_CPU = -1;
constexpr int METRIC_REPORTER_ID_UTILITY = -2;
#endif

// Note: This can be replaced with std::byte starting in c++17
using Byte = unsigned char;

#define TIMESPEC_TO_NANOS(TS) \
  ((TS).tv_sec * triton::core::NANOS_PER_SECOND + (TS).tv_nsec)
#define TIMESPEC_TO_MILLIS(TS) \
  (TIMESPEC_TO_NANOS(TS) / triton::core::NANOS_PER_MILLIS)

#define DISALLOW_MOVE(TypeName) TypeName(Context&& o) = delete;
#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete;
#define DISALLOW_ASSIGN(TypeName) void operator=(const TypeName&) = delete;
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  DISALLOW_COPY(TypeName)                  \
  DISALLOW_ASSIGN(TypeName)

}}  // namespace triton::core
