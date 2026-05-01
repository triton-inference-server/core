// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
constexpr char kPyTorchLibTorchPlatformAlt[] = "libtorch";
constexpr char kPyTorchLibTorchFilename[] = "model.pt";
constexpr char kPyTorchBackend[] = "pytorch";

constexpr char kPyTorchAotiPlatform[] = "torch_aoti";
constexpr char kPyTorchAotiFilename[] = "model.pt2";
constexpr char kPyTorchAotiBackend[] = "pytorch";

constexpr char kPythonFilename[] = "model.py";
constexpr char kPythonBackend[] = "python";

#ifdef TRITON_ENABLE_ENSEMBLE
constexpr char kEnsemblePlatform[] = "ensemble";
constexpr uint64_t ENSEMBLE_CB_POOL_SIZE = 8u;
#endif  // TRITON_ENABLE_ENSEMBLE

constexpr char kTensorRTExecutionAccelerator[] = "tensorrt";
constexpr char kOpenVINOExecutionAccelerator[] = "openvino";
constexpr char kCUDAExecutionAccelerator[] = "cuda";
constexpr char kGPUIOExecutionAccelerator[] = "gpu_io";
constexpr char kAutoMixedPrecisionExecutionAccelerator[] =
    "auto_mixed_precision";

constexpr char kModelConfigPbTxt[] = "config.pbtxt";
constexpr char kPbTxtExtension[] = ".pbtxt";
constexpr char kModelConfigFolder[] = "configs";

constexpr char kMetricsLabelModelNamespace[] = "namespace";
constexpr char kMetricsLabelModelName[] = "model";
constexpr char kMetricsLabelModelVersion[] = "version";
constexpr char kMetricsLabelGpuUuid[] = "gpu_uuid";

constexpr char kWarmupDataFolder[] = "warmup";
constexpr char kInitialStateFolder[] = "initial_state";

// Metric names
constexpr char kPendingRequestMetric[] = "inf_pending_request_count";
constexpr char kModelLoadTimeMetric[] = "model_load_time";
constexpr char kFirstResponseHistogram[] = "first_response_histogram";

constexpr uint64_t NANOS_PER_SECOND = 1000000000;
constexpr uint64_t NANOS_PER_MILLIS = 1000000;
constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
constexpr uint64_t SEQUENCE_IDLE_DEFAULT_MICROSECONDS = 1000 * 1000;
constexpr size_t CUDA_IPC_STRUCT_SIZE = 64;

constexpr int kMutexTimeoutSeconds = 300;

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
