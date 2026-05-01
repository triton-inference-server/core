// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "model_config_cuda.h"

#include <cuda_runtime_api.h>

namespace triton { namespace core {

int
GetCudaStreamPriority(
    inference::ModelOptimizationPolicy::ModelPriority priority)
{
  // Default priority is 0
  int cuda_stream_priority = 0;

  int min, max;
  cudaError_t cuerr = cudaDeviceGetStreamPriorityRange(&min, &max);
  if ((cuerr != cudaErrorNoDevice) && (cuerr != cudaSuccess)) {
    return 0;
  }

  switch (priority) {
    case inference::ModelOptimizationPolicy::PRIORITY_MAX:
      cuda_stream_priority = max;
      break;
    case inference::ModelOptimizationPolicy::PRIORITY_MIN:
      cuda_stream_priority = min;
      break;
    default:
      cuda_stream_priority = 0;
      break;
  }

  return cuda_stream_priority;
}

}}  // namespace triton::core
