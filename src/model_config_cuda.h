// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdint.h>

#include "model_config.pb.h"

namespace triton { namespace core {

/// Get the CUDA stream priority for a given ModelPriority
/// \param priority The inference::ModelOptimizationPolicy::ModelPriority
/// priority. \param cuda_stream_priority Returns the CUDA stream priority.
/// \return The error status.
int GetCudaStreamPriority(
    inference::ModelOptimizationPolicy::ModelPriority priority);

}}  // namespace triton::core
