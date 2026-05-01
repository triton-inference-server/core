// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <map>
#include <thread>
#include <vector>

#include "status.h"
#include "triton/common/model_config.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

// Helper function to set memory policy and thread affinity on current thread
Status SetNumaConfigOnThread(
    const triton::common::HostPolicyCmdlineConfig& host_policy);

// Restrict the memory allocation to specific NUMA node.
Status SetNumaMemoryPolicy(
    const triton::common::HostPolicyCmdlineConfig& host_policy);

// Retrieve the node mask used to set memory policy for the current thread
Status GetNumaMemoryPolicyNodeMask(unsigned long* node_mask);

// Reset the memory allocation setting.
Status ResetNumaMemoryPolicy();

// Set a thread affinity to be on specific cpus.
Status SetNumaThreadAffinity(
    std::thread::native_handle_type thread,
    const triton::common::HostPolicyCmdlineConfig& host_policy);


}}  // namespace triton::core
