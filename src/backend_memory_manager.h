// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace triton { namespace core {

// Currently there is just a global memory manager that is used for
// all backends and which simply forwards requests on to the core
// memory manager.
struct TritonMemoryManager {};

}}  // namespace triton::core
