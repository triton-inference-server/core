// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#define _COMPILING_TRITONSERVER 1
#define _COMPILING_TRITONBACKEND 1
#define _COMPILING_TRITONREPOAGENT 1
#define _COMPILING_TRITONCACHE 1

#include "triton/core/tritonbackend.h"
#include "triton/core/tritoncache.h"
#include "triton/core/tritonrepoagent.h"
#include "triton/core/tritonserver.h"

#undef _COMPILING_TRITONSERVER
#undef _COMPILING_TRITONBACKEND
#undef _COMPILING_TRITONREPOAGENT
#undef _COMPILING_TRITONCACHE
