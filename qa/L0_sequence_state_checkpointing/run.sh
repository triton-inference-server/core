#!/usr/bin/env bash
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${TRITON_CORE_BUILD:?set TRITON_CORE_BUILD}"
: "${TRITON_BACKEND_DIR:?set TRITON_BACKEND_DIR}"
: "${ONNXRUNTIME_LIB_DIR:?set ONNXRUNTIME_LIB_DIR}"
: "${TEST_BUILD_DIR:?set TEST_BUILD_DIR}"
PYTHON=${PYTHON:-python3}

mkdir -p "$TEST_BUILD_DIR"
RUN_ROOT=$(mktemp -d "$TEST_BUILD_DIR/run.XXXXXX")
"$PYTHON" "$ROOT/qa/L0_sequence_state_checkpointing/models.py" "$RUN_ROOT/models"
g++ -std=c++20 -O2 -pthread "$ROOT/qa/L0_sequence_state_checkpointing/integration.cc" -I"$ROOT/include" -L"$TRITON_CORE_BUILD" -Wl,-rpath,"$TRITON_CORE_BUILD" -ltritonserver -o "$RUN_ROOT/integration"

export LD_LIBRARY_PATH="$TRITON_CORE_BUILD:$ONNXRUNTIME_LIB_DIR"
"$RUN_ROOT/integration" "$RUN_ROOT/models" "$TRITON_BACKEND_DIR" "$RUN_ROOT/checkpoints" tests
"$RUN_ROOT/integration" "$RUN_ROOT/models" "$TRITON_BACKEND_DIR" "$RUN_ROOT/restart" restart-save
"$RUN_ROOT/integration" "$RUN_ROOT/models" "$TRITON_BACKEND_DIR" "$RUN_ROOT/restart" restart-load
