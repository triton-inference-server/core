# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tritonserver._c as triton_bindings
from tritonserver._api.wrapper import (
    InferenceRequest,
    InstanceGroupKind,
    LogFormat,
    MetricFamily,
    MetricFormat,
    MetricKind,
    Model,
    ModelBatchFlag,
    ModelControlMode,
    ModelIndexFlag,
    ModelTxnPropertyFlag,
    Options,
    RateLimitMode,
    Server,
)

exceptions = [
    triton_bindings.TritonError,
    triton_bindings.NotFoundError,
    triton_bindings.UnknownError,
    triton_bindings.InternalError,
    triton_bindings.InvalidArgumentError,
    triton_bindings.UnavailableError,
    triton_bindings.UnsupportedError,
    triton_bindings.AlreadyExistsError,
]


# Rename module for exceptions to simplify stack trace
for exception in exceptions:
    exception.__module__ = "tritonserver"
    globals()[exception.__name__] = exception

del triton_bindings
del exceptions
__all__ = ["Server", "Model", "InferenceRequest"]
