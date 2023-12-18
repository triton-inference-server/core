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

"""In process Python API for Triton Inference Server."""

import tritonserver._c as _triton_bindings
from tritonserver._api.wrapper import InferenceRequest as InferenceRequest
from tritonserver._api.wrapper import InstanceGroupKind as InstanceGroupKind
from tritonserver._api.wrapper import LogFormat as LogFormat
from tritonserver._api.wrapper import Metric as Metric
from tritonserver._api.wrapper import MetricFamily as MetricFamily
from tritonserver._api.wrapper import MetricFormat as MetricFormat
from tritonserver._api.wrapper import MetricKind as MetricKind
from tritonserver._api.wrapper import Model as Model
from tritonserver._api.wrapper import ModelBatchFlag as ModelBatchFlag
from tritonserver._api.wrapper import ModelControlMode as ModelControlMode
from tritonserver._api.wrapper import ModelIndexFlag as ModelIndexFlag
from tritonserver._api.wrapper import ModelTxnPropertyFlag as ModelTxnPropertyFlag
from tritonserver._api.wrapper import Options as Options
from tritonserver._api.wrapper import RateLimitMode as RateLimitMode
from tritonserver._api.wrapper import Server as Server
from tritonserver._c import AlreadyExistsError as AlreadyExistsError
from tritonserver._c import InternalError as InternalError
from tritonserver._c import InvalidArgumentError as InvalidArgumentError
from tritonserver._c import NotFoundError as NotFoundError
from tritonserver._c import TritonError as TritonError
from tritonserver._c import UnavailableError as UnavailableError
from tritonserver._c import UnknownError as UnknownError
from tritonserver._c import UnsupportedError as UnsupportedError

_exceptions = [
    _triton_bindings.TritonError,
    _triton_bindings.NotFoundError,
    _triton_bindings.UnknownError,
    _triton_bindings.InternalError,
    _triton_bindings.InvalidArgumentError,
    _triton_bindings.UnavailableError,
    _triton_bindings.AlreadyExistsError,
    _triton_bindings.UnsupportedError,
]


# Rename module for exceptions to simplify stack trace
for exception in _exceptions:
    exception.__module__ = "tritonserver"
    globals()[exception.__name__] = exception

del _triton_bindings
del _exceptions
__all__ = []
