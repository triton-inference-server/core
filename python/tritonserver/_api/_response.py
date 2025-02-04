# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Class for receiving inference responses to Triton Inference Server Models"""

from __future__ import annotations

import asyncio
import concurrent
import inspect
import queue
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy
from tritonserver._api import _model

if TYPE_CHECKING:
    from tritonserver._api._model import Model

from tritonserver._api._dlpack import DLDeviceType as DLDeviceType
from tritonserver._api._logging import LogMessage
from tritonserver._api._memorybuffer import DeviceOrMemoryType, MemoryBuffer
from tritonserver._api._tensor import Tensor
from tritonserver._c.triton_bindings import (
    InternalError,
    InvalidArgumentError,
    TritonError,
)
from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel
from tritonserver._c.triton_bindings import TRITONSERVER_MemoryType as MemoryType
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ResponseCompleteFlag,
    TRITONSERVER_Server,
)


@dataclass
class InferenceResponse:
    """Dataclass representing an inference response.

    Inference response objects are returned from response iterators
    which are in turn returned from model inference methods. They
    contain output data, output parameters, any potential errors
    reported and a flag to indicate if the response is the final one
    for a request.

    See c:func:`TRITONSERVER_InferenceResponse` for more details

    Parameters
    ----------
    model : Model
        Model instance associated with the response.
    request_id : Optional[str], default None
        Unique identifier for the inference request (if provided)
    parameters : dict[str, str | int | bool], default {}
        Additional parameters associated with the response.
    outputs : dict [str, Tensor], default {}
        Output tensors for the inference.
    error : Optional[TritonError], default None
        Error (if any) that occurred in the processing of the request.
    classification_label : Optional[str], default None
        Classification label associated with the inference. Not currently supported.
    final : bool, default False
        Flag indicating if the response is final

    """

    model: _model.Model
    request_id: Optional[str] = None
    parameters: dict[str, str | int | bool] = field(default_factory=dict)
    outputs: dict[str, Tensor] = field(default_factory=dict)
    error: Optional[TritonError] = None
    classification_label: Optional[str] = None
    final: bool = False

    @staticmethod
    def _from_tritonserver_inference_response(
        model: _model.Model,
        response,
        flags: TRITONSERVER_ResponseCompleteFlag,
        output_memory_type: Optional[DeviceOrMemoryType] = None,
    ):
        result = InferenceResponse(
            model,
            final=(flags == TRITONSERVER_ResponseCompleteFlag.FINAL),
        )

        try:
            if response is None:
                return result

            try:
                response.throw_if_response_error()
            except TritonError as error:
                error.args += (result,)
                result.error = error

            name, version = response.model
            result.model.name = name
            result.model.version = version
            result.request_id = response.id
            parameters = {}
            for parameter_index in range(response.parameter_count):
                name, type_, value = response.parameter(parameter_index)
                parameters[name] = value
            result.parameters = parameters
            outputs = {}
            for output_index in range(response.output_count):
                (
                    name,
                    data_type,
                    shape,
                    data_ptr,
                    byte_size,
                    memory_type,
                    memory_type_id,
                ) = response.output(output_index)
                memory_buffer = MemoryBuffer(
                    data_ptr=data_ptr,
                    memory_type=memory_type,
                    memory_type_id=memory_type_id,
                    size=byte_size,
                    owner=response,
                )
                tensor = Tensor(data_type, shape, memory_buffer)
                outputs[name] = tensor
            result.outputs = outputs
        except Exception as e:
            error = InternalError(f"Unexpected error in creating response object: {e}")
            error.args += (result,)
            result.error = error

        # TODO: [DLIS-7824] Allocate the requested output memory type directly in C++.
        if output_memory_type is not None:
            try:
                outputs = {}
                for name, tensor in result.outputs.items():
                    outputs[name] = tensor.to_device(output_memory_type)
                result.outputs = outputs
            except Exception as e:
                raise InvalidArgumentError(
                    f"Memory type {output_memory_type} not supported: {e}"
                )

        # TODO: support classification
        # values["classification_label"] = response.output_classification_label()

        return result


class AsyncResponseIterator:
    def __init__(self, model, request, inference_request):
        self._model = model
        self._request = request
        self._inference_request = inference_request
        self._complete = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._complete:
            raise StopAsyncIteration

        # The binding could be improved by returning an awaitable object, but it is
        # easier for now to pass in an async future object to be set by the binding when
        # fetching responses.
        future = asyncio.get_running_loop().create_future()
        self._request.get_next_response(future)
        response, flags = await future

        response = InferenceResponse._from_tritonserver_inference_response(
            self._model,
            response,
            flags,
            self._inference_request.output_memory_type,
        )
        self._complete = response.final

        return response


class ResponseIterator:
    def __init__(self, model, request, inference_request):
        self._model = model
        self._request = request
        self._inference_request = inference_request
        self._complete = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._complete:
            raise StopIteration

        # The binding could be improved by releasing the GIL and block, but it is easier
        # for now to pass in an future object to be set by the binding when fetching
        # responses.
        future = concurrent.futures.Future()
        self._request.get_next_response(future)
        response, flags = future.result()

        response = InferenceResponse._from_tritonserver_inference_response(
            self._model,
            response,
            flags,
            self._inference_request.output_memory_type,
        )
        self._complete = response.final

        return response
