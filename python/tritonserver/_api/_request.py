# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Class for sending inference requests to Triton Inference Server Models"""

from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass, field
from typing import Any, Optional

from tritonserver._api import _model
from tritonserver._api._datautils import CustomKeyErrorDict
from tritonserver._api._dlpack import DLDeviceType as DLDeviceType
from tritonserver._api._memorybuffer import DeviceOrMemoryType
from tritonserver._api._tensor import Tensor
from tritonserver._c.triton_bindings import InvalidArgumentError
from tritonserver._c.triton_bindings import TRITONSERVER_DataType as DataType
from tritonserver._c.triton_bindings import TRITONSERVER_InferenceRequest
from tritonserver._c.triton_bindings import TRITONSERVER_MemoryType as MemoryType
from tritonserver._c.triton_bindings import TRITONSERVER_Server


@dataclass
class InferenceRequest:

    """Dataclass representing an inference request.

    Inference request objects are created using Model factory
    methods. They contain input parameters and input data as well as
    configuration for response output memory allocation.

    See c:func:`TRITONSERVER_InferenceRequest` for more details

    Parameters
    ----------
    model : Model
        Model instance associated with the inference request.
    request_id : Optional[str], default None
        Unique identifier for the inference request.
    flags : int, default 0
        Flags indicating options for the inference request.
    correlation_id : Optional[Union[int, str]], default None
        Correlation ID associated with the inference request.
    priority : int, default 0
        Priority of the inference request.
    timeout : int, default 0
        Timeout for the inference request in microseconds.
    inputs : Dict[str, Union[Tensor, Any]], default {}
        Dictionary of input names and corresponding input tensors or data.
    parameters : Dict[str, Union[str, int, bool, float]], default {}
        Dictionary of parameters for the inference request.
    output_memory_type : Optional[DeviceOrMemoryType], default None
        output_memory_type : Optional[DeviceOrMemoryType], default
        None Type of memory to allocate for inference response
        output. If not provided memory type will be chosen based on
        backend / model preference with MemoryType.CPU as
        fallback. Memory type can be given as a string, MemoryType,
        tuple [MemoryType, memory_type__id], or tuple[DLDeviceType,
        device_id].
    response_queue : Optional[Union[queue.SimpleQueue, asyncio.Queue]], default None
        Queue for asynchronous handling of inference responses. If
        provided Inference responses will be added to the queue in
        addition to the response iterator. Must be queue.SimpleQueue
        for non asyncio requests and asyncio.Queue for asyncio
        requests.

    Examples
    --------

    # Creating a request explicitly

    >>> request = server.model("test").create_request()
    request = server.model("test").create_request()
    request.inputs["fp16_input"] = numpy.array([[1.0]]).astype(numpy.float16)
    for response in server.model("test_2").infer(request):
       print(numpy.from_dlpack(response.outputs["fp16_output"]))
    [[1.]]

    # Creating a request implicitly

    for response in server.model("test_2").infer(
        inputs={"fp16_input": numpy.array([[1.0]]).astype(numpy.float16)}
    ):
        print(numpy.from_dlpack(response.outputs["fp16_output"]))

    [[1.]]


    """

    model: _model.Model
    request_id: Optional[str] = None
    flags: int = 0
    correlation_id: Optional[int | str] = None
    priority: int = 0
    timeout: int = 0
    inputs: dict[str, Tensor | Any] = field(default_factory=dict)
    parameters: dict[str, str | int | bool | float] = field(default_factory=dict)
    output_memory_type: Optional[DeviceOrMemoryType] = None
    response_queue: Optional[queue.SimpleQueue | asyncio.Queue] = None
    _serialized_inputs: dict[str, Tensor] = field(init=False, default_factory=dict)
    _server: TRITONSERVER_Server = field(init=False)

    _set_parameter_methods = CustomKeyErrorDict(
        "Value",
        "Request Parameter",
        {
            str: TRITONSERVER_InferenceRequest.set_string_parameter,
            int: TRITONSERVER_InferenceRequest.set_int_parameter,
            float: TRITONSERVER_InferenceRequest.set_double_parameter,
            bool: TRITONSERVER_InferenceRequest.set_bool_parameter,
        },
    )

    def __post_init__(self):
        self._server = self.model._server

    def _release_request(self, _request, _flags, _user_object):
        pass

    def _add_inputs(self, request):
        for name, value in self.inputs.items():
            if not isinstance(value, Tensor):
                tensor = Tensor._from_object(value)
            else:
                tensor = value
            if tensor.data_type == DataType.BYTES:
                # to ensure lifetime of array
                self._serialized_inputs[name] = tensor
            request.add_input(name, tensor.data_type, tensor.shape)

            request.append_input_data_with_buffer_attributes(
                name,
                tensor.data_ptr,
                tensor.memory_buffer._create_tritonserver_buffer_attributes(),
            )

    def _set_parameters(self, request):
        for key, value in self.parameters.items():
            InferenceRequest._set_parameter_methods[type(value)](request, key, value)

    def _create_tritonserver_inference_request(self):
        request = TRITONSERVER_InferenceRequest(
            self._server, self.model.name, self.model.version
        )
        if self.request_id is not None:
            request.id = self.request_id
        request.priority_uint64 = self.priority
        request.timeout_microseconds = self.timeout
        if self.correlation_id is not None:
            if isinstance(self.correlation_id, int):
                request.correlation_id = self.correlation_id
            else:
                request.correlation_id_string = self.correlation_id
        request.flags = self.flags

        self._add_inputs(request)

        self._set_parameters(request)

        return request
