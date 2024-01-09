# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import inspect
import queue
from dataclasses import dataclass, field
from typing import Optional

from _model import Model
from tritonserver._c.triton_bindings import InternalError, TRITONSERVER_InferenceRequest
from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel
from tritonserver._c.triton_bindings import TRITONSERVER_LogMessage as LogMessage
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ResponseCompleteFlag,
    TRITONSERVER_Server,
)


class AsyncResponseIterator:

    """Asyncio compatible response iterator

    Response iterators are returned from model inference methods and
    allow users to process inference responses in the order they were
    received for a request.

    """

    def __init__(
        self,
        _model: Model,
        _server: TRITONSERVER_Server,
        _request: TRITONSERVER_InferenceRequest,
        _user_queue: Optional[asyncio.Queue] = None,
        _raise_on_error: bool = False,
        _loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Initialize AsyncResponseIterator

        AsyncResponseIterator objects are obtained from Model inference
        methods and not instantiated directly. See `Model` documentation.

        Parameters
        ----------
        _model : Model
            model associated with inference request
        _server : TRITONSERVER_Server
            Underlying C binding server object. Private.
        _request : TRITONSERVER_InferenceRequest
            Underlying C binding TRITONSERVER_InferenceRequest
            object. Private.
        _user_queue : Optional[asyncio.Queue]
            Optional user queue for responses in addition to internal
            iterator queue.
        _raise_on_error : bool
            if True response errors will be raised as exceptions.
        _loop : Optional[asyncio.AbstractEventLoop]
            asyncio loop object


        Examples
        --------

        Todo

        """

        self._server = _server
        if _loop is None:
            _loop = asyncio.get_running_loop()
        self._loop = _loop
        self._queue = asyncio.Queue()
        self._user_queue = _user_queue
        self._complete = False
        self._request = _request
        self._model = _model
        self._raise_on_error = _raise_on_error

    def __aiter__(self) -> AsyncResponseIterator:
        return self

    async def __anext__(self):
        """Returns the next response received for a request

        Returns the next response received for a request as an
        awaitable object.

        Raises
        ------
        response.error
            If raise_on_error is set to True, response errors are
            raised as exceptions
        StopAsyncIteration
            Indicates all responses for a request have been received.
            Final responses may or may not include outputs and must be
            checked.
        Examples
        --------
        Todo

        """

        if self._complete:
            raise StopAsyncIteration
        response = await self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self) -> None:
        """Cancel an inflight request

        Cancels an in-flight request. Cancellation is handled on a
        best effort basis and may not prevent execution of a request
        if it is already started or completed.

        See c:func:`TRITONSERVER_ServerInferenceRequestCancel`

        Examples
        --------

        responses = server.model("test").infer(inputs={"text_input":["hello"]})

        responses.cancel()

        """

        if self._request is not None:
            self._request.cancel()

    def _response_callback(self, response, flags, unused):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            response = InferenceResponse._from_TRITONSERVER_InferenceResponse(
                self._model, self._server, self._request, response, flags
            )
            asyncio.run_coroutine_threadsafe(self._queue.put(response), self._loop)
            if self._user_queue is not None:
                asyncio.run_coroutine_threadsafe(
                    self._user_queue.put(response), self._loop
                )
            if flags == TRITONSERVER_ResponseCompleteFlag.FINAL:
                del self._request
                self._request = None
        except Exception as e:
            current_frame = inspect.currentframe()
            if current_frame is not None:
                line_number = current_frame.f_lineno
            else:
                line_number = -1
            LogMessage(
                LogLevel.ERROR,
                __file__,
                line_number,
                str(e),
            )


class ResponseIterator:
    """Response iterator

    Response iterators are returned from model inference methods and
    allow users to process inference responses in the order they were
    received for a request.

    """

    def __init__(
        self,
        _model: Model,
        _server: TRITONSERVER_Server,
        _request: TRITONSERVER_InferenceRequest,
        _user_queue: Optional[queue.SimpleQueue] = None,
        _raise_on_error: bool = False,
    ):
        """Initialize ResponseIterator

        ResponseIterator objects are obtained from Model inference
        methods and not instantiated directly. See `Model` documentation.

        Parameters
        ----------
        _model : Model
            model associated with inference request
        _server : TRITONSERVER_Server
            Underlying C binding server object. Private.
        _request : TRITONSERVER_InferenceRequest
            Underlying C binding TRITONSERVER_InferenceRequest
            object. Private.
        _user_queue : Optional[asyncio.Queue]
            Optional user queue for responses in addition to internal
            iterator queue.
        _raise_on_error : bool
            if True response errors will be raised as exceptions.

        Examples
        --------

        Todo

        """

        self._queue = queue.SimpleQueue()
        self._user_queue = _user_queue
        self._server = _server
        self._complete = False
        self._request = _request
        self._model = _model
        self._raise_on_error = _raise_on_error

    def __iter__(self) -> ResponseIterator:
        return self

    def __next__(self):
        """Returns the next response received for a request

        Raises
        ------
        response.error
            If raise_on_error is set to True, response errors are
            raised as exceptions
        StopIteration
            Indicates all responses for a request have been received.
            Final responses may or may not include outputs and must be
            checked.
        Examples
        --------
        Todo

        """

        if self._complete:
            raise StopIteration
        response = self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self):
        """Cancel an inflight request

        Cancels an in-flight request. Cancellation is handled on a
        best effort basis and may not prevent execution of a request
        if it is already started or completed.

        See c:func:`TRITONSERVER_ServerInferenceRequestCancel`

        Examples
        --------

        responses = server.model("test").infer(inputs={"text_input":["hello"]})

        responses.cancel()

        """
        if self._request is not None:
            self._request.cancel()

    def _response_callback(self, response, flags, unused):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            response = InferenceResponse._from_TRITONSERVER_InferenceResponse(
                self._model, self._server, self._request, response, flags
            )
            self._queue.put(response)
            if self._user_queue is not None:
                self._user_queue.put(response)
            if flags == TRITONSERVER_ResponseCompleteFlag.FINAL:
                del self._request
                self._request = None
        except Exception as e:
            current_frame = inspect.currentframe()
            if current_frame is not None:
                line_number = current_frame.f_lineno
            else:
                line_number = -1

            LogMessage(
                LogLevel.ERROR,
                __file__,
                line_number,
                str(e),
            )


@dataclass
class InferenceResponse:
    model: Model
    _server: _triton_bindings.TRITONSERVER_Server
    request_id: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: Optional[_triton_bindings.TritonError] = None
    classification_label: Optional[str] = None
    final: bool = False

    @staticmethod
    def _from_TRITONSERVER_InferenceResponse(
        model: Model,
        server: _triton_bindings.TRITONSERVER_Server,
        request: _triton_bindings.TRITONSERVER_InferenceRequest,
        response,
        flags: _triton_bindings.TRITONSERVER_ResponseCompleteFlag,
    ):
        values: dict = {
            "_server": server,
            "model": model,
            "request_id": request.id,
            "final": flags == _triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL,
        }

        try:
            if response is None:
                return InferenceResponse(**values)

            try:
                response.throw_if_response_error()
            except _triton_bindings.TritonError as error:
                error.args += tuple(values.items())
                values["error"] = error

            name, version = response.model
            values["model"] = Model(server, name, version)
            values["request_id"] = response.id
            parameters = {}
            for parameter_index in range(response.parameter_count):
                name, type_, value = response.parameter(parameter_index)
                parameters[name] = value
            values["parameters"] = parameters
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
                    memory_buffer,
                ) = response.output(output_index)
                tensor = _datautils.Tensor(data_type, shape, memory_buffer)

                outputs[name] = tensor
            values["outputs"] = outputs
        except Exception as e:
            error = InternalError(f"Unexpected error in creating response object: {e}")
            error.args += tuple(values.items())
            values["error"] = error

        # values["classification_label"] = response.output_classification_label()

        return InferenceResponse(**values)
