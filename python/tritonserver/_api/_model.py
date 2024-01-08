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

"""Class for interacting with Triton Inference Server Models"""
import asyncio
from typing import Annotated, Any, Optional, TypedDict

from tritonserver._c.triton_bindings import TRITONSERVER_Server


class Model:
    """Class for interacting with Triton Inference Server models

    Model objects are returned from server factory methods and allow
    users to query metadata and execute inference
    requests.

    """

    def __init__(
        self,
        _server: TRITONSERVER_Server,
        name: str,
        version: int = -1,
        state: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Initialize model

        Model objects should be obtainted from Server factory methods
        and not instantiated directly. See `Server` documentation.

        Parameters
        ----------
        server : TRITONSERVER_Server
            Underlying C binding server structure. Private.
        name : str
            model name
        version : int
            model version
        state : Optional[str]
            state of model (if known)
        reason : Optional[str]
            reason for model state (if known)

        Examples
        --------
        >>> server.model("test")
        server.model("test")
        {'name': 'test', 'version': -1, 'state': None}

        """

        self._server = _server
        self.name = name
        self.version = version
        self.state = state
        self.reason = reason

    def create_request(self, **kwargs: Unpack[InferenceRequest]) -> InferenceRequest:
        """Inference request factory method

        Return an inference request object that can be used with
        model.infer() ro model.async_infer()

        Parameters
        ----------

        kwargs : Unpack[InferenceRequest]
            Keyname arguments passed to `InferenceRequest` constructor. See
            `InferenceRequest` documentation for details.

        Returns
        -------
        InferenceRequest
            Inference request associated with this model

        Examples
        --------

        >>> server.model("test").create_request()
        server.model("test").create_request()
        InferenceRequest(model={'name': 'test', 'version': -1,
        'state': None},
        _server=<tritonserver._c.triton_bindings.TRITONSERVER_Server
        object at 0x7f5827156bf0>, request_id=None, flags=0,
        correlation_id=None, priority=0, timeout=0, inputs={},
        parameters={}, output_memory_type=None,
        output_memory_allocator=None, response_queue=None,
        _serialized_inputs={})

        """

        return InferenceRequest(model=self, _server=self._server, **kwargs)

    def async_infer(
        self,
        inference_request: Optional[InferenceRequest] = None,
        raise_on_error: bool = False,
        **kwargs: Unpack[InferenceRequest],
    ) -> AsyncResponseIterator:
        """Send an inference request to the model for execution

        Sends an inference request to the model. Responses are
        returned using an async.io compatible iterator. See
        c:func:`TRITONSERVER_ServerInferAsync`

        Parameters
        ----------
        inference_request : Optional[InferenceRequest]
            inference request object. If not provided inference
            request will be created using remaining key,value
            arguments.
        raise_on_error : bool
            if True iterator will raise an error on any response
            errors returned from the model. If False errors will be
            returned as part of the response object.
        kwargs : Unpack[InferenceRequest]
            If a request object is not provided, a new object will be
            created with remaining keyname arguments. See
            `InferenceRequest` documentation for valid arguments.


        Returns
        -------
        AsyncResponseIterator
            async.io compatible iterator

        Raises
        ------
        InvalidArgumentError
            if any invalid arguments are provided

        Examples
        --------
        Todo
        """

        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, asyncio.Queue)
        ):
            raise InvalidArgumentError(
                "asyncio.Queue must be used for async response iterator"
            )

        request = inference_request._create_TRITONSERVER_InferenceRequest()

        response_iterator = AsyncResponseIterator(
            self,
            self._server,
            request,
            inference_request.response_queue,
            raise_on_error,
        )

        response_allocator = _datautils.ResponseAllocator(
            inference_request.output_memory_allocator,
            inference_request.output_memory_type,
        ).create_TRITONSERVER_ResponseAllocator()

        request.set_response_callback(
            response_allocator, None, response_iterator._response_callback, None
        )

        self._server.infer_async(request)
        return response_iterator

    def infer(
        self,
        inference_request: Optional[InferenceRequest] = None,
        raise_on_error: bool = False,
        **kwargs: Unpack[InferenceRequest],
    ) -> ResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, queue.SimpleQueue)
        ):
            raise InvalidArgumentError(
                "queue.SimpleQueue must be used for response iterator"
            )

        request = inference_request._create_TRITONSERVER_InferenceRequest()
        response_iterator = ResponseIterator(
            self,
            self._server,
            request,
            inference_request.response_queue,
            raise_on_error,
        )
        response_allocator = _datautils.ResponseAllocator(
            inference_request.output_memory_allocator,
            inference_request.output_memory_type,
        ).create_TRITONSERVER_ResponseAllocator()

        request.set_response_callback(
            response_allocator, None, response_iterator._response_callback, None
        )

        self._server.infer_async(request)
        return response_iterator

    def metadata(self) -> dict[str, Any]:
        return json.loads(
            self._server.model_metadata(self.name, self.version).serialize_to_json()
        )

    def get_metadata(self) -> dict[str, Any]:
        return json.loads(
            self._server.model_metadata(self.name, self.version).serialize_to_json()
        )

    def is_ready(self) -> bool:
        return self._server.model_is_ready(self.name, self.version)

    def ready(self) -> bool:
        return self._server.model_is_ready(self.name, self.version)

    def batch_properties(self) -> ModelBatchFlag:
        flags, _ = self._server.model_batch_properties(self.name, self.version)
        return ModelBatchFlag(flags)

    def transaction_properties(self) -> ModelTxnPropertyFlag:
        txn_properties, _ = self._server.model_transaction_properties(
            self.name, self.version
        )
        return ModelTxnPropertyFlag(txn_properties)

    def statistics(self) -> dict[str, Any]:
        return json.loads(
            self._server.model_statistics(self.name, self.version).serialize_to_json()
        )

    def config(self, config_version: int = 1) -> dict[str, Any]:
        return json.loads(
            self._server.model_config(
                self.name, self.version, config_version
            ).serialize_to_json()
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s" % (
            {"name": self.name, "version": self.version, "state": self.state}
        )
