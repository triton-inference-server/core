from typing import Any
from ._common import *
from enum import IntEnum
from collections.abc import Iterable, Union, Tuple, Mapping, Callable
import abc
import queue


# [FIXME] implement dlpack interface, buffer attribute?
class Tensor:

    def __init__(self) -> None:
        # buffer attributes..
        # [FIXME] need to point out where exactly the buffer is managed
        self.buffer: int = 0
        self.byte_size: int = 0
        self.memory_type: Tuple(MemoryType, int) = (MemoryType.CPU, 0)
        # [FIXME] CUDA specific.. how to / should introduce CUDA dependency?
        self.cuda_ipc_handle: Any = None
        self.host_policy_name: str = None

    def __dlpack__(self, stream=None):
        raise "Not Implemented"

    def __dlpack_device__(self):
        raise "Not Implemented"


class ResponseAllocator(abc.ABC):
    _allocator: Any = None

    def __init__(self) -> None:
        super().__init__()
        # Create Triton allocator once and for all as
        # a Triton allocator is collection of callbacks and
        # can be reused across inference
        if type(self)._allocator is None:
            # TRITONSERVER_ResponseAllocatorNew
            raise "Not Implemented"

    # [FIXME] 'response_allocator_userp' can be enclosed in the derieved
    # allocator instance.
    @optional_callabck
    def start(self):
        # is TRITONSERVER_ResponseAllocatorStartFn_t
        raise "Not Implemented"

    # [FIXME] Should tensor be returned? Or a more primal form?
    # [FIXME] return 'buffer_userp'?
    @abc.abstractmethod
    def alloc(self, tensor_name: str,
              requested_tensor: Tensor) -> Tuple[Tensor, Any]:
        # is TRITONSERVER_InferenceTraceTensorActivityFn_t
        raise "Not Implemented"

    @abc.abstractmethod
    def release(self, tensor: Tensor, buffer_user_object: Any = None):
        # is TRITONSERVER_ResponseAllocatorReleaseFn_t
        raise "Not Implemented"

    @optional_callabck
    def last_alloc(self,
                   tensor_name: str,
                   user_object: Any = None,
                   buffer_user_object: Any = None) -> Tensor:
        # is TRITONSERVER_ResponseAllocatorBufferAttributesFn_t
        raise "Not Implemented"

    # [FIXME] pre-alloc? Query the Tensor to be returned if the same parameters
    # are used for alloc() function. This function may be called to configure
    # internal optimization in preparation for handling the allocated tensor.
    @optional_callabck
    def query(self, tensor: Tensor, tensor_name: str = None) -> Tensor:
        # is TRITONSERVER_ResponseAllocatorQueryFn_t
        raise "Not Implemented"


class RequestFlag(IntEnum):
    SEQUENCE_START: int = 1
    SEQUENCE_END: int = 2


# Release callback is hidden, user don't need to be exposed to
# Triton request detail.
# Variables should not be changed until "consumed", an callback may be
# provided to be invoked when the request is no longer in used
class InferenceRequest:

    def __init__(self,
                 name: str,
                 version: int = -1,
                 consumed_callback: Callable = None) -> None:
        self.name = name
        self.version = version
        self.consumed_callback = consumed_callback

        self.request_id: str = ""
        self.timeout_ms: int = 0
        self.priority: int = 0
        # Sequence..
        self.request_flag: int = 0
        self.correlation_id: Union[str, int] = 0

        self.inputs: Mapping[str, Union[Tensor, Any]] = {}
        self.requested_outputs: set = set()
        self.parameters: Mapping[str, Union[str, int, bool]] = {}

        self._in_use_count: int = 0

    def consumed(self) -> bool:
        return self._in_use_count == 0


# Returned from infer async, use as receiver of the response or
# exception that represents the response error. On each retrival of
# the response, the return value will be
# Union[InferenceResponse, TritonError, None]
# [FIXME] Note that response complete flag is hidden from user and it is
# currently reflected as the "end" state of the handle
class ResponseHandle(queue.Queue):

    def __init__(self) -> None:
        super().__init__()
        self._end = False

    def __iter__(self):
        while not self.exhausted():
            yield self.get()

    def exhausted(self) -> bool:
        return self._end and self.empty()


# [WIP] manage Triton response -> internally control output validity
class InferenceResponse:

    def __init__(self, name: str, version: int = -1) -> None:
        self.name = name
        self.version = version
        self.request_id: str = ""
        self.parameters: Mapping[str, Union[str, int, bool]] = {}

        # Currently the equivalent of
        # TRITONSERVER_InferenceResponseOutputClassificationLabel
        # will not be provided. If request, may provide 'top_k_labels'
        # function that is built on top of the C API.
        self.outputs: Mapping[str, Tensor] = {}
