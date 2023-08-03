from typing import Any
from ._common import *
from enum import IntEnum
from collections.abc import Iterable, Union, Tuple, Mapping, Callable
import abc
import queue


class Tensor:

    def __init__(self) -> None:
        # tensor metadata
        self.data_type: DataType = None
        self.shape: Iterable[int] = None
        
        # buffer attributes..
        self.buffer_address: int = 0
        self.byte_size: int = 0
        self.memory_type: Tuple(MemoryType, int) = (MemoryType.CPU, 0)
        self.cuda_ipc_handle: Any = None
        self.host_policy_name: str = None

        # will be set internally after ResponseAllocator.alloc()
        # allocator may add additional attributes to manage the allocated
        # buffer
        self._allocator: ResponseAllocator = None
        # self._allocated_buffer ...

    def release(self) -> None:
        if self._allocator is not None:
            self._allocator.release(self)
            self._allocator = None

    def __del__(self) -> None:
        self.release()

    def __dlpack__(self, stream=None):
        raise "Not Implemented"

    def __dlpack_device__(self):
        raise "Not Implemented"


# [FIXME] 'response_allocator_userp' equivalent will be enclosed in the derived
# allocator instance. In other words, if the user wish to carry per-request info
# for the allocation, they may just add it to the attributes of the allocator
# instance
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

    @optional_callabck
    def start(self):
        # is TRITONSERVER_ResponseAllocatorStartFn_t
        raise "Not Implemented"

    # Call when Triton requests a buffer to store the tensor value, the
    # 'requested_tensor' will have buffer attribute related field set to
    # indicate the preferred configuration of the tensor. The allocator is
    # not required to satisfy any of the preference, but it should update the
    # fields in the returned tensor to reflect the attributes of the allocated
    # buffer.
    @abc.abstractmethod
    def alloc(self, tensor_name: str, requested_tensor: Tensor) -> Tensor:
        # is TRITONSERVER_InferenceTraceTensorActivityFn_t
        raise "Not Implemented"

    @abc.abstractmethod
    def release(self, tensor: Tensor):
        # is TRITONSERVER_ResponseAllocatorReleaseFn_t
        raise "Not Implemented"

    # [FIXME] TRITONSERVER_ResponseAllocatorBufferAttributesFn_t will be hidden
    # from user, BufferAttributes related will be set as part of "alloc" and
    # internally referred through 'buffer_userp'.

    # Query the Tensor to be returned if the same parameters are used for
    # alloc() function. This function may be called to configure
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


# [FIXME] the Triton API doesn't decouple Triton response and its output, a
# deletion of TRITONSERVER_InferenceResponse will call allocator release to
# release the output buffers.
# To ensure output tensor validity, either:
#   1. InferenceResponse is tied to TRITONSERVER_InferenceResponse, the user
#      must make sure InferenceResponse is valid while accessing outputs. So
#      the below tensor extraction is not allow:
#          def move_output():
#               res = []
#               for response in response_handle:
#                   res.append(response.outputs)
#               return res
#   2. Add another layer between TRITONSERVER_ResponseAllocator and
#      ResponseAllocator. The release callback of TRITONSERVER_ResponseAllocator
#      is actually no-op so we may delete TRITONSERVER_InferenceResponse once
#      InferenceResponse is constructed. the actual release is triggered as
#      part of the Tensor lifecycle
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


# Example implementation of allocator, and also use as current default.

# Deriving NumpyTensor in combination of the allocator to allow type check on
# the response's tensors before assuming derived class method is available, i.e.
# def read_output_as_numpy(response: InferenceResponse) -> Iterable[numpy.ndarray]:
#     res = []
#     for output in response.outputs:
#         if isinstance(output, NumpyTensor):
#             res.append(output.as_numpy())
#         else:
#             res.append(numpy.from_dlpack(output))
import numpy
class NumpyTensor(Tensor):
    def to_numpy_dtype(self):
        match self.data_type:
            case DataType.BOOL:
                return bool
            case DataType.UINT8:
                return numpy.uint8
            case DataType.UINT16:
                return numpy.uint16
            case DataType.UINT32:
                return numpy.uint32
            case DataType.UINT64:
                return numpy.uint64
            case DataType.INT8:
                return numpy.int8
            case DataType.INT16:
                return numpy.int16
            case DataType.INT32:
                return numpy.int32
            case DataType.INT64:
                return numpy.int64
            case DataType.FP16:
                return numpy.float16
            case DataType.FP32:
                return numpy.float32
            case DataType.FP64:
                return numpy.float64
        # Mark not currently supported because string type needs extra handling
        #    case DataType.BYTES:
        raise TritonErrorUnsupported("Unsupported type for numpy")

    def __init__(self) -> None:
        super().__init__()
        self._buffer: numpy.ndarray = None
    
    def as_numpy(self) -> numpy.ndarray:
        return self._buffer.view(self.to_numpy_dtype).reshape(self.shape)

class NumpyAllocator(ResponseAllocator):
    def alloc(self, tensor_name: str, requested_tensor: Tensor) -> Tensor:
        allocated_tensor = NumpyTensor()
        # extract info from requested_tensor
        allocated_tensor._buffer = numpy.empty(requested_tensor.byte_size, numpy.byte)
        # Update the fields for Triton use
        allocated_tensor.byte_size = requested_tensor.byte_size
        allocated_tensor.buffer_address = allocated_tensor._buffer.ctypes.data
        # other fields' default values are sufficient
        
    def release(self, tensor: Tensor):
        # Release will be done implicitly in 'tensor' GC
        pass
