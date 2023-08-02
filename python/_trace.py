from enum import IntEnum, Enum
from ._common import *
from typing import Iterable, Any, Union
import abc


class TraceLevel(IntEnum):
    TIMESTAMPS: int = 4
    TENSORS: int = 8

    def __str__(self) -> str:
        # TRITONSERVER_InferenceTraceLevelString
        raise "Not Implemented"


# Activity related to timeline tracing
class TraceActivity(Enum):
    REQUEST_START: int = 0
    QUEUE_START: int = 1
    COMPUTE_START: int = 2
    COMPUTE_INPUT_END: int = 3
    COMPUTE_OUTPUT_START: int = 4
    COMPUTE_END: int = 5
    REQUEST_END: int = 6

    def __str__(self) -> str:
        # TRITONSERVER_InferenceTraceActivityString
        raise "Not Implemented"


# Activity related to tensor tracing
class TraceTensorActivity(Enum):
    TENSOR_QUEUE_INPUT: int = 7
    TENSOR_BACKEND_INPUT: int = 8
    TENSOR_BACKEND_OUTPUT: int = 9

    def __str__(self) -> str:
        # TRITONSERVER_InferenceTraceActivityString
        raise "Not Implemented"


# Trace object that Triton will provide
class Trace:

    @property
    def id(self) -> int:
        pass

    @property
    def parent_id(self) -> int:
        pass

    @property
    def request_id(self) -> int:
        pass

    @property
    def model_name(self) -> str:
        pass

    @property
    def model_version(self) -> int:
        pass


class TraceReportor(abc.ABC):

    def __init__(self, level: Union[TraceLevel, int]) -> None:
        super().__init__()
        self._level = int(level)

    # Callback that Triton will invoke to report timestamp of the
    # given activity. Note that this invocation does not imply that
    # the activity is ocurring at the moment.
    @abc.abstractmethod
    def trace_activity(self, activity: TraceActivity, timestamp: int,
                       trace: Trace):
        # is TRITONSERVER_InferenceTraceActivityFn_t
        raise "Not Implemented"

    # [FIXME] optional
    # [WIP] Tensor abstraction
    @abc.abstractmethod
    def trace_tensor(self, activity: TraceTensorActivity, name: str,
                     data_type: DataType, base: int, byte_size: int,
                     shape: Iterable[int], memory_type: MemoryType,
                     memory_type_id: int, trace: Trace):
        # is TRITONSERVER_InferenceTraceTensorActivityFn_t
        raise "Not Implemented"

    @abc.abstractmethod
    def trace_release(self, trace: Trace):
        # is TRITONSERVER_InferenceTraceReleaseFn_t
        raise "Not Implemented"


# Usage:
# inherit from TraceReportor to the abstract functions,
# intiantiate the reportor() object with the desire configuration
# Detail:
# obj(TraceReportor) ->
#     TraceTensorNew(&trace, obj._level, obj.trace_activity,
#         obj.trace_tensor,
#         [](){
#             obj.trace_release;
#             TraceDelete();
#             // assumping root trace is always released last..
#             if trace.parent_id == -1 {
#                 obj.ref_dec();
#             }}, obj);
#     obj.ref_inc()
