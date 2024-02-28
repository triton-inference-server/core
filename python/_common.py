
from typing import Tuple
from enum import Enum

def version() -> Tuple[int, int]:
    # TRITONSERVER_ApiVersion
    raise "Not Implemented"

class TritonError(Exception):
    pass

# Dedicated exception type for different error code
class TritonErrorUnknown(TritonError):
    pass
class TritonErrorInternal(TritonError):
    pass
class TritonErrorNotFound(TritonError):
    pass
class TritonErrorInvalidArgument(TritonError):
    pass
class TritonErrorUnavailable(TritonError):
    pass
class TritonErrorUnsupported(TritonError):
    pass
class TritonErrorAlreadyExists(TritonError):
    pass

class InstanceGroupKind(Enum):
    AUTO: int = 0
    CPU: int = 1
    GPU: int = 2
    MODEL: int = 3

    def __str__(self) -> str:
        # TRITONSERVER_InstanceGroupKindString
        raise "Not implemented"
        return "mock"

class MemoryType(Enum):
    CPU: int = 0
    CPU_PINNED: int = 1
    GPU: int = 2

    def __str__(self) -> str:
        # TRITONSERVER_MemoryTypeString
        raise "Not implemented"
        return "mock"

class DataType(Enum):
    INVALID: int = 0
    BOOL: int = 1
    UINT8: int = 2
    UINT16: int = 3
    UINT32: int = 4
    UINT64: int = 5
    INT8: int = 6
    INT16: int = 7
    INT32: int = 8
    INT64: int = 9
    FP16: int = 10
    FP32: int = 11
    FP64: int = 12
    BYTES: int = 13
    BF16: int = 14

    # 1-1 mapping from DataType to TRITONSERVER_DataType
    @classmethod
    def str_to_type(cls, type : str) -> "DataType":
        # TRITONSERVER_StringToDataType
        mock_triton_type = 1
        raise "Not implemented"
        return DataType(mock_triton_type)
    
    def __str__(self) -> str:
        # TRITONSERVER_DataTypeString
        raise "Not implemented"
        return "mock"
    
    def byte_size(self) -> int:
        # TRITONSERVER_DataTypeByteSize
        raise "Not implemented"
        return 0

# [FIXME] Triton API reflection of the error object, not expose to user.
# User should interact with TritonException above.
class Code(Enum):
    UNKNOWN: int = 0
    INTERNAL: int = 1
    NOT_FOUND: int = 2
    INVALID_ARG: int = 3
    UNAVAILABLE: int = 4
    UNSUPPORTED: int = 5
    ALREADY_EXISTS: int = 6

class Error:
    def __init__(self, triton_error) -> None:
        self._err = triton_error
        pass
    def __del__(self):
        # TRITONSERVER_ErrorDelete
        raise "Not implemented"

    @property
    def code(self) -> Code:
        # TRITONSERVER_ErrorCode
        raise "Not implemented"
    
    @property
    def message(self) -> str:
        # TRITONSERVER_ErrorMessage
        raise "Not implemented"

def raise_if_triton_error(err: Error = None):
    if err is None:
        return
    match err.code:
        case Code.UNKNOWN:
            raise TritonErrorUnknown(err.message)
        case Code.INTERNAL:
            raise TritonErrorInternal(err.message)
        case Code.NOT_FOUND:
            raise TritonErrorNotFound(err.message)
        case Code.INVALID_ARG:
            raise TritonErrorInvalidArgument(err.message)
        case Code.UNAVAILABLE:
            raise TritonErrorUnavailable(err.message)
        case Code.UNSUPPORTED:
            raise TritonErrorUnsupported(err.message)
        case Code.ALREADY_EXISTS:
            raise TritonErrorAlreadyExists(err.message)

def optional_callabck(func):
    # additional function attribute to be checked when
    # this module interacts with Triton in-process API.
    func._triton_callback_not_provided = True
    return func
