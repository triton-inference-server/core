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

from __future__ import annotations

import ctypes
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, ClassVar, Optional, Sequence, Type

import numpy
import tritonserver._c as _triton_bindings
from cupy.cuda import Device
from tritonserver._c import InvalidArgumentError
from tritonserver._c import TRITONSERVER_BufferAttributes as BufferAttributes
from tritonserver._c import TRITONSERVER_DataType as DataType
from tritonserver._c import TRITONSERVER_MemoryType as MemoryType
from tritonserver._c import UnsupportedError
from typing_extensions import Self

from . import _dlpack

try:
    import cupy
except ImportError:
    cupy = None

DeviceOrMemoryType = Optional[
    tuple[MemoryType, int] | MemoryType | tuple[_dlpack.DLDeviceType, int] | str
]


class _CustomKeyErrorDict(dict):
    def __init__(
        self,
        from_name: str,
        to_name: str,
        *args,
        exception: Type[Exception] = InvalidArgumentError,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._to_name = to_name
        self._from_name = from_name
        self._exception = exception

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise self._exception(
                f"Unsupported {self._from_name}. Can't convert {key} to {self._to_name}"
            ) from None


STRING_TO_TRITON_MEMORY_TYPE: dict[str, MemoryType] = _CustomKeyErrorDict(
    "String",
    "Triton server memory type",
    {"CPU": MemoryType.CPU, "CPU_PINNED": MemoryType.CPU_PINNED, "GPU": MemoryType.GPU},
)


DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE: dict[
    _dlpack.DLDeviceType, MemoryType
] = _CustomKeyErrorDict(
    "DLPack device type",
    "Triton server memory type",
    {
        _dlpack.DLDeviceType.kDLCUDA: MemoryType.GPU,
        _dlpack.DLDeviceType.kDLCPU: MemoryType.CPU,
    },
)

TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE: dict[
    MemoryType, _dlpack.DLDeviceType
] = _CustomKeyErrorDict(
    "Triton server memory type",
    "DLPack device type",
    {
        **{
            value: key
            for key, value in DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE.items()
        },
        **{MemoryType.CPU_PINNED: _dlpack.DLDeviceType.kDLCPU},
    },
)

DLPACK_TO_TRITON_DTYPE: dict[
    tuple[_dlpack.DLDataTypeCode, int], DataType
] = _CustomKeyErrorDict(
    "DLPack data type",
    "Triton server data type",
    {
        (_dlpack.DLDataTypeCode.kDLBool, 1): DataType.BOOL,
        (_dlpack.DLDataTypeCode.kDLInt, 8): DataType.INT8,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            16,
        ): DataType.INT16,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            32,
        ): DataType.INT32,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            64,
        ): DataType.INT64,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            8,
        ): DataType.UINT8,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            16,
        ): DataType.UINT16,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            32,
        ): DataType.UINT32,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            64,
        ): DataType.UINT64,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            16,
        ): DataType.FP16,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            32,
        ): DataType.FP32,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            64,
        ): DataType.FP64,
        (
            _dlpack.DLDataTypeCode.kDLBfloat,
            16,
        ): DataType.BF16,
    },
)

TRITON_TO_DLPACK_DTYPE: dict[DataType, _dlpack.DLDataType] = _CustomKeyErrorDict(
    "Triton server data type",
    "DLPack data type",
    {
        value: _dlpack.DLDataType(type_code=key[0], bits=key[1], lanes=1)
        for key, value in DLPACK_TO_TRITON_DTYPE.items()
    },
)

NUMPY_TO_TRITON_DTYPE: dict[type, DataType] = _CustomKeyErrorDict(
    "Numpy data type",
    "Triton server data type",
    {
        bool: DataType.BOOL,
        numpy.int8: DataType.INT8,
        numpy.int16: DataType.INT16,
        numpy.int32: DataType.INT32,
        numpy.int64: DataType.INT64,
        numpy.uint8: DataType.UINT8,
        numpy.uint16: DataType.UINT16,
        numpy.uint32: DataType.UINT32,
        numpy.uint64: DataType.UINT64,
        numpy.float16: DataType.FP16,
        numpy.float32: DataType.FP32,
        numpy.float64: DataType.FP64,
        numpy.bytes_: DataType.BYTES,
        numpy.str_: DataType.BYTES,
        numpy.object_: DataType.BYTES,
    },
)

TRITON_TO_NUMPY_DTYPE: dict[DataType, type] = _CustomKeyErrorDict(
    "Triton data type",
    "Numpy data type",
    {
        **{value: key for key, value in NUMPY_TO_TRITON_DTYPE.items()},
        **{DataType.BYTES: numpy.object_},
    },
)


@dataclass
class MemoryBuffer:
    """Memory allocated for a Tensor"""

    data_ptr: int
    memory_type: MemoryType
    memory_type_id: int
    size: int
    owner: Any

    @staticmethod
    def from_dlpack(
        owner: Any, dlpack_object: Optional[DLPackObject] = None
    ) -> MemoryBuffer:
        if dlpack_object is None:
            if not hasattr(owner, "__dlpack__"):
                raise InvalidArgumentError("Object does not support DLpack protocol")

            dlpack_object = DLPackObject(owner)

        if not dlpack_object.contiguous:
            raise InvalidArgumentError("Only contiguous memory is supported")

        return MemoryBuffer(
            int(dlpack_object.data_ptr),
            dlpack_object.memory_type,
            dlpack_object.memory_type_id,
            dlpack_object.byte_size,
            owner,
        )

    def _create_buffer_attributes(self) -> BufferAttributes:
        buffer_attributes = BufferAttributes()
        buffer_attributes.memory_type = self.memory_type
        buffer_attributes.memory_type_id = self.memory_type_id
        buffer_attributes.byte_size = self.size
        #        buffer_attributes.cuda_ipc_handle = None
        return buffer_attributes


class MemoryAllocator(ABC):
    @abstractmethod
    def allocate(
        self, size: int, memory_type: Any, memory_type_id: int, tensor_name: str
    ) -> MemoryBuffer:
        """Allocate memory buffer for tensor.

        Parameters
        ----------
        size : int
            number of bytes requested
        memory_type : MemoryType
                type of memory requested (CPU, GPU, etc.)
        memory_type_id : int
            memory type id requested (typically device id)
        tensor_name : str
            name of tensor

        Returns
        -------
        MemoryBuffer
            memory buffer with requested size

        Examples
        --------
        memory_buffer = allocator.allocate(100,MemoryType.CPU,0,"output")
        """

        pass


class NumpyAllocator(MemoryAllocator):
    def __init__(self):
        pass

    def allocate(
        self, size: int, memory_type: MemoryType, memory_type_id: int, tensor_name: str
    ) -> MemoryBuffer:
        ndarray = numpy.empty(size, numpy.byte)
        return MemoryBuffer.from_dlpack(ndarray)


default_memory_allocators: dict[MemoryType, MemoryAllocator] = dict(
    {
        MemoryType.CPU: NumpyAllocator(),
    }
)

if cupy is not None:

    class CupyAllocator(MemoryAllocator):
        def __init__(self):
            pass

        def allocate(
            self,
            size: int,
            memory_type: MemoryType,
            memory_type_id: int,
            tensor_name: str,
        ) -> MemoryBuffer:
            with cupy.cuda.Device(memory_type_id):
                ndarray = cupy.empty(size, cupy.byte)

            return MemoryBuffer.from_dlpack(ndarray)

    default_memory_allocators[MemoryType.GPU] = CupyAllocator()


class ResponseAllocator:
    def __init__(
        self,
        memory_allocator: Optional[MemoryAllocator] = None,
        memory_type: DeviceOrMemoryType = None,
    ):
        self._memory_allocator = memory_allocator
        self._memory_type: Optional[MemoryType] = None
        self._memory_type_id: int = 0
        self._response_allocator = None
        self._set_memory_type(memory_type)

    def _set_memory_type(self, memory_type: DeviceOrMemoryType) -> None:
        if memory_type is None:
            return
        if isinstance(memory_type, tuple):
            if isinstance(memory_type[0], MemoryType):
                self._memory_type = memory_type[0]
                self._memory_type_id = memory_type[1]
                return
            if isinstance(memory_type[0], _dlpack.DLDeviceType):
                self._memory_type = DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE[
                    memory_type[0]
                ]
                self._memory_type_id = memory_type[1]
                return
        if isinstance(memory_type, MemoryType):
            self._memory_type = memory_type
            self._memory_type_id = 0
        if isinstance(memory_type, str):
            memory_str_tuple = memory_type.split(":")
            if len(memory_str_tuple) > 2:
                raise InvalidArgumentError(f"Invalid memory type string {memory_type}")
            self._memory_type = STRING_TO_TRITON_MEMORY_TYPE[memory_str_tuple[0]]
            if len(memory_str_tuple) == 2:
                try:
                    self._memory_type_id = int(memory_str_tuple[1])
                except ValueError:
                    raise InvalidArgumentError(
                        f"Invalid memory type string {memory_type}"
                    ) from None

    def allocate(
        self,
        allocator,
        tensor_name,
        byte_size,
        memory_type,
        memory_type_id,
        user_object,
    ):
        if self._memory_type is not None:
            memory_type = self._memory_type
            memory_type_id = self._memory_type_id

        memory_allocator = self._memory_allocator
        if memory_allocator is None:
            memory_allocator = default_memory_allocators[memory_type]

        memory_buffer = memory_allocator.allocate(
            byte_size, memory_type, memory_type_id, tensor_name
        )

        return (
            memory_buffer.data_ptr,
            memory_buffer,
            memory_buffer.memory_type,
            memory_buffer.memory_type_id,
        )

    def release(
        self,
        allocator,
        buffer_,
        buffer_user_object,
        byte_size,
        memory_type,
        memory_type_id,
    ):
        pass

    def start(self, allocator, user_object):
        pass

    def query_preferred_memory_type(
        self,
        allocator,
        user_object,
        tensor_name,
        byte_size,
        memory_type: MemoryType,
        memory_type_id,
    ):
        if self._memory_type is not None:
            memory_type = self._memory_type
            memory_type_id = self._memory_type_id

        return (memory_type, memory_type_id)

    def set_buffer_attributes(
        self, allocator, tensor_name, buffer_attributes, user_object, buffer_user_object
    ):
        if self._memory_type:
            buffer_attributes.memory_type = self._memory_type
            buffer_attributes.memory_type_id = self._memory_type_id
        return buffer_attributes

    def create_response_allocator(self):
        self._response_allocator = _triton_bindings.TRITONSERVER_ResponseAllocator(
            self.allocate, self.release, self.start
        )
        self._response_allocator.set_query_function(self.query_preferred_memory_type)

        self._response_allocator.set_buffer_attributes_function(
            self.set_buffer_attributes
        )
        return self._response_allocator


class DLPackObject:
    def __init__(self, value) -> None:
        try:
            self._capsule = _dlpack.get_dlpack_capsule(value)
            self._tensor = _dlpack.get_managed_tensor(self._capsule).dl_tensor
        except Exception as e:
            raise InvalidArgumentError(
                f"Object does not support DLPack protocol: {e}"
            ) from None

    def __eq__(self, other) -> bool:
        if not isinstance(other, DLPackObject):
            return False
        if self.byte_size != other.byte_size:
            return False
        if self.memory_type != other.memory_type:
            return False
        if self.memory_type_id != other.memory_type_id:
            return False
        if self.shape != other.shape:
            return False
        if self.data_ptr != other.data_ptr:
            return False
        if self.contiguous != other.contiguous:
            return False
        if self.triton_data_type != other.triton_data_type:
            return False
        return True

    @property
    def byte_size(self) -> int:
        return _dlpack.get_byte_size(
            self._tensor.dtype, self._tensor.ndim, self._tensor.shape
        )

    @property
    def memory_type(self) -> MemoryType:
        return DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE[self._tensor.device.device_type]

    @property
    def memory_type_id(self) -> int:
        return self._tensor.device.device_id

    @property
    def shape(self) -> list[int]:
        return [self._tensor.shape[i] for i in range(self._tensor.ndim)]

    @property
    def triton_data_type(self) -> DataType:
        return DLPACK_TO_TRITON_DTYPE[self.data_type]

    @property
    def data_type(self) -> tuple[_dlpack.DLDataTypeCode, int]:
        return (self._tensor.dtype.type_code, self._tensor.dtype.bits)

    @property
    def data_ptr(self) -> ctypes.c_void_p:
        return self._tensor.data + self._tensor.byte_offset

    @property
    def contiguous(self) -> bool:
        return _dlpack.is_contiguous_data(
            self._tensor.ndim, self._tensor.shape, self._tensor.strides
        )


@dataclass
class Tensor:
    data_type: DataType
    shape: Sequence[int]
    memory_buffer: MemoryBuffer

    @property
    def data_ptr(self) -> int:
        return self.memory_buffer.data_ptr

    @property
    def memory_type(self) -> MemoryType:
        return self.memory_buffer.memory_type

    @property
    def memory_type_id(self) -> int:
        return self.memory_buffer.memory_type_id

    @property
    def size(self) -> int:
        return self.memory_buffer.size

    def __dlpack__(self, *, stream=None):
        #        if not (stream is None or (isinstance(stream, int) and stream == 0)):
        #           raise UnsupportedError(
        #              f"DLPack stream synchronization on {stream} not currently supported"
        #         )

        dl_managed_tensor = Tensor._create_managed_tensor()
        dl_managed_tensor.dl_tensor.data = self.data_ptr
        dl_managed_tensor.dl_tensor.device = _dlpack.DLDevice(
            TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE[self.memory_type],
            self.memory_type_id,
        )

        dl_managed_tensor.dl_tensor.dtype = TRITON_TO_DLPACK_DTYPE[self.data_type]
        dl_managed_tensor.dl_tensor.ndim = len(self.shape)
        dl_managed_tensor.dl_tensor.shape = (ctypes.c_int64 * len(self.shape))(
            *self.shape
        )
        dl_managed_tensor.dl_tensor.strides = ctypes.POINTER(ctypes.c_int64)()
        dl_managed_tensor.dl_tensor.byte_offset = 0
        dl_managed_tensor.deleter = Tensor._managed_tensor_deleter

        self._set_dlpack_manager_ctx(dl_managed_tensor)
        pycapsule = ctypes.pythonapi.PyCapsule_New(
            ctypes.byref(dl_managed_tensor),
            _dlpack.c_str_dltensor,
            Tensor._pycapsule_deleter,
        )
        return pycapsule

    def __dlpack_device__(self):
        return (
            TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE[self.memory_type],
            self.memory_type_id,
        )

    def to_ndarray(self, _type: ModuleType | type) -> Any:
        if _type in Tensor._to_converters:
            return Tensor._to_converters[_type](self)
        elif hasattr(_type, "from_dlpack"):
            return _type.from_dlpack(self)
        raise InvalidArgumentError(
            f"Output type {_type} not supported. Must be one of {list(Tensor._to_converters.keys())} or the type must support from_dlpack"
        )

    @staticmethod
    def from_memory_buffer(data_type, shape, memory_buffer):
        return Tensor(data_type, shape, memory_buffer)

    @staticmethod
    def from_ndarray(obj: Any):
        if type(obj) in Tensor._from_converters:
            return Tensor._from_converters[type(obj)](obj)
        elif hasattr(obj, "__dlpack__"):
            return Tensor.from_dlpack(obj)
        else:
            raise InvalidArgumentError(
                f"Input type {type(obj)} not supported. Must be one of {list(Tensor._from_converters.keys())} or the type must support __dlpack__"
            )

    @staticmethod
    def from_dlpack(obj: Any) -> Tensor:
        dlpack_object = DLPackObject(obj)
        data_type = dlpack_object.triton_data_type
        shape = dlpack_object.shape
        memory_buffer = MemoryBuffer.from_dlpack(obj, dlpack_object=dlpack_object)
        return Tensor(data_type, shape, memory_buffer)

    def _to_host(self) -> numpy.ndarray:
        array_module = Tensor._array_modules[self.memory_type]
        if array_module is not None:
            array_obj = array_module.from_dlpack(self)
            if hasattr(array_module, "asnumpy"):
                return array_module.asnumpy(array_obj)
            if hasattr(array_obj, "numpy"):
                return array_obj.numpy(force=True)
        raise UnsupportedError(
            f"Conversion from {self.memory_type} to numpy array not supported."
        )

    def _to_numpy(self) -> numpy.ndarray:
        if self.memory_type == MemoryType.GPU:
            numpy_ndarray = self._to_host()
        else:
            numpy_ndarray = numpy.from_dlpack(self)

        if self.data_type == DataType.BYTES:
            numpy_ndarray = Tensor._deserialize_bytes_array(numpy_ndarray)

        return numpy_ndarray

    @staticmethod
    def _deserialize_bytes_array(numpy_ndarray: numpy.ndarray) -> numpy.ndarray:
        result = []
        _buffer = memoryview(numpy_ndarray)
        offset = 0
        while offset < len(_buffer):
            (item_length,) = struct.unpack_from("@I", _buffer, offset)
            offset += 4
            result.append(bytes(_buffer[offset : offset + item_length]))
            offset += item_length
        return numpy.array(result, dtype=numpy.object_)

    @staticmethod
    def _serialize_numpy_bytes_array(array: numpy.ndarray) -> numpy.ndarray:
        result = []
        for array_item in numpy.nditer(array, flags=["refs_ok"], order="C"):
            item = array_item.item()
            if not isinstance(item, bytes):
                item = str(item).encode("utf-8")
            result.append(struct.pack("@I", len(item)))
            result.append(item)
        return numpy.frombuffer(b"".join(result), dtype=numpy.byte)

    @staticmethod
    def _from_numpy(obj: numpy.ndarray | numpy.generic) -> Tensor:
        data_type = NUMPY_TO_TRITON_DTYPE[obj.dtype.type]
        shape = obj.shape

        if isinstance(obj, numpy.generic):
            obj = numpy.asarray(obj)

        if data_type == DataType.BYTES:
            obj = Tensor._serialize_numpy_bytes_array(obj)

        memory_buffer = MemoryBuffer(
            data_ptr=obj.ctypes.data,
            memory_type=MemoryType.CPU,
            memory_type_id=0,
            size=obj.itemsize * obj.size,
            owner=obj,
        )

        return Tensor(data_type, shape, memory_buffer)

    @staticmethod
    def _create_managed_tensor():
        size = ctypes.c_size_t(ctypes.sizeof(_dlpack.DLManagedTensor))
        address = ctypes.pythonapi.PyMem_RawMalloc(size)
        return _dlpack.DLManagedTensor.from_address(address)

    @staticmethod
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def _managed_tensor_deleter(handle: int) -> None:
        dl_managed_tensor = _dlpack.DLManagedTensor.from_address(handle)
        tensor_obj_ptr = ctypes.cast(
            dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
        )
        tensor_obj = tensor_obj_ptr.contents
        ctypes.pythonapi.Py_DecRef(tensor_obj)
        shape_obj = ctypes.py_object(dl_managed_tensor.dl_tensor.shape)
        ctypes.pythonapi.Py_DecRef(shape_obj)
        ctypes.pythonapi.PyMem_RawFree(handle)

    @staticmethod
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def _pycapsule_deleter(handle: ctypes.c_void_p) -> None:
        try:
            pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
            if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _dlpack.c_str_dltensor):
                dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
                    pycapsule, _dlpack.c_str_dltensor
                )

                Tensor._managed_tensor_deleter(dl_managed_tensor)

                ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)
        except Exception as e:
            print(f"Exception occurred while deleting capsule: {e}")
            raise e

    def _set_dlpack_manager_ctx(self, dl_managed_tensor):
        tensor_obj = ctypes.py_object(self)
        tensor_obj_ptr = ctypes.pointer(tensor_obj)
        dl_managed_tensor.manager_ctx = ctypes.cast(tensor_obj_ptr, ctypes.c_void_p)
        shape_obj = ctypes.py_object(dl_managed_tensor.dl_tensor.shape)
        ctypes.pythonapi.Py_IncRef(tensor_obj)
        ctypes.pythonapi.Py_IncRef(shape_obj)

    _from_converters: ClassVar[dict[type, Callable[[Any], Tensor]]] = dict(
        {
            numpy.ndarray: _from_numpy,
            numpy.generic: _from_numpy,
        },
    )

    _to_converters: ClassVar[dict[type | ModuleType, Callable[[Self], Any]]] = dict(
        {numpy.ndarray: _to_numpy, numpy: _to_numpy},
    )

    _array_modules: ClassVar[dict[MemoryType, ModuleType | None]] = dict(
        {MemoryType.CPU: numpy, MemoryType.GPU: cupy},
    )
