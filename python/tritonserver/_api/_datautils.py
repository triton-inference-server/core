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

from __future__ import annotations

import ctypes
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Any, Optional, Type
import numpy
from . import _dlpack
import tritonserver._c as triton_bindings
from tritonserver._c import TRITONSERVER_DataType as DataType
from tritonserver._c import TRITONSERVER_BufferAttributes as BufferAttributes
from tritonserver._c import TRITONSERVER_MemoryType as MemoryType
from tritonserver._c.triton_bindings import InvalidArgumentError

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


DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE = _CustomKeyErrorDict(
    "DLPack device type",
    "Triton server memory type",
    {
        _dlpack.DLDeviceType.kDLCUDA: MemoryType.GPU,
        _dlpack.DLDeviceType.kDLCPU: MemoryType.CPU,
    },
)

# TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE = (lambda: None, {})


DLPACK_TO_TRITON_DTYPE = _CustomKeyErrorDict(
    "DLPack data type",
    "Triton server data type",
    {
        (_dlpack.DLDataTypeCode.kDLBool, 1): triton_bindings.TRITONSERVER_DataType.BOOL,
        (_dlpack.DLDataTypeCode.kDLInt, 8): triton_bindings.TRITONSERVER_DataType.INT8,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            16,
        ): triton_bindings.TRITONSERVER_DataType.INT16,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            32,
        ): triton_bindings.TRITONSERVER_DataType.INT32,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            64,
        ): triton_bindings.TRITONSERVER_DataType.INT64,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            8,
        ): triton_bindings.TRITONSERVER_DataType.UINT8,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            16,
        ): triton_bindings.TRITONSERVER_DataType.UINT16,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            32,
        ): triton_bindings.TRITONSERVER_DataType.UINT32,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            64,
        ): triton_bindings.TRITONSERVER_DataType.UINT64,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            16,
        ): triton_bindings.TRITONSERVER_DataType.FP16,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            32,
        ): triton_bindings.TRITONSERVER_DataType.FP32,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            64,
        ): triton_bindings.TRITONSERVER_DataType.FP64,
        (
            _dlpack.DLDataTypeCode.kDLBfloat,
            16,
        ): triton_bindings.TRITONSERVER_DataType.BF16,
    },
)


NUMPY_TO_TRITON_DTYPE = _CustomKeyErrorDict(
    "Numpy data type",
    "Triton server data type",
    {
        bool: triton_bindings.TRITONSERVER_DataType.BOOL,
        numpy.int8: triton_bindings.TRITONSERVER_DataType.INT8,
        numpy.int16: triton_bindings.TRITONSERVER_DataType.INT16,
        numpy.int32: triton_bindings.TRITONSERVER_DataType.INT32,
        numpy.int64: triton_bindings.TRITONSERVER_DataType.INT64,
        numpy.uint8: triton_bindings.TRITONSERVER_DataType.UINT8,
        numpy.uint16: triton_bindings.TRITONSERVER_DataType.UINT16,
        numpy.uint32: triton_bindings.TRITONSERVER_DataType.UINT32,
        numpy.uint64: triton_bindings.TRITONSERVER_DataType.UINT64,
        numpy.float16: triton_bindings.TRITONSERVER_DataType.FP16,
        numpy.float32: triton_bindings.TRITONSERVER_DataType.FP32,
        numpy.float64: triton_bindings.TRITONSERVER_DataType.FP64,
        numpy.bytes_: triton_bindings.TRITONSERVER_DataType.BYTES,
        numpy.str_: triton_bindings.TRITONSERVER_DataType.BYTES,
        numpy.object_: triton_bindings.TRITONSERVER_DataType.BYTES,
    },
)

TRITON_TO_NUMPY_DTYPE = _CustomKeyErrorDict(
    "Triton data type",
    "Numpy data type",
    {value: key for key, value in NUMPY_TO_TRITON_DTYPE.items()},
)


class _UnsupportedModule:
    def __init__(self, module_name):
        self._module_name = module_name

    def __getattribute__(self, name):
        raise triton_bindings.UnsupportedError(f"{self._module_name} not supported")

    def __setattribute__(self, name):
        raise triton_bindings.UnsupportedError(f"{self._module_name} not supported")


try:
    import cupy
except ImportError:
    cupy = _UnsupportedModule("cupy")


@dataclass
class MemoryBuffer:
    data_ptr: ctypes.c_void_p
    memory_type: MemoryType
    memory_type_id: int
    size: int
    owner: Any

    @staticmethod
    def from_dlpack(owner: Any) -> MemoryBuffer:
        if not hasattr(owner, "__dlpack__"):
            raise InvalidArgument("Object does not support DLpack protocol")

        dlpack_object = DLPackObject(owner)

        if not dlpack_object.contiguous:
            raise InvalidArgument("Only contiguous memory is supported")

        return MemoryBuffer(
            dlpack_object.data_ptr,
            dlpack_object.memory_type,
            dlpack_object.memory_type_id,
            dlpack_object.byte_size,
            owner,
        )


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
            return a memory buffer with requested size

        Examples
        --------
        memory_buffer = allocator.allocate(100,MemoryType.CPU,0,"output")
        """

        pass


DeviceOrMemoryType = Optional[
    tuple[MemoryType, int] | MemoryType | tuple[_dlpack.DLDeviceType, int] | str
]


class _ResponseAllocator:
    def __init__(
        self, memory_allocator: MemoryAllocator, memory_type: DeviceOrMemoryType = None
    ):
        self._memory_allocator = memory_allocator
        self._set_memory_type(memory_type)
        self._response_allocator = None

    def _set_memory_type(self, memory_type: DeviceOrMemoryType) -> None:
        self._memory_type = None
        self._memory_type_id = None

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
            return

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

        memory_buffer = self._memory_allocator.allocate(
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

    def create_response_allocator(self):
        if self._response_allocator is None:
            print("creating allocator", flush=True)
            self._response_allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
                self.allocate, self.release, self.start
            )

            if hasattr(self, "query_preferred_memory_type"):
                self._response_allocator.set_query_function(
                    self.query_preferred_memory_type
                )

            if hasattr(self, "set_buffer_attributes"):
                self._response_allocator.set_buffer_attributes_function(
                    self.set_buffer_attributes
                )
        return self._response_allocator


try:
    import cupy

    class CupyAllocator(MemoryAllocator):
        def __init__(self):
            pass

        def start(self, allocator, user_object):
            pass

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

        def allocate(
            self,
            allocator,
            tensor_name,
            byte_size,
            memory_type,
            memory_type_id,
            user_object,
        ):
            with cupy.cuda.Device(memory_type_id):
                _buffer = cupy.empty(byte_size, cupy.byte)

            dlpack_object = DLPackObject(_buffer)

            return (
                dlpack_object.data_ptr,
                _buffer,
                dlpack_object.memory_type,
                dlpack_object.memory_type_id,
            )

except ImportError:
    pass


class NumpyAllocator(MemoryAllocator):
    def __init__(self):
        pass

    def allocate(
        self, size: int, memory_type: MemoryType, memory_type_id: int, tensor_name: str
    ) -> MemoryBuffer:
        ndarray = numpy.empty(size, numpy.byte)
        return MemoryBuffer.from_dlpack(ndarray)


class DefaultAllocator(MemoryAllocator):
    def __init__(self):
        self._cpu_allocator = NumpyAllocator()
        try:
            self._gpu_allocator = CupyAllocator()
        except Exception:
            self._gpu_allocator = None
        self._allocators: dict[
            triton_bindings.TRITONSERVER_MemoryType, MemoryAllocator
        ] = defaultdict(lambda: self._cpu_allocator)
        self._allocators[
            triton_bindings.TRITONSERVER_MemoryType.CPU
        ] = self._cpu_allocator
        if self._gpu_allocator is not None:
            self._allocators[
                triton_bindings.TRITONSERVER_MemoryType.GPU
            ] = self._gpu_allocator

    def allocate(
        self, size: int, memory_type: MemoryType, memory_type_id: int, tensor_name: str
    ) -> MemoryBuffer:
        return self._allocators[memory_type].allocate(
            size, memory_type, memory_type_id, tensor_name
        )


class DLPackObject:
    def __init__(self, value):
        try:
            self._capsule = _dlpack.get_dlpack_capsule(value)
            self._tensor = _dlpack.get_managed_tensor(self._capsule).dl_tensor
        except Exception:
            raise InvalidArgumentError(
                "Object does not support DLPack protocol"
            ) from None

    @property
    def byte_size(self):
        return _dlpack.get_byte_size(
            self._tensor.dtype, self._tensor.ndim, self._tensor.shape
        )

    @property
    def memory_type(self):
        return DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE[self._tensor.device.device_type]

    @property
    def memory_type_id(self):
        return self._tensor.device.device_id

    @property
    def shape(self):
        return [self._tensor.shape[i] for i in range(self._tensor.ndim)]

    @property
    def triton_data_type(self):
        return DLPACK_TO_TRITON_DTYPE[self.data_type]

    @property
    def data_type(self):
        return (self._tensor.dtype.type_code, self._tensor.dtype.bits)

    @property
    def data_ptr(self):
        return self._tensor.data + self._tensor.byte_offset

    @property
    def contiguous(self):
        return _dlpack.is_contiguous_data(
            self._tensor.ndim, self._tensor.shape, self._tensor.strides
        )


@dataclass
class Tensor:
    data_type: DataType
    shape: Iterable[int]
    buffer_attributes: BufferAttributes
    _buffer: ctypes.c_void_p
    _value: Any

    @staticmethod
    def _create_managed_tensor():
        size = ctypes.c_size_t(ctypes.sizeof(_dlpack.DLManagedTensor))
        return _dlpack.DLManagedTensor.from_address(
            ctypes.pythonapi.PyMem_RawMalloc(size)
        )

    def __dlpack__(self, stream=None):
        context = _dlpack.DataViewContext(self.shape)
        dl_managed_tensor = Tensor._create_managed_tensor()
        dl_managed_tensor.dl_tensor.data = self._buffer
        dl_managed_tensor.dl_tensor.device = self.buffer_attributes.memory_type

    def __dlpack__(self, stream=None):
        return self._value.__dlpack__(stream)

    def __dlpack_device__(self):
        return self._value.__dlpack_device__()

    @property
    def squeeze(self):
        return self._value.squeeze

    @property
    def __array_interface__(self):
        print("hello")
        return self._value.__array_interface__

    @property
    def __array__(self):
        print("hello foo")
        return self._value.__array_()

    @property
    def __cuda_array_interface__(self):
        print("goodbye")
        return self._value.__cuda_array_interface__

    # if object has __cuda_array_interface
    # or __array_interface__ interface
    # else fail

    def as_array(self) -> cp.ndarray | numpy.ndarray:
        return self._value

    # convert to cpu / numpy if possible
    def as_numpy(self):
        if (
            self.buffer_attributes.memory_type
            == triton_bindings.TRITONSERVER_MemoryType.CPU
        ):
            return numpy.from_dlpack(self._value)
        else:
            return cupy.asnumpy(self._value)

    # def to_cpu(self):
    #     if self.buffer_attributes.memory_type == triton_bindings.TRITONSERVER_MemoryType.CPU:
    #         return numpy.from_dlpack(self._value)
    #     else:
    #         return cupy.asnumpy(self._value)
    #     pass

    # def to_gpu(self, device_id):
    #     pass

    # def ndarray(self):
    #     print(dir(self._value))
    #     return numpy.from_dlpack(self._value)
    #     pass

    @staticmethod
    def from_buffer(
        data_type, shape, buffer_, byte_size, memory_type, memory_type_id, value
    ):
        buffer_attributes = BufferAttributes()
        buffer_attributes.memory_type = memory_type
        buffer_attributes.memory_type_id = memory_type_id
        buffer_attributes.byte_size = byte_size
        if data_type == DataType.BYTES:
            value = Tensor._deserialize_bytes_array(value)
        numpy_dtype = TRITON_TO_NUMPY_DTYPE[data_type]
        value = value.view(numpy_dtype).reshape(shape)
        return Tensor(data_type, shape, buffer_attributes, buffer_, value)

    @staticmethod
    def from_value(value):
        if type(value) in Tensor._supported_conversions:
            return Tensor._supported_conversions[type(value)](value)
        elif hasattr(value, "__dlpack__"):
            return Tensor._from_dlpack(value)
        else:
            raise triton_bindings.InvalidArgumentError(
                f"Input type {type(value)} not supported. Must be one of {list(Tensor._supported_conversions.keys())} or the type must support __dlpack__"
            )

    @staticmethod
    def _from_dlpack(value):
        dlpack_object = DLPackObject(value)
        data_type = dlpack_object.triton_data_type
        if data_type == triton_bindings.TRITONSERVER_DataType.INVALID:
            raise triton_bindings.InvalidArgumentError(
                f"DLPack dtype {dlpack_object.data_type} not supported"
            )

        if data_type == triton_bindings.TRITONSERVER_DataType.BYTES:
            raise triton_bindings.InvalidArgumentError(
                f"DLPack does not support {data_type}"
            )

        shape = dlpack_object.shape

        buffer_ = dlpack_object.data_ptr

        buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type = dlpack_object.memory_type
        buffer_attributes.memory_type_id = dlpack_object.memory_type_id
        buffer_attributes.byte_size = dlpack_object.byte_size

        return Tensor(data_type, shape, buffer_attributes, buffer_, value)

    @staticmethod
    def _deserialize_bytes_array(array):
        result = []
        try:
            _buffer = memoryview(array)
        except:
            _buffer = array.tobytes()
        offset = 0
        while offset < len(_buffer):
            (item_length,) = struct.unpack_from("@I", _buffer, offset)
            offset += 4
            result.append(bytes(_buffer[offset : offset + item_length]))
            offset += item_length
        return numpy.array(result, dtype=numpy.object_)

    @staticmethod
    def _serialize_numpy_bytes_array(array):
        result = []
        for array_item in numpy.nditer(array, flags=["refs_ok"], order="C"):
            item = array_item.item()
            if type(item) != bytes:
                item = str(item).encode("utf-8")
            result.append(struct.pack("@I", len(item)))
            result.append(item)
        return numpy.frombuffer(b"".join(result), dtype=numpy.byte)

    @staticmethod
    def _from_numpy(value: numpy.ndarray | numpy.generic) -> Tensor:
        data_type = NUMPY_TO_TRITON_DTYPE[value.dtype.type]
        if data_type == triton_bindings.TRITONSERVER_DataType.INVALID:
            raise triton_bindings.InvalidArgumentError(
                f"Numpy type {value.dtype.type} not supported"
            )
        shape = value.shape
        if data_type == triton_bindings.TRITONSERVER_DataType.BYTES:
            value = Tensor._serialize_numpy_bytes_array(value)
        buffer_ = value.ctypes.data
        buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
        buffer_attributes.memory_type_id = 0
        buffer_attributes.byte_size = value.itemsize * value.size

        return Tensor(data_type, shape, buffer_attributes, buffer_, value)

    _supported_conversions: ClassVar[dict] = dict(
        {
            numpy.ndarray: _from_numpy,
            numpy.generic: _from_numpy,
        },
    )
