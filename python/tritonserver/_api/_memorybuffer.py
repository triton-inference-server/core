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

"""Memory Buffer for Tensor Memory"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy
from tritonserver._api._dlpack import DLDeviceType
from tritonserver._c.triton_bindings import TRITONSERVER_MemoryType as MemoryType

DeviceOrMemoryType = (
    tuple[MemoryType, int] | MemoryType | tuple[DLDeviceType, int] | str
)

from tritonserver._api._datautils import DLPackObject
from tritonserver._c.triton_bindings import (
    InvalidArgumentError,
    TRITONSERVER_BufferAttributes,
)


@dataclass
class MemoryBuffer:
    """Memory allocated for a Tensor.

    This object does not own the memory but holds a reference to the
    owner.

    Parameters
    ----------
    data_ptr : int
        Pointer to the allocated memory.
    memory_type : MemoryType
        memory type
    memory_type_id : int
        memory type id (typically the same as device id)
    size : int
        Size of the allocated memory in bytes.
    owner : Any
        Object that owns or manages the memory buffer.  Allocated
        memory must not be freed while a reference to the owner is
        held.

    Examples
    --------
    >>> buffer = MemoryBuffer.from_dlpack(numpy.array([100],dtype=numpy.uint8))

    """

    data_ptr: int
    memory_type: MemoryType
    memory_type_id: int
    size: int
    owner: Any

    @staticmethod
    def from_dlpack(owner: Any) -> MemoryBuffer:
        if not hasattr(owner, "__dlpack__"):
            raise InvalidArgumentError("Object does not support DLpack protocol")

        dlpack_object = DLPackObject(owner)

        return MemoryBuffer._from_dlpack_object(owner, dlpack_object)

    @staticmethod
    def _from_dlpack_object(owner: Any, dlpack_object: DLPackObject) -> MemoryBuffer:
        if not dlpack_object.contiguous:
            raise InvalidArgumentError("Only contiguous memory is supported")

        return MemoryBuffer(
            int(dlpack_object.data_ptr),
            dlpack_object.memory_type,
            dlpack_object.memory_type_id,
            dlpack_object.byte_size,
            owner,
        )

    def _create_tritonserver_buffer_attributes(self) -> TRITONSERVER_BufferAttributes:
        buffer_attributes = TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type = self.memory_type
        buffer_attributes.memory_type_id = self.memory_type_id
        buffer_attributes.byte_size = self.size
        # TODO: Support allocation / use of cuda shared memory
        #        buffer_attributes.cuda_ipc_handle = None
        return buffer_attributes
