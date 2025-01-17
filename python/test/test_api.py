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

import asyncio
import copy
import gc
import json
import os
import shutil
import sys
import time
import unittest
from collections import Counter
from contextlib import contextmanager

import numpy
import pytest
import tritonserver

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch

    if not torch.cuda.is_available():
        torch = None
except ImportError:
    torch = None

TEST_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_MODEL_DIR = os.path.abspath(os.path.join(TEST_ROOT, "test_api_models"))
TEST_LOGS_DIR = os.path.abspath(os.path.join(TEST_ROOT, "test_api_logs"))


@pytest.fixture(autouse=True, scope="module")
def create_log_dir():
    shutil.rmtree(TEST_LOGS_DIR, ignore_errors=True)
    os.makedirs(TEST_LOGS_DIR)


@pytest.fixture()
def server_options(request):
    return tritonserver.Options(
        server_id="TestServer",
        model_repository=TEST_MODEL_DIR,
        log_verbose=6,
        log_error=True,
        log_warn=True,
        log_info=True,
        exit_on_error=True,
        strict_model_config=False,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
        exit_timeout=5,
        log_file=os.path.join(TEST_LOGS_DIR, request.node.name + ".server.log"),
    )


class TestModels:
    def test_create_request(self, server_options):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        request = server.models()["test"].create_request()

        request = tritonserver.InferenceRequest(server.model("test"))


class TestOutputMemory:
    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_memory_fallback_to_cpu(self, server_options):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        assert server.ready()

        # The memory allocator is internal to the binding, and before GPU memory support
        # is added, it will always fallback to CPU memory regardless of the memory
        # preferred by the backend.
        # TODO: Revisit this test when GPU memory support is added, i.e. the backend
        #       prefers GPU memory, but the system only has CPU memory.
        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {
                            "decoupled": {"string_value": "False"},
                            "request_gpu_memory": {"string_value": "True"},
                        },
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
        ):
            assert (
                response.outputs["fp16_output"].memory_type
                == tritonserver.MemoryType.CPU
            )
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            assert fp16_input[0][0] == fp16_output[0][0]

    def test_unsupported_memory_type(self, server_options):
        # TODO: Revisit this test when GPU memory support is added, i.e. the request
        #       specifies output to be in GPU memory, but the system only has CPU
        #       memory, which an exception should be raised during inference.
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        assert server.ready()

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        with pytest.raises(tritonserver.InvalidArgumentError):
            for response in server.model("test").infer(
                inputs={
                    "string_input": tritonserver.Tensor.from_string_array([["hello"]])
                },
                output_memory_type="unsupported",
            ):
                pass


class TestTensor:
    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_cpu_to_gpu(self):
        cpu_array = numpy.random.rand(1, 3, 100, 100).astype(numpy.float32)
        cpu_tensor = tritonserver.Tensor.from_dlpack(cpu_array)
        gpu_tensor = cpu_tensor.to_device("gpu:0")
        gpu_array = cupy.from_dlpack(gpu_tensor)

        assert gpu_array.device == cupy.cuda.Device(0)

        numpy.testing.assert_array_equal(cpu_array, gpu_array.get())

        memory_buffer = tritonserver.MemoryBuffer.from_dlpack(gpu_array)

        assert gpu_array.__cuda_array_interface__["data"][0] == memory_buffer.data_ptr

    @pytest.mark.skipif(
        torch is None, reason="Skipping gpu memory, torch not installed"
    )
    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_gpu_tensor_from_dl_pack(self):
        cupy_array = cupy.ones([100]).astype(cupy.float64)
        tensor = tritonserver.Tensor.from_dlpack(cupy_array)
        torch_tensor = torch.from_dlpack(cupy_array)

        assert torch_tensor.data_ptr() == tensor.data_ptr
        assert torch_tensor.nbytes == tensor.size
        assert torch_tensor.__dlpack_device__() == tensor.__dlpack_device__()

    @pytest.mark.skipif(torch is None, reason="Skipping test, torch not installed")
    def test_tensor_from_numpy(self):
        cpu_array = numpy.random.rand(1, 3, 100, 100).astype(numpy.float32)
        tensor = tritonserver.Tensor.from_dlpack(cpu_array)
        torch_tensor = torch.from_dlpack(tensor)
        numpy.testing.assert_array_equal(torch_tensor.numpy(), cpu_array)
        assert torch_tensor.data_ptr() == cpu_array.ctypes.data

    async def _tensor_from_numpy(self):
        owner = numpy.ones(2**27)
        tensor = tritonserver.Tensor.from_dlpack(owner)
        array = numpy.from_dlpack(tensor)
        del owner
        del tensor
        del array
        await asyncio.sleep(0.1)

    async def _async_test_runs(self):
        tasks = []
        for _ in range(100):
            tasks.append(asyncio.create_task(self._tensor_from_numpy()))
        try:
            await asyncio.wait(tasks)
        except Exception as e:
            print(e)

    @staticmethod
    @contextmanager
    def object_collector():
        gc.collect()
        objects_before = gc.get_objects()
        yield
        objects_after = gc.get_objects()
        new_objects = [type(x) for x in objects_after[len(objects_before) :]]
        tensor_objects = [
            x for x in objects_after if isinstance(x, tritonserver.Tensor)
        ]
        if tensor_objects:
            print("Tensor objects")
            print(len(tensor_objects))
            print(type(tensor_objects[-1].memory_buffer.owner))
            print(
                f"\nTotal Collected Objects ({len(new_objects)}) {Counter(new_objects)}"
            )
        assert len(tensor_objects) == 0, "Leaked Tensors"

    def test_cpu_memory_leak_async(self):
        with TestTensor.object_collector():
            asyncio.run(self._async_test_runs())

    def test_cpu_memory_leak_sync(self):
        with TestTensor.object_collector():
            for _ in range(100):
                owner = numpy.ones(2**27)
                tensor = tritonserver.Tensor.from_dlpack(owner)
                array = numpy.from_dlpack(tensor)
                del owner
                del tensor
                del array

    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_gpu_memory_leak(self):
        with TestTensor.object_collector():
            for _ in range(100):
                owner = cupy.ones(2**27)
                tensor = tritonserver.Tensor.from_dlpack(owner)
                array = cupy.from_dlpack(tensor)
                del owner
                del tensor
                del array

    def test_reference_counts(self):
        with TestTensor.object_collector():
            owner = numpy.ones(2**27)
            owner_data = owner.ctypes.data
            assert sys.getrefcount(owner) - 1 == 1, "Invalid Count"

            tensor = tritonserver.Tensor.from_dlpack(owner)
            assert sys.getrefcount(owner) - 1 == 2, "Invalid Count"
            assert sys.getrefcount(tensor) - 1 == 1, "Invalid Count"
            del owner

            numpy_array = numpy.from_dlpack(tensor)
            assert owner_data == numpy_array.ctypes.data
            assert sys.getrefcount(tensor) - 1 == 2, "Invalid Count"
            assert sys.getrefcount(numpy_array) - 1 == 1, "Invalid Count"

            tensor.shape = [2, 2**26]

            assert numpy_array.shape == (2**27,), "Invalid Shape"

            numpy_array_2 = numpy.from_dlpack(tensor)
            del tensor
            assert owner_data == numpy_array.ctypes.data
            assert numpy_array_2.shape == (2, 2**26)
            del numpy_array
            del numpy_array_2


class TestServer:
    def test_not_started(self):
        server = tritonserver.Server()
        with pytest.raises(tritonserver.InvalidArgumentError):
            server.ready()

    def test_invalid_option_type(self):
        server = tritonserver.Server(server_id=1)
        with pytest.raises(TypeError):
            server.start()

        server = tritonserver.Server(model_repository=1)
        with pytest.raises(TypeError):
            server.start()

    def test_invalid_repo(self):
        with pytest.raises(tritonserver.InternalError):
            tritonserver.Server(model_repository="foo").start()

    def test_ready(self, server_options):
        server = tritonserver.Server(server_options).start()
        assert server.ready()

    @pytest.mark.xfail(
        run=False,
        reason="Some request/response object may not be released which may cause server stop to fail",
    )
    def test_stop(self, server_options):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        assert server.ready()

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                        "instance_group": [{"kind": "KIND_CPU"}],
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            numpy.testing.assert_array_equal(fp16_input, fp16_output)

        server.stop()

    def test_model_repository_not_specified(self):
        with pytest.raises(tritonserver.InvalidArgumentError):
            tritonserver.Server(model_repository=None).start()


class TestInference:
    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cupy not installed")
    def test_gpu_output(self, server_options):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        assert server.ready()

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            output_memory_type="gpu",
        ):
            fp16_output = cupy.from_dlpack(response.outputs["fp16_output"])
            assert fp16_input[0][0] == fp16_output[0][0]

        for response in server.model("test").infer(
            inputs={"string_input": [["hello"]]},
            output_memory_type="gpu",
        ):
            text_output = response.outputs["string_output"].to_string_array()
            assert text_output[0][0] == "hello"

        for response in server.model("test").infer(
            inputs={"string_input": tritonserver.Tensor.from_string_array([["hello"]])},
            output_memory_type="gpu",
        ):
            text_output = response.outputs["string_output"].to_string_array()
            text_output = response.outputs["string_output"].to_string_array()
            assert text_output[0][0] == "hello"

    def test_basic_inference(self, server_options):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        assert server.ready()

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        inputs = {
            "fp16_input": numpy.random.rand(1, 100).astype(dtype=numpy.float16),
            "bool_input": numpy.random.rand(1, 100).astype(dtype=numpy.bool_),
        }

        for response in server.model("test").infer(
            inputs=inputs,
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            for input_name, input_value in inputs.items():
                output_value = response.outputs[input_name.replace("input", "output")]
                output_value = numpy.from_dlpack(output_value)
                numpy.testing.assert_array_equal(input_value, output_value)

        # test normal bool
        inputs = {"bool_input": [[True, False, False, True]]}

        for response in server.model("test").infer(
            inputs=inputs,
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            for input_name, input_value in inputs.items():
                output_value = numpy.from_dlpack(
                    response.outputs[input_name.replace("input", "output")]
                )
                numpy.testing.assert_array_equal(input_value, output_value)

    def test_parameters(self, server_options):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        assert server.ready()

        server.load(
            "test",
            {
                "config": json.dumps(
                    {
                        "backend": "python",
                        "parameters": {"decoupled": {"string_value": "False"}},
                    }
                )
            },
        )

        fp16_input = numpy.random.rand(1, 100).astype(dtype=numpy.float16)

        input_parameters = {
            "int_parameter": 0,
            "float_parameter": 0.5,
            "bool_parameter": False,
            "string_parameter": "test",
        }
        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            parameters=input_parameters,
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            numpy.testing.assert_array_equal(fp16_input, fp16_output)
            output_parameters = json.loads(
                response.outputs["output_parameters"].to_string_array()[0]
            )
            assert input_parameters == output_parameters

        with pytest.raises(tritonserver.InvalidArgumentError):
            input_parameters = {
                "invalid": {"test": 1},
            }

            server.model("test").infer(
                inputs={"fp16_input": fp16_input},
                parameters=input_parameters,
                output_memory_type="cpu",
                raise_on_error=True,
            )

        with pytest.raises(tritonserver.InvalidArgumentError):
            input_parameters = {
                "invalid": None,
            }

            server.model("test").infer(
                inputs={"fp16_input": fp16_input},
                parameters=input_parameters,
                output_memory_type="cpu",
                raise_on_error=True,
            )
