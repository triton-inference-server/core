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

import asyncio
import queue
import time
import unittest

import numpy
import pytest
import tritonserver

try:
    import cupy
except ImportError:
    cupy = None

server_options = tritonserver.Options(
    server_id="TestServer",
    model_repository="/workspace/test/test_api_models",
    log_verbose=0,
    exit_on_error=False,
)


class ModelTests(unittest.TestCase):
    def test_create_request(self):
        server = tritonserver.Server(server_options).start(wait_until_ready=True)

        request = server.models()["test"].create_request()

        request = tritonserver.InferenceRequest(server.model("test"))

        request = tritonserver.InferenceRequest(server.model("test"), _server="foo")

        pass


class TensorTests(unittest.TestCase):
    def test_cpu_to_gpu(self):
        cpu_array = numpy.random.rand(1, 3, 100, 100).astype(numpy.float32)
        cpu_tensor = tritonserver.Tensor.from_dlpack(cpu_array)
        gpu_tensor = cpu_tensor.to_device("gpu")


class ServerTests(unittest.TestCase):
    server_options = tritonserver.Options(
        server_id="TestServer",
        model_repository="test_api_models",
        log_verbose=0,
        exit_on_error=False,
    )

    def test_not_started(self):
        server = tritonserver.Server()
        with self.assertRaises(tritonserver.InvalidArgumentError):
            server.ready()

    def test_invalid_option_type(self):
        server = tritonserver.Server(server_id=1)
        with self.assertRaises(TypeError):
            server.start()

        server = tritonserver.Server(model_repository=1)
        with self.assertRaises(TypeError):
            server.start()

    def test_invalid_repo(self):
        with self.assertRaises(tritonserver.InternalError):
            server = tritonserver.Server(model_repository="foo").start()

    def test_ready(self):
        server = tritonserver.Server(ServerTests.server_options).start()
        self.assertTrue(server.ready())


class InferenceTests(unittest.TestCase):
    server_options = tritonserver.Options(
        server_id="TestServer",
        model_repository="test_api_models",
        log_verbose=1,
        exit_on_error=False,
        exit_timeout=5,
    )

    def test_basic_inference(self):
        server = tritonserver.Server(InferenceTests.server_options).start(
            wait_until_ready=True
        )

        self.assertTrue(server.ready())

        fp16_input = numpy.array([[5]], dtype=numpy.float16)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            output_memory_type="cpu",
            raise_on_error=True,
        ):
            fp16_output = numpy.from_dlpack(response.outputs["fp16_output"])
            self.assertEqual(fp16_input, fp16_output)

        for response in server.model("test").infer(
            inputs={"fp16_input": fp16_input},
            output_memory_type="gpu",
            raise_on_error=True,
        ):
            fp16_output = cupy.from_dlpack(response.outputs["fp16_output"])
            self.assertEqual(fp16_input[0][0], fp16_output[0][0])
