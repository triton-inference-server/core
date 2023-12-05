import asyncio
import queue
import time
import unittest

import cupy
import numpy
import pytest
import tritonserver


class TrtionServerAPITest(unittest.TestCase):
    def test_not_started(self):
        server = tritonserver.Server()
        with self.assertRaises(tritonserver.InvalidArgumentError):
            server.is_ready()

    def test_gpu_memory(self):
        #        server = tritonserver.Server(
        #           model_repository_paths=["/workspace/models"],
        #          log_verbose=True,
        #         log_error=True,
        #    )
        server = tritonserver.Server()
        server.start(
            model_repository="/workspace/models",
            log_verbose=True,
            log_error=True,
            blocking=True,
        )

        test = server.get_model("test")
        fp16_input = cupy.array([[5], [6], [7], [8]], dtype=numpy.float16)

        # text_input = numpy.array([c for c in "hello"], dtype=numpy.object_).astype(numpy.byte)

        # text_input = cupy.array(["hello"], dtype=numpy.str_),
        # text_input_device = numba.cuda.to_device(text_input)
        #        fp16_input = numpy.array([["1"]], dtype=numpy.float16)
        responses_1 = test.infer(inputs={"fp16_input": fp16_input}, request_id="1")

        for response in responses_1:
            print(response)

        server.stop()

    def test_inference(self):
        server = tritonserver.Server(
            model_repository_paths=["/workspace/models"],
            #           log_verbose=True,
            #            log_error=True,
        )
        server.start()
        while not server.is_ready():
            pass

        response_queue = queue.SimpleQueue()

        test = server.get_model("test")
        test_2 = server.get_model("test_2")

        inputs = {
            "text_input": numpy.array(["hello"], dtype=numpy.object_),
            "fp16_input": numpy.array([["1"]], dtype=numpy.float16),
        }

        responses_1 = test.infer(
            inputs=inputs, request_id="1", response_queue=response_queue
        )
        responses_2 = test.infer(
            inputs=inputs, request_id="2", response_queue=response_queue
        )

        responses_3 = test_2.infer(inputs=inputs)

        for response in responses_3:
            print(response)

        count = 0
        while count < 2:
            response = response_queue.get()
            count += 1
            print(response, count)
            print(response.outputs["text_output"])
            print(bytes(response.outputs["text_output"][0]))
            print(type(response.outputs["text_output"][0]))
            print(response.outputs["fp16_output"])
            print(type(response.outputs["fp16_output"][0]))

        #     for response in test.infer(inputs=inputs):
        #        print(response.outputs["text_output"])
        #       print(response.outputs["fp16_output"])

        print(test.statistics())
        print(test_2.statistics())

        #        print(server.metrics())

        server.stop()


class AsyncInferenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_inference(self):
        server = tritonserver.Server(
            model_repository_paths=["/workspace/models"],
            #                                         log_verbose=True,
            #                                        log_error=True)
        )
        server.start()
        while not server.is_ready():
            pass

        test = server.models["test"]

        inputs = {
            "text_input": numpy.array(["hello"], dtype=numpy.object_),
            "fp16_input": numpy.array([["1"]], dtype=numpy.float16),
        }

        response_queue = asyncio.Queue()
        responses = test.async_infer(
            inputs=inputs, response_queue=response_queue, request_id="1"
        )
        responses_2 = test.async_infer(
            inputs=inputs, response_queue=response_queue, request_id="2"
        )
        responses_3 = test.async_infer(
            inputs=inputs, response_queue=response_queue, request_id="3"
        )
        responses.cancel()
        async for response in responses:
            print(response.outputs["text_output"])
            print(response.outputs["fp16_output"])
            print(response.request_id)

        count = 0
        while count < 3:
            response = await response_queue.get()
            print(response, count)
            count += 1

        server.stop()

        pass
