import unittest
import json
import os

import triton_bindings


class BindingTest(unittest.TestCase):

    def test_exceptions(self):
        ex_list = [
            triton_bindings.Unknown, triton_bindings.Internal,
            triton_bindings.NotFound, triton_bindings.InvalidArgument,
            triton_bindings.Unavailable, triton_bindings.Unsupported,
            triton_bindings.AlreadyExists
        ]
        for ex_type in ex_list:
            try:
                raise ex_type("Error message")
            # 'TritonError' should catch all
            except triton_bindings.TritonError as te:
                self.assertTrue(isinstance(te, ex_type))
                self.assertEqual(str(te), "Error message")

    def test_data_type(self):
        t_list = [
            (triton_bindings.TRITONSERVER_DataType.INVALID, "<invalid>", 0),
            (triton_bindings.TRITONSERVER_DataType.BOOL, "BOOL", 1),
            (triton_bindings.TRITONSERVER_DataType.UINT8, "UINT8", 1),
            (triton_bindings.TRITONSERVER_DataType.UINT16, "UINT16", 2),
            (triton_bindings.TRITONSERVER_DataType.UINT32, "UINT32", 4),
            (triton_bindings.TRITONSERVER_DataType.UINT64, "UINT64", 8),
            (triton_bindings.TRITONSERVER_DataType.INT8, "INT8", 1),
            (triton_bindings.TRITONSERVER_DataType.INT16, "INT16", 2),
            (triton_bindings.TRITONSERVER_DataType.INT32, "INT32", 4),
            (triton_bindings.TRITONSERVER_DataType.INT64, "INT64", 8),
            (triton_bindings.TRITONSERVER_DataType.FP16, "FP16", 2),
            (triton_bindings.TRITONSERVER_DataType.FP32, "FP32", 4),
            (triton_bindings.TRITONSERVER_DataType.FP64, "FP64", 8),
            (triton_bindings.TRITONSERVER_DataType.BYTES, "BYTES", 0),
            (triton_bindings.TRITONSERVER_DataType.BF16, "BF16", 2),
        ]

        for t, t_str, t_size in t_list:
            self.assertEqual(triton_bindings.TRITONSERVER_DataTypeString(t),
                             t_str)
            self.assertEqual(
                triton_bindings.TRITONSERVER_StringToDataType(t_str), t)
            self.assertEqual(triton_bindings.TRITONSERVER_DataTypeByteSize(t),
                             t_size)

    def test_memory_type(self):
        t_list = [
            (triton_bindings.TRITONSERVER_MemoryType.CPU, "CPU"),
            (triton_bindings.TRITONSERVER_MemoryType.CPU_PINNED, "CPU_PINNED"),
            (triton_bindings.TRITONSERVER_MemoryType.GPU, "GPU"),
        ]
        for t, t_str in t_list:
            self.assertEqual(triton_bindings.TRITONSERVER_MemoryTypeString(t),
                             t_str)

    def test_parameter_type(self):
        t_list = [
            (triton_bindings.TRITONSERVER_ParameterType.STRING, "STRING"),
            (triton_bindings.TRITONSERVER_ParameterType.INT, "INT"),
            (triton_bindings.TRITONSERVER_ParameterType.BOOL, "BOOL"),
            (triton_bindings.TRITONSERVER_ParameterType.BYTES, "BYTES"),
        ]
        for t, t_str in t_list:
            self.assertEqual(
                triton_bindings.TRITONSERVER_ParameterTypeString(t), t_str)

    def test_parameter(self):
        # C API doesn't provide additional API for parameter, can only test
        # New/Delete unless we mock the implementation to expose more info.
        str_param = triton_bindings.TRITONSERVER_Parameter(
            "str_key", "str_value")
        int_param = triton_bindings.TRITONSERVER_Parameter("int_key", 123)
        bool_param = triton_bindings.TRITONSERVER_Parameter("bool_key", True)
        # bytes parameter doesn't own the buffer
        b = bytes("abc", 'utf-8')
        bytes_param = triton_bindings.TRITONSERVER_Parameter("bytes_key", b)
        del str_param
        del int_param
        del bool_param
        del bytes_param
        import gc
        gc.collect()

    def test_instance_kind(self):
        t_list = [
            (triton_bindings.TRITONSERVER_InstanceGroupKind.AUTO, "AUTO"),
            (triton_bindings.TRITONSERVER_InstanceGroupKind.CPU, "CPU"),
            (triton_bindings.TRITONSERVER_InstanceGroupKind.GPU, "GPU"),
            (triton_bindings.TRITONSERVER_InstanceGroupKind.MODEL, "MODEL"),
        ]
        for t, t_str in t_list:
            self.assertEqual(
                triton_bindings.TRITONSERVER_InstanceGroupKindString(t), t_str)

    def test_log(self):
        # This test depends on 'TRITONSERVER_ServerOptions' operates properly
        # to modify log settings.

        # Direct Triton to log message into a file so that the log may be
        # retrieved on the Python side. Otherwise the log will be default
        # on stderr and Python utils can not redirect the pipe on Triton side.
        log_file = "triton_binding_test_log_output.txt"
        default_format_regex = r'[0-9][0-9][0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]'
        iso8601_format_regex = r'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        try:
            options = triton_bindings.TRITONSERVER_ServerOptions()
            # Enable subset of log levels
            options.set_log_file(log_file)
            options.set_log_info(True)
            options.set_log_warn(False)
            options.set_log_error(True)
            options.set_log_verbose(0)
            options.set_log_format(
                triton_bindings.TRITONSERVER_LogFormat.DEFAULT)
            for ll, enabled in [
                (triton_bindings.TRITONSERVER_LogLevel.INFO, True),
                (triton_bindings.TRITONSERVER_LogLevel.WARN, False),
                (triton_bindings.TRITONSERVER_LogLevel.ERROR, True),
                (triton_bindings.TRITONSERVER_LogLevel.VERBOSE, False),
            ]:
                self.assertEqual(triton_bindings.TRITONSERVER_LogIsEnabled(ll),
                                 enabled)
            # Write message to each of the log level
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.INFO, "filename", 123,
                "info_message")
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.WARN, "filename", 456,
                "warn_message")
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.ERROR, "filename", 789,
                "error_message")
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.VERBOSE, "filename", 147,
                "verbose_message")
            with open(log_file, "r") as f:
                log = f.read()
                # Check level
                self.assertRegex(log, r'filename:123.*info_message')
                self.assertNotRegex(log, r'filename:456.*warn_message')
                self.assertRegex(log, r'filename:789.*error_message')
                self.assertNotRegex(log, r'filename:147.*verbose_message')
                # Check format "MMDD hh:mm:ss.ssssss".
                self.assertRegex(log, default_format_regex)
                # sanity check that there is no log with other format "YYYY-MM-DDThh:mm:ssZ L"
                self.assertNotRegex(log, iso8601_format_regex)
            # Test different format
            options.set_log_format(
                triton_bindings.TRITONSERVER_LogFormat.ISO8601)
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.INFO, "fn", 258,
                "info_message")
            with open(log_file, "r") as f:
                log = f.read()
                self.assertRegex(log, r'fn:258.*info_message')
                self.assertRegex(log, iso8601_format_regex)
        finally:
            # Must make sure the log settings are reset as the logger is unique
            # within the process
            options.set_log_file("")
            options.set_log_info(False)
            options.set_log_warn(False)
            options.set_log_error(False)
            options.set_log_verbose(0)
            options.set_log_format(
                triton_bindings.TRITONSERVER_LogFormat.DEFAULT)
            os.remove(log_file)

    def test_buffer_attributes(self):
        expected_memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU_PINNED
        expected_memory_type_id = 4
        expected_byte_size = 1024
        buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type_id = expected_memory_type_id
        self.assertEqual(buffer_attributes.memory_type_id,
                         expected_memory_type_id)
        buffer_attributes.memory_type = expected_memory_type
        self.assertEqual(buffer_attributes.memory_type, expected_memory_type)
        buffer_attributes.byte_size = expected_byte_size
        self.assertEqual(buffer_attributes.byte_size, expected_byte_size)
        # cuda_ipc_handle is supposed to be cudaIpcMemHandle_t, must initialize buffer
        # of that size to avoid segfault. The handle getter/setter is different from other
        # attributes that different pointers may be returned from the getter, but the byte
        # content pointed by the pointer should be the same
        from array import array
        import ctypes
        handle_byte_size = 64
        mock_handle = array("b", [i for i in range(handle_byte_size)])
        buffer_attributes.cuda_ipc_handle = mock_handle.buffer_info()[0]
        res_arr = (ctypes.c_char * handle_byte_size).from_address(
            buffer_attributes.cuda_ipc_handle)
        for i in range(handle_byte_size):
            self.assertEqual(int.from_bytes(res_arr[i], "big"), mock_handle[i])

    def test_allocator(self):

        def alloc_fn(allocator, tensor_name, byte_size, memory_type,
                     memory_type_id, user_object):
            return (123, None, triton_bindings.TRITONSERVER_MemoryType.GPU, 1)

        def release_fn(allocator, buffer, buffer_user_object, byte_size,
                       memory_type, memory_type_id):
            pass

        def start_fn(allocator, user_object):
            pass

        def query_fn(allocator, user_object, tensor_name, byte_size,
                     memory_type, memory_type_id):
            return (triton_bindings.TRITONSERVER_MemoryType.GPU, 1)

        def buffer_fn(allocator, tensor_name, buffer_attribute, user_object,
                      buffer_user_object):
            return buffer_attribute

        # allocator without start_fn
        allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
            alloc_fn, release_fn)
        del allocator
        import gc
        gc.collect()

        # allocator with start_fn
        allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
            alloc_fn, release_fn, start_fn)
        allocator.set_buffer_attributes_function(buffer_fn)
        allocator.set_query_function(query_fn)

    def test_options(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()

        # Generic
        options.set_server_id("server_id")
        options.set_min_supported_compute_capability(7.0)
        options.set_exit_on_error(False)
        options.set_strict_readiness(False)
        options.set_exit_timeout(30)

        # Models
        options.set_model_repository_path("model_repo_0")
        options.set_model_repository_path("model_repo_1")
        for m in [
                triton_bindings.TRITONSERVER_ModelControlMode.NONE,
                triton_bindings.TRITONSERVER_ModelControlMode.POLL,
                triton_bindings.TRITONSERVER_ModelControlMode.EXPLICIT
        ]:
            options.set_model_control_mode(m)
        options.set_startup_model("*")
        options.set_strict_model_config(True)
        options.set_model_load_thread_count(2)
        options.set_model_namespacing(True)
        # Only support Kind GPU for now
        options.set_model_load_device_limit(
            triton_bindings.TRITONSERVER_InstanceGroupKind.GPU, 0, 0.5)
        for k in [
                triton_bindings.TRITONSERVER_InstanceGroupKind.AUTO,
                triton_bindings.TRITONSERVER_InstanceGroupKind.CPU,
                triton_bindings.TRITONSERVER_InstanceGroupKind.MODEL
        ]:
            with self.assertRaises(triton_bindings.TritonError) as context:
                options.set_model_load_device_limit(k, 0, 0)
            self.assertTrue("not supported" in str(context.exception))

        # Backend
        options.set_backend_directory("backend_dir_0")
        options.set_backend_directory("backend_dir_1")
        options.set_backend_config("backend_name", "setting", "value")

        # Rate limiter
        for r in [
                triton_bindings.TRITONSERVER_RateLimitMode.OFF,
                triton_bindings.TRITONSERVER_RateLimitMode.EXEC_COUNT
        ]:
            options.set_rate_limiter_mode(r)
        options.add_rate_limiter_resource("shared_resource", 4, -1)
        options.add_rate_limiter_resource("device_resource", 1, 0)
        # memory pools
        options.set_pinned_memory_pool_byte_size(1024)
        options.set_cuda_memory_pool_byte_size(0, 2048)
        # cache
        options.set_response_cache_byte_size(4096)
        options.set_cache_config(
            "cache_name",
            json.dumps({
                "config_0": "value_0",
                "config_1": "value_1"
            }))
        options.set_cache_directory("cache_dir_0")
        options.set_cache_directory("cache_dir_1")
        # Log
        try:
            options.set_log_file("some_file")
            options.set_log_info(True)
            options.set_log_warn(True)
            options.set_log_error(True)
            options.set_log_verbose(2)
            for f in [
                    triton_bindings.TRITONSERVER_LogFormat.DEFAULT,
                    triton_bindings.TRITONSERVER_LogFormat.ISO8601
            ]:
                options.set_log_format(f)
        finally:
            # Must make sure the log settings are reset as the logger is unique
            # within the process
            options.set_log_file("")
            options.set_log_info(False)
            options.set_log_warn(False)
            options.set_log_error(False)
            options.set_log_verbose(0)
            options.set_log_format(
                triton_bindings.TRITONSERVER_LogFormat.DEFAULT)

        # Metrics
        options.set_gpu_metrics(True)
        options.set_cpu_metrics(True)
        options.set_metrics_interval(5)
        options.set_metrics_config("metrics_group", "setting", "value")

        # Misc..
        with self.assertRaises(triton_bindings.TritonError) as context:
            options.set_host_policy("policy_name", "setting", "value")
        self.assertTrue(
            "Unsupported host policy setting" in str(context.exception))
        options.set_repo_agent_directory("repo_agent_dir_0")
        options.set_repo_agent_directory("repo_agent_dir_1")
        options.set_buffer_manager_thread_count(4)


if __name__ == "__main__":
    unittest.main()
