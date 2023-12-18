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

"""In process Python API for Triton Inference Server."""

from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import inspect
import json
import queue
import time
from dataclasses import dataclass
from typing import Annotated, Any

from tritonserver import _c as _triton_bindings
from tritonserver._api import _datautils
from tritonserver._api._datautils import MemoryAllocator

# :MetricFamily Metric group created with MetricKind, name, and description
from tritonserver._c import TRITONSERVER_InstanceGroupKind as InstanceGroupKind
from tritonserver._c import TRITONSERVER_LogFormat as LogFormat
from tritonserver._c import TRITONSERVER_MetricFamily as MetricFamily
from tritonserver._c import TRITONSERVER_MetricFormat as MetricFormat
from tritonserver._c import TRITONSERVER_MetricKind as MetricKind
from tritonserver._c import TRITONSERVER_ModelBatchFlag as ModelBatchFlag
from tritonserver._c import TRITONSERVER_ModelControlMode as ModelControlMode
from tritonserver._c import TRITONSERVER_ModelIndexFlag as ModelIndexFlag
from tritonserver._c import TRITONSERVER_ModelTxnPropertyFlag as ModelTxnPropertyFlag
from tritonserver._c import TRITONSERVER_RateLimitMode as RateLimitMode

uint = Annotated[int, ctypes.c_uint]


@dataclass
class RateLimiterResource:
    """Resource count for rate limiting.

    The amount of a resource available.

    See :c:func:`TRITONSERVER_ServerOptionsAddRateLimiterResource`

    Parameters
    ----------
    name : str
         Name of resource
    count : uint
          Count of resource available
    device : uint
           The id of the device
    """

    name: str
    count: uint
    device: uint


@dataclass
class ModelLoadDeviceLimit:
    """Memory limit for loading models on a device.

    See :c:func:`TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit`

    Parameters
    ----------
    kind : InstanceGroupKind
         The kind of device
    device : uint
           The id of the device
    fraction : float
             The limit on memory usage as a fraction
    """

    kind: InstanceGroupKind
    device: uint
    fraction: float


@dataclass(slots=True)
class Options:
    """Server Options.

    Parameters
    ----------
    server_id : str
              Id for server.
    model_repository : str | list[str]
                     Model repository path(s).
                     At least one path is required
                     See :c:func:`TRITONSERVER_ServerOptionsSetModelRepositoryPath`
    model_control_mode : ModelControlMode
                       Model control mode.
                       See :c:func:`TRITONSERVER_ServerOptionsSetModelControlMode`
    startup_models : list[str]
                   List of models to load at startup.
                   See :c:func:`TRITONSERVER_ServerOptionsSetStartupModel`
    strict_model_config : bool
                        Enable or disable strict model configuration.
                        See :c:func:`TRITONSERVER_ServerOptionsSetStrictModelConfig`
    rate_limiter_mode : RateLimitMode
                      Rate limit mode.
                      See :c:func:`TRITONSERVER_ServerOptionsSetRateLimiterMode`
    rate_limiter_resources : list[RateLimiterResource]
                      Rate limited resources.
                      See :c:func:`TRITONSERVER_ServerOptionsAddRateLimiterResource`
    pinned_memory_pool_size : uint
                            Total pinned memory size.
                            See :c:func:`TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize`
    cuda_memory_pool_sizes : dict[uint, uint]
                           Total cuda memory pool size per device.
                           See :c:func:`TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize`
    cache_config : dict[str, dict[str, Any]]
                 Key value configuration parameters for cache provider.
                 See :c:func:`TRITONSERVER_ServerOptionsSetCacheConfig`
    cache_directory : str
                    Directory for cache provider implementations.
                    See :c:func:`TRITONSERVER_ServerOptionsSetCacheDirectory`
    min_supported_compute_capability : float
                                     Minimum required cuda compute capability.
                                     See :c:func:`TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability`
    exit_on_error : bool
                  Whether to exit on an initialization error.
                  See :c:func:`TRITONSERVER_ServerOptionsSetExitOnError`

    """

    server_id: str = "triton"
    model_repository: str | list[str] = dataclasses.field(default_factory=list[str])
    model_control_mode: ModelControlMode = ModelControlMode.NONE
    startup_models: list[str] = dataclasses.field(default_factory=list[str])
    strict_model_config: bool = True

    rate_limiter_mode: RateLimitMode = RateLimitMode.OFF
    rate_limiter_resources: list[RateLimiterResource] = dataclasses.field(
        default_factory=list[RateLimiterResource]
    )

    pinned_memory_pool_size: uint = 1 << 28
    cuda_memory_pool_sizes: dict[uint, uint] = dataclasses.field(
        default_factory=dict[uint, uint]
    )

    #   response_cache_size: Annotated[int, ctypes.c_uint] = 0
    cache_config: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict[str, dict[str, Any]]
    )
    cache_directory: str = "/opt/tritonserver/caches"

    min_supported_compute_capability: float = 6.0

    exit_on_error: bool = True
    strict_readiness: bool = True
    exit_timeout: uint = 30
    buffer_manager_thread_count: uint = 0
    model_load_thread_count: uint = 4
    model_namespacing: bool = False

    log_file: str = None
    log_info: bool = False
    log_warn: bool = False
    log_error: bool = False
    log_format: LogFormat = LogFormat.DEFAULT
    log_verbose: bool = False

    metrics: bool = True
    gpu_metrics: bool = True
    cpu_metrics: bool = True
    metrics_interval: uint = 2000

    backend_directory: str = "/opt/tritonserver/backends"
    repo_agent_directory: str = "/opt/tritonserver/repoagents"
    model_load_device_limits: list[ModelLoadDeviceLimit] = dataclasses.field(
        default_factory=list[ModelLoadDeviceLimit]
    )
    backend_configuration: dict[str, dict[str, str]] = dataclasses.field(
        default_factory=dict[str, dict[str, str]]
    )
    host_policies: dict[str, dict[str, str]] = dataclasses.field(
        default_factory=dict[str, dict[str, str]]
    )
    metrics_configuration: dict[str, dict[str, str]] = dataclasses.field(
        default_factory=dict[str, dict[str, str]]
    )

    def _create_server_options(self):
        options = _triton_bindings.TRITONSERVER_ServerOptions()

        options.set_server_id(self.server_id)

        if not isinstance(self.model_repository, list):
            self.model_repository = [self.model_repository]
        for model_repository_path in self.model_repository:
            options.set_model_repository_path(model_repository_path)
        options.set_model_control_mode(self.model_control_mode)

        for startup_model in self.startup_models:
            options.set_startup_model(startup_model)

        options.set_strict_model_config(self.strict_model_config)
        options.set_rate_limiter_mode(self.rate_limiter_mode)

        for rate_limiter_resource in self.rate_limiter_resources:
            options.set_rate_limiter_resouces(
                rate_limiter_resource.name,
                rate_limiter_resource.count,
                rate_limiter_resource.device,
            )
        options.set_pinned_memory_pool_byte_size(self.pinned_memory_pool_size)

        for device, memory_size in self.cuda_memory_pool_sizes.items():
            options.set_cuda_memory_pool_byte_size(device, memory_size)
        for cache_name, settings in self.cache_config:
            options.set_cache_config(cache_name, json.dumps(settings))

        options.set_cache_directory(self.cache_directory)
        options.set_min_supported_compute_capability(
            self.min_supported_compute_capability
        )
        options.set_exit_on_error(self.exit_on_error)
        options.set_strict_readiness(self.strict_readiness)
        options.set_exit_timeout(self.exit_timeout)
        options.set_buffer_manager_thread_count(self.buffer_manager_thread_count)
        options.set_model_load_thread_count(self.model_load_thread_count)
        options.set_model_namespacing(self.model_namespacing)

        if self.log_file:
            options.set_log_file(self.log_file)

        options.set_log_info(self.log_info)
        options.set_log_warn(self.log_warn)
        options.set_log_error(self.log_error)
        options.set_log_format(self.log_format)
        options.set_log_verbose(self.log_verbose)
        options.set_metrics(self.metrics)
        options.set_cpu_metrics(self.cpu_metrics)
        options.set_gpu_metrics(self.gpu_metrics)
        options.set_metrics_interval(self.metrics_interval)
        options.set_backend_directory(self.backend_directory)
        options.set_repo_agent_directory(self.repo_agent_directory)

        for model_load_device_limit in self.model_load_device_limits:
            options.set_model_load_device_limit(
                model_load_device_limit.kind,
                model_load_device_limit.device,
                model_load_device_limit.fraction,
            )

        for host_policy, settings in self.host_policies.items():
            for setting_name, setting_value in settings.items():
                options.set_host_policy(host_policy, setting_name, setting_value)

        for config_name, settings in self.metrics_configuration.items():
            for setting_name, setting_value in settings.items():
                options.set_metrics_config(config_name, setting_name, setting_value)

        for backend, settings in self.backend_configuration.items():
            for setting_name, setting_value in settings.items():
                options.set_backend_config(backend, setting_name, setting_value)

        return options


class ModelDictionary(dict):
    def __init__(self, server, models):
        super().__init__()
        for model in models:
            self[(model.name, model.version)] = model
        self._server = server
        self._model_names = [x[0] for x in self.keys()]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                raise KeyError(f"Unknown Model: {key}") from None
        else:
            if key in self._model_names:
                return Model(self._server, name=key, version=-1)
            else:
                raise KeyError(f"Unknown Model: {key}")


class Server:
    UNLOADED_STATES = [None, "UNAVAILABLE"]

    def __init__(self, options: Options = None, **kwargs):
        if options is None:
            options = Options(**kwargs)
        self._options = options
        self._server = Server._UnstartedServer()

    def start(self, blocking=False, polling_interval=0.1):
        if not isinstance(self._server, Server._UnstartedServer):
            raise _triton_bindings.InvalidArgumentError("Server already started")

        self._server = _triton_bindings.TRITONSERVER_Server(
            self._options._create_server_options()
        )
        while blocking and not self.is_ready():
            time.sleep(polling_interval)

    def stop(self):
        self._server.stop()
        self._server = Server._UnstartedServer()

    def unregister_model_repository(self, repository_path: str):
        self._server.unregister_model_repository(repository_path)

    def register_model_repository(
        self, repository_path: str, name_mapping: dict[str, str]
    ):
        """Add a new model repository.

        Adds a new model repository.

        See :c:func:`TRITONSERVER_ServerRegisterModelRepository`

        Parameters
        ----------
        repository_path : str
            repository path
        name_mapping : dict[str, str]
            override model names

        Examples
        --------
        server.register_model_repository("/workspace/models",{"test_model":"new_model"})

        """
        name_mapping_list = [
            _triton_bindings.TRITONSERVER_Parameter(name, value)
            for name, value in name_mapping.items()
        ]

        self._server.register_model_repository(repository_path, name_mapping_list)

    def poll_model_repository(self):
        return self._server.poll_model_repository()

    def metadata(self):
        return json.loads(self._server.metadata().serialize_to_json())

    def is_live(self):
        return self._server.is_live()

    def is_ready(self):
        return self._server.is_ready()

    def get_model(self, model_name, model_version=-1):
        return Model(self._server, model_name, model_version)

    @property
    def models(self):
        return ModelDictionary(self._server, self._model_index())

    def _model_index(self, ready=False):
        models = json.loads(self._server.model_index(ready).serialize_to_json())

        for model in models:
            if "version" in model:
                model["version"] = int(model["version"])

        return [Model(self._server, **model) for model in models]

    def load_model(
        self, model_name: str, parameters: dict[str, str | int | bool | bytes] = None
    ):
        if parameters:
            parameter_list = [
                _triton_bindings.TRITONSERVER_Parameter(name, value)
                for name, value in parameters.items()
            ]
            self._server.load_model_with_parameters(model_name, parameter_list)
        else:
            self._server.load_model(model_name)
        return self.get_model(model_name)

    def unload_model(
        self,
        model: str | Model = None,
        unload_dependents: bool = False,
        blocking: bool = False,
        polling_interval: float = 0.1,
        **kwargs,
    ):
        """Unload model and dependents (optional)

        Parameters
        ----------
        model : str | Model
            model name or model object
        unload_dependents : bool
            if True unload models dependent on this (ensembles)
        blocking: bool
            if True block until model is unavailable

        Examples
        --------
        FIXME: Add docs.

        """
        if model is None:
            model = Model(**kwargs)
        elif isinstance(model, str):
            model = Model(self._server, model)

        if unload_dependents:
            self._server.unload_model_and_dependents(model.name)
        else:
            self._server.unload_model(model.name)
        if blocking:
            model_versions = [key for key in self.models.keys() if key[0] == model.name]
            while [
                key
                for key in model_versions
                if self.models[key].state not in Server.UNLOADED_STATES
            ]:
                time.sleep(polling_interval)

    def metrics(self, metric_format: MetricFormat = MetricFormat.PROMETHEUS):
        return self._server.metrics().formatted(metric_format)

    class _UnstartedServer(object):
        def __init__(self):
            pass

        def __getattribute__(self, name):
            raise _triton_bindings.InvalidArgumentError("Server not started")

        def __setattr__(self, name, value):
            raise _triton_bindings.InvalidArgumentError("Server not started")


class Model:
    def __init__(
        self,
        server: _triton_bindings.TRITONSERVER_Server,
        name: str,
        version: int = -1,
        state: str = None,
        reason: str = None,
    ):
        self.name = name
        self.version = version
        self._server = server
        self.state = state
        self.reason = reason

    def create_inference_request(self, **kwargs):
        return InferenceRequest(model=self, _server=self._server, **kwargs)

    def async_infer(
        self, inference_request: InferenceRequest = None, **kwargs
    ) -> AsyncResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )
        server_request, response_iterator = inference_request._create_server_request(
            use_async_iterator=True
        )
        self._server.infer_async(server_request)
        return response_iterator

    def infer(
        self, inference_request: InferenceRequest = None, **kwargs
    ) -> ResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )
        server_request, response_iterator = inference_request._create_server_request()
        self._server.infer_async(server_request)
        return response_iterator

    def metadata(self):
        return json.loads(
            self._server.model_metadata(self.name, self.version).serialize_to_json()
        )

    def is_ready(self):
        return self._server.model_is_ready(self.name, self.version)

    def batch_properties(self):
        flags, _ = self._server.model_batch_properties(self.name, self.version)
        return ModelBatchFlag(flags)

    def transaction_properties(self):
        txn_properties, _ = self._server.model_transaction_properties(
            self.name, self.version
        )
        return ModelTxnPropertyFlag(txn_properties)

    def statistics(self):
        return json.loads(
            self._server.model_statistics(self.name, self.version).serialize_to_json()
        )

    def config(self, config_version=1):
        return json.loads(
            self._server.model_config(
                self.name, self.version, config_version
            ).serialize_to_json()
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s" % (
            {"name": self.name, "version": self.version, "state": self.state}
        )


class AsyncResponseIterator:
    def response_callback(self, response, flags, unused):
        try:
            response = InferenceResponse._set_from_server_response(
                self._server, self._request, response, flags
            )
            asyncio.run_coroutine_threadsafe(self._queue.put(response), self._loop)
            if self._user_queue is not None:
                asyncio.run_coroutine_threadsafe(
                    self._user_queue.put(response), self._loop
                )
            if flags == _triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
                del self._request
                self._request = None
        except Exception as e:
            _triton_bindings.TRITONSERVER_LogMessage(
                _triton_bindings.TRITONSERVER_LogLevel.ERROR,
                __file__,
                inspect.currentframe().f_lineno,
                str(e),
            )

    def __init__(self, server, request, loop=None, user_queue=None):
        self._server = server
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = asyncio.Queue()
        self._user_queue = user_queue
        self._complete = False
        self._request = request

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._complete:
            raise StopAsyncIteration
        response = await self._queue.get()
        self._complete = response.final
        return response

    def cancel(self):
        self._request.cancel()


class ResponseIterator:
    def response_callback(self, response, flags, unused):
        try:
            response = InferenceResponse._set_from_server_response(
                self._server, self._request, response, flags
            )
            self._queue.put(response)
            if self._user_queue is not None:
                self._user_queue.put(response)
            if flags == _triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
                del self._request
                self._request = None
        except Exception as e:
            _triton_bindings.TRITONSERVER_LogMessage(
                _triton_bindings.TRITONSERVER_LogLevel.ERROR,
                __file__,
                inspect.currentframe().f_lineno,
                str(e),
            )

    def __init__(self, server, request, user_queue: queue.SimpleQueue = None):
        self._queue = queue.SimpleQueue()
        self._user_queue = user_queue
        self._server = server
        self._complete = False
        self._request = request

    def __iter__(self):
        return self

    def __next__(self):
        if self._complete:
            raise StopIteration
        response = self._queue.get()
        self._complete = response.final
        return response

    def cancel(self):
        if self._request is not None:
            self._request.cancel()


@dataclass
class InferenceResponse:
    request_id: str = None
    parameters: dict = dataclasses.field(default_factory=dict)
    outputs: dict = dataclasses.field(default_factory=dict)
    error: _triton_bindings.TritonError = None
    classification_label: str = None
    final: bool = False
    _server: _triton_bindings.TRITONSERVER_Server = None
    model: Model = None

    @staticmethod
    def _set_from_server_response(server, request, response, flags):
        values = {}
        if response is None:
            if flags == _triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
                values["final"] = True
            if request.id:
                values["request_id"] = request.id
            return InferenceResponse(**values)

        try:
            response.throw_if_response_error()
        except _triton_bindings.TritonError as error:
            values["error"] = error

        if flags == _triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
            values["final"] = True

        name, version = response.model
        values["model"] = Model(server, name, version)
        values["request_id"] = response.id
        parameters = {}
        for parameter_index in range(response.parameter_count):
            name, type_, value = response.parameter(parameter_index)
            parameters[name] = value
        values["parameters"] = parameters
        outputs = {}
        for output_index in range(response.output_count):
            (name, *buffer_details) = response.output(output_index)
            memory_buffer = _datautils.MemoryBuffer.from_details(*buffer_details)
            outputs[name] = memory_buffer.value
        values["outputs"] = outputs
        values["_server"] = server

        # values["classification_label"] = response.output_classification_label()

        return InferenceResponse(**values)


@dataclass
class InferenceRequest:
    request_id: str = None
    flags: int = 0
    correlation_id: int | str = None
    priority: int = 0
    timeout: int = 0
    inputs: dict = dataclasses.field(default_factory=dict)
    parameters: dict[str, str | int | bool] = dataclasses.field(default_factory=dict)
    response_allocator: MemoryAllocator = None
    model: Model = None
    response_queue: queue.SimpleQueue | asyncio.Queue = None
    _server: _triton_bindings.TRITONSERVER_Server = None
    _serialized_inputs: dict = dataclasses.field(default_factory=dict)

    _default_allocator = _datautils.DefaultAllocator().create_response_allocator()

    def _release_request(self, request, flags, user_object):
        pass

    def _add_inputs(self, request):
        for name, value in self.inputs.items():
            memory_buffer = _datautils.MemoryBuffer.from_value(value)
            if memory_buffer.data_type == _triton_bindings.TRITONSERVER_DataType.BYTES:
                # to ensure lifetime of array
                self._serialized_inputs[name] = memory_buffer.value
            request.add_input(name, memory_buffer.data_type, memory_buffer.shape)

            request.append_input_data_with_buffer_attributes(
                name, memory_buffer.buffer_, memory_buffer.buffer_attributes
            )

    def _set_callbacks(self, request, use_async_iterator=False):
        if use_async_iterator:
            response_iterator = AsyncResponseIterator(
                self._server, request, user_queue=self.response_queue
            )
        else:
            response_iterator = ResponseIterator(
                self._server, request, user_queue=self.response_queue
            )
        request.set_release_callback(self._release_request, None)

        allocator = InferenceRequest._default_allocator

        if self.response_allocator is not None:
            allocator = self.response_allocator.create_response_allocator()

        request.set_response_callback(
            allocator,
            None,
            response_iterator.response_callback,
            None,
        )
        return response_iterator

    def _set_parameters(self, request):
        for key, value in self.parameters.items():
            if isinstance(value, str):
                request.set_string_parameter(key, value)
            elif isinstance(value, int):
                request.set_int_parameter(key, value)
            elif isinstance(value, bool):
                request.set_bool_parameter(key, value)
            else:
                raise _triton_bindings.InvalidArgumentError(
                    f"Invalid parameter type {type(value)} for key {key}"
                )

    def _create_server_request(self, use_async_iterator=False):
        request = _triton_bindings.TRITONSERVER_InferenceRequest(
            self._server, self.model.name, self.model.version
        )
        if self.request_id is not None:
            request.id = self.request_id
        request.priority_uint64 = self.priority
        request.timeout_microseconds = self.timeout
        if self.correlation_id is not None:
            if isinstance(self.correlation_id, int):
                request.correlation_id = self.correlation_id
            else:
                request.correlation_id_string = self.correlation_id
        request.flags = self.flags

        self._add_inputs(request)

        self._set_parameters(request)

        response_iterator = self._set_callbacks(request, use_async_iterator)

        return request, response_iterator


class Metric(_triton_bindings.TRITONSERVER_Metric):
    def __init__(self, family: MetricFamily, labels: dict[str, str] = None):
        if labels is not None:
            parameters = [
                _triton_bindings.TRITONSERVER_Parameter(name, value)
                for name, value in labels.items()
            ]
        else:
            parameters = []

        _triton_bindings.TRITONSERVER_Metric.__init__(self, family, parameters)
