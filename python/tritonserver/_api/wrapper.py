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

"""In process Python API for Triton Inference Server."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import json
import queue
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Annotated, Any, Optional, TypedDict

import tritonserver._c.triton_bindings as _triton_bindings
from tritonserver._api import _datautils
from tritonserver._api._datautils import MemoryAllocator
from tritonserver._c import InternalError
from tritonserver._c.triton_bindings import InvalidArgumentError
from tritonserver._c.triton_bindings import (
    TRITONSERVER_InstanceGroupKind as InstanceGroupKind,
)
from tritonserver._c.triton_bindings import TRITONSERVER_LogFormat as LogFormat
from tritonserver._c.triton_bindings import TRITONSERVER_MetricFamily as MetricFamily
from tritonserver._c.triton_bindings import TRITONSERVER_MetricFormat as MetricFormat
from tritonserver._c.triton_bindings import TRITONSERVER_MetricKind as MetricKind
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelBatchFlag as ModelBatchFlag,
)
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelControlMode as ModelControlMode,
)
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelIndexFlag as ModelIndexFlag,
)
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelTxnPropertyFlag as ModelTxnPropertyFlag,
)
from tritonserver._c.triton_bindings import TRITONSERVER_RateLimitMode as RateLimitMode
from tritonserver._c.triton_bindings import UnavailableError, UnsupportedError
from typing_extensions import NotRequired, Unpack

uint = Annotated[int, ctypes.c_uint]


@dataclass(slots=True)
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


@dataclass(slots=True)
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
    model_repository: str | list[str] = field(default_factory=list[str])
    model_control_mode: ModelControlMode = ModelControlMode.NONE
    startup_models: list[str] = field(default_factory=list[str])
    strict_model_config: bool = True

    rate_limiter_mode: RateLimitMode = RateLimitMode.OFF
    rate_limiter_resources: list[RateLimiterResource] = field(
        default_factory=list[RateLimiterResource]
    )

    pinned_memory_pool_size: uint = 1 << 28
    cuda_memory_pool_sizes: dict[uint, uint] = field(default_factory=dict[uint, uint])

    #   response_cache_size: Annotated[int, ctypes.c_uint] = 0
    cache_config: dict[str, dict[str, Any]] = field(
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

    log_file: Optional[str] = None
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
    model_load_device_limits: list[ModelLoadDeviceLimit] = field(
        default_factory=list[ModelLoadDeviceLimit]
    )
    backend_configuration: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )
    host_policies: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )
    metrics_configuration: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )

    def _create_TRITONSERVER_ServerOptions(
        self,
    ) -> _triton_bindings.TRITONSERVER_ServerOptions:
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
            options.add_rate_limiter_resource(
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
    def __init__(self, server, models) -> None:
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


@dataclass
class ServerMetadata:
    name: Optional[str] = None
    version: Optional[str] = None
    extensions: Optional[list[str]] = None


class Server:
    def __init__(
        self, options: Optional[Options] = None, **kwargs: Unpack[Options]
    ) -> None:
        if options is None:
            options = Options(**kwargs)
        self._options = options
        self._server = Server._UnstartedServer()

    def start(
        self,
        wait_until_ready: bool = False,
        polling_interval: float = 0.1,
        timeout: Optional[float] = None,
    ) -> None:
        if not isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server already started")

        self._server = _triton_bindings.TRITONSERVER_Server(
            self._options._create_TRITONSERVER_ServerOptions()
        )
        start_time = time.time()
        while (
            wait_until_ready
            and not self.ready()
            and ((timeout is None) or (time.time() - start_time) < timeout)
        ):
            time.sleep(polling_interval)
        if not self.ready():
            raise UnavailableError("Timeout before ready")

    def stop(self) -> None:
        self._server.stop()
        self._server = Server._UnstartedServer()

    def unregister_model_repository(self, repository_path: str) -> None:
        self._server.unregister_model_repository(repository_path)

    def register_model_repository(
        self, repository_path: str, name_mapping: dict[str, str]
    ) -> None:
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

    def poll_model_repository(self) -> None:
        return self._server.poll_model_repository()

    def metadata(self) -> dict[str, Any]:
        return json.loads(self._server.metadata().serialize_to_json())

    def get_metadata(self) -> dict[str, Any]:
        return json.loads(self._server.metadata().serialize_to_json())

    def live(self) -> bool:
        return self._server.is_live()

    def is_live(self) -> bool:
        return self._server.is_live()

    def ready(self) -> bool:
        return self._server.is_ready()

    def is_ready(self) -> bool:
        return self._server.is_ready()

    def model(self, model_name: str, model_version: int = -1) -> Model:
        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")
        return Model(self._server, model_name, model_version)

    def get_model(self, model_name, model_version=-1) -> Model:
        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")

        return Model(self._server, model_name, model_version)

    def models(self, exclude_not_ready: bool = False) -> ModelDictionary:
        return ModelDictionary(self._server, self._model_index(exclude_not_ready))

    def get_models(self, exclude_not_ready: bool = False) -> ModelDictionary:
        return ModelDictionary(self._server, self._model_index(exclude_not_ready))

    def model_index(self, exclude_not_ready: bool = False) -> ModelDictionary:
        return ModelDictionary(self._server, self._model_index(exclude_not_ready))

    def load_model(
        self,
        model_name: str,
        parameters: Optional[dict[str, str | int | bool | bytes]] = None,
    ) -> Model:
        if parameters is not None:
            parameter_list = [
                _triton_bindings.TRITONSERVER_Parameter(name, value)
                for name, value in parameters.items()
            ]
            self._server.load_model_with_parameters(model_name, parameter_list)
        else:
            self._server.load_model(model_name)
        return self.model(model_name)

    def unload_model(
        self,
        model: Optional[str | Model] = None,
        unload_dependents: bool = False,
        wait_until_unloaded: bool = False,
        polling_interval: float = 0.1,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
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
        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")

        if model is None:
            kwargs["server"] = self._server
            model = Model(**kwargs)
        elif isinstance(model, str):
            model = Model(self._server, model)

        if unload_dependents:
            self._server.unload_model_and_dependents(model.name)
        else:
            self._server.unload_model(model.name)

        if wait_until_unloaded:
            model_versions = [
                key for key in self.models().keys() if key[0] == model.name
            ]
            start_time = time.time()
            while not self._model_unloaded(model_versions) and (
                (timeout is None) or (time.time() - start_time < timeout)
            ):
                time.sleep(polling_interval)

    def metrics(self, metric_format: MetricFormat = MetricFormat.PROMETHEUS) -> str:
        return self._server.metrics().formatted(metric_format)

    def get_metrics(self, metric_format: MetricFormat = MetricFormat.PROMETHEUS) -> str:
        return self._server.metrics().formatted(metric_format)

    class _UnstartedServer(object):
        def __init__(self):
            pass

        def __getattribute__(self, name):
            raise _triton_bindings.InvalidArgumentError("Server not started")

        def __setattr__(self, name, value):
            raise _triton_bindings.InvalidArgumentError("Server not started")

    def _model_unloaded(self, model_versions: list[tuple[str, int]]) -> bool:
        model_states = self.models()
        for model_version in model_versions:
            if model_states[model_version].state not in Server._UNLOADED_STATES:
                return False
        return True

    def _model_index(self, exclude_not_ready=False) -> list[Model]:
        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")

        models = json.loads(
            self._server.model_index(exclude_not_ready).serialize_to_json()
        )

        for model in models:
            if "version" in model:
                model["version"] = int(model["version"])

        return [Model(self._server, **model) for model in models]

    _UNLOADED_STATES = [None, "UNAVAILABLE"]


class Model:
    def __init__(
        self,
        server: _triton_bindings.TRITONSERVER_Server,
        name: str,
        version: int = -1,
        state: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.name = name
        self.version = version
        self._server = server
        self.state = state
        self.reason = reason

    def create_request(self, **kwargs) -> InferenceRequest:
        return InferenceRequest(model=self, _server=self._server, **kwargs)

    def async_infer(
        self,
        inference_request: Optional[InferenceRequest] = None,
        raise_on_error: bool = False,
        **kwargs: Unpack[InferenceRequest],
    ) -> AsyncResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, asyncio.Queue)
        ):
            raise InvalidArgumentError(
                "asyncio.Queue must be used for async response iterator"
            )

        request = inference_request._create_TRITONSERVER_InferenceRequest()

        response_iterator = AsyncResponseIterator(
            self,
            self._server,
            request,
            inference_request.response_queue,
            raise_on_error,
        )

        response_allocator = _datautils.ResponseAllocator(
            inference_request.output_memory_allocator,
            inference_request.output_memory_type,
        ).create_TRITONSERVER_ResponseAllocator()

        request.set_response_callback(
            response_allocator, None, response_iterator._response_callback, None
        )

        self._server.infer_async(request)
        return response_iterator

    def infer(
        self,
        inference_request: Optional[InferenceRequest] = None,
        raise_on_error: bool = False,
        **kwargs: Unpack[InferenceRequest],
    ) -> ResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, queue.SimpleQueue)
        ):
            raise InvalidArgumentError(
                "queue.SimpleQueue must be used for response iterator"
            )

        request = inference_request._create_TRITONSERVER_InferenceRequest()
        response_iterator = ResponseIterator(
            self,
            self._server,
            request,
            inference_request.response_queue,
            raise_on_error,
        )
        response_allocator = _datautils.ResponseAllocator(
            inference_request.output_memory_allocator,
            inference_request.output_memory_type,
        ).create_TRITONSERVER_ResponseAllocator()

        request.set_response_callback(
            response_allocator, None, response_iterator._response_callback, None
        )

        self._server.infer_async(request)
        return response_iterator

    def metadata(self) -> dict[str, Any]:
        return json.loads(
            self._server.model_metadata(self.name, self.version).serialize_to_json()
        )

    def get_metadata(self) -> dict[str, Any]:
        return json.loads(
            self._server.model_metadata(self.name, self.version).serialize_to_json()
        )

    def is_ready(self) -> bool:
        return self._server.model_is_ready(self.name, self.version)

    def ready(self) -> bool:
        return self._server.model_is_ready(self.name, self.version)

    def batch_properties(self) -> ModelBatchFlag:
        flags, _ = self._server.model_batch_properties(self.name, self.version)
        return ModelBatchFlag(flags)

    def transaction_properties(self) -> ModelTxnPropertyFlag:
        txn_properties, _ = self._server.model_transaction_properties(
            self.name, self.version
        )
        return ModelTxnPropertyFlag(txn_properties)

    def statistics(self) -> dict[str, Any]:
        return json.loads(
            self._server.model_statistics(self.name, self.version).serialize_to_json()
        )

    def config(self, config_version: int = 1) -> dict[str, Any]:
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
    def __init__(
        self,
        model: Model,
        server: _triton_bindings.TRITONSERVER_Server,
        request: _triton_bindings.TRITONSERVER_InferenceRequest,
        user_queue: Optional[asyncio.Queue] = None,
        raise_on_error: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._server = server
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = asyncio.Queue()
        self._user_queue = user_queue
        self._complete = False
        self._request = request
        self._model = model
        self._raise_on_error = raise_on_error

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._complete:
            raise StopAsyncIteration
        response = await self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self) -> None:
        if self._request is not None:
            self._request.cancel()

    def _response_callback(self, response, flags, unused):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            response = InferenceResponse._from_TRITONSERVER_InferenceResponse(
                self._model, self._server, self._request, response, flags
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


class ResponseIterator:
    def __init__(
        self,
        model: Model,
        server: _triton_bindings.TRITONSERVER_Server,
        request: _triton_bindings.TRITONSERVER_InferenceRequest,
        user_queue: Optional[queue.SimpleQueue] = None,
        raise_on_error: bool = False,
    ):
        self._queue = queue.SimpleQueue()
        self._user_queue = user_queue
        self._server = server
        self._complete = False
        self._request = request
        self._model = model
        self._raise_on_error = raise_on_error

    def __iter__(self):
        return self

    def __next__(self):
        if self._complete:
            raise StopIteration
        response = self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self):
        if self._request is not None:
            self._request.cancel()

    def _response_callback(self, response, flags, unused):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            response = InferenceResponse._from_TRITONSERVER_InferenceResponse(
                self._model, self._server, self._request, response, flags
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


@dataclass
class InferenceResponse:
    model: Model
    _server: _triton_bindings.TRITONSERVER_Server
    request_id: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: Optional[_triton_bindings.TritonError] = None
    classification_label: Optional[str] = None
    final: bool = False

    @staticmethod
    def _from_TRITONSERVER_InferenceResponse(
        model: Model,
        server: _triton_bindings.TRITONSERVER_Server,
        request: _triton_bindings.TRITONSERVER_InferenceRequest,
        response,
        flags: _triton_bindings.TRITONSERVER_ResponseCompleteFlag,
    ):
        values: dict = {
            "_server": server,
            "model": model,
            "request_id": request.id,
            "final": flags == _triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL,
        }

        try:
            if response is None:
                return InferenceResponse(**values)

            try:
                response.throw_if_response_error()
            except _triton_bindings.TritonError as error:
                error.args += tuple(values.items())
                values["error"] = error

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
                (
                    name,
                    data_type,
                    shape,
                    data_ptr,
                    byte_size,
                    memory_type,
                    memory_type_id,
                    memory_buffer,
                ) = response.output(output_index)
                tensor = _datautils.Tensor.from_memory_buffer(
                    data_type, shape, memory_buffer
                )

                outputs[name] = tensor
            values["outputs"] = outputs
        except Exception as e:
            error = InternalError(f"Unexpected error in creating response object: {e}")
            error.args += tuple(values.items())
            values["error"] = error

        # values["classification_label"] = response.output_classification_label()

        return InferenceResponse(**values)


@dataclass
class InferenceRequest:
    model: Model
    _server: _triton_bindings.TRITONSERVER_Server
    request_id: Optional[str] = None
    flags: int = 0
    correlation_id: Optional[int | str] = None
    priority: int = 0
    timeout: int = 0
    inputs: dict[str, _datautils.Tensor | Any] = field(default_factory=dict)
    parameters: dict[str, str | int | bool] = field(default_factory=dict)
    output_memory_type: Optional[_datautils.DeviceOrMemoryType] = None
    output_memory_allocator: Optional[_datautils.MemoryAllocator] = None
    response_queue: Optional[queue.SimpleQueue | asyncio.Queue] = None
    _serialized_inputs: dict[str, _datautils.Tensor] = field(default_factory=dict)

    def _release_request(self, _request, _flags, _user_object):
        pass

    def _add_inputs(self, request):
        for name, value in self.inputs.items():
            if not isinstance(value, _datautils.Tensor):
                tensor = _datautils.Tensor.from_object(value)
            else:
                tensor = value
            if tensor.data_type == _triton_bindings.TRITONSERVER_DataType.BYTES:
                # to ensure lifetime of array
                self._serialized_inputs[name] = tensor
            request.add_input(name, tensor.data_type, tensor.shape)

            request.append_input_data_with_buffer_attributes(
                name,
                tensor.data_ptr,
                tensor.memory_buffer._create_TRITONSERVER_BufferAttributes(),
            )

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

    def _create_TRITONSERVER_InferenceRequest(self):
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

        request.set_release_callback(self._release_request, None)

        return request


class Metric(_triton_bindings.TRITONSERVER_Metric):
    def __init__(self, family: MetricFamily, labels: Optional[dict[str, str]] = None):
        if labels is not None:
            parameters = [
                _triton_bindings.TRITONSERVER_Parameter(name, value)
                for name, value in labels.items()
            ]
        else:
            parameters = []

        _triton_bindings.TRITONSERVER_Metric.__init__(self, family, parameters)
