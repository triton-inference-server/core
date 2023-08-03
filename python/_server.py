from enum import Enum, IntEnum
from collections.abc import Iterable, Mapping, Tuple
from typing import Any, Union
from ._infer import *
from ._trace import *


class ModelControlMode(Enum):
    NONE: int = 0
    POLL: int = 1
    EXPLICIT: int = 2


# Use IntEnum for flags that may be multiplexed
class BatchProperty(IntEnum):
    UNKNOWN: int = 1
    FIRST_DIM: int = 2


class TransactionProperty(IntEnum):
    ONE_TO_ONE: int = 1
    DECOUPLED: int = 2


class ModelIndexFlag(IntEnum):
    READY: int = 1


class RateLimiterOptions:

    class Mode(Enum):
        OFF: int = 0
        EXECUTION_COUNT: int = 1

    class Resource:

        def __init__(self, name: str, count: int, device_id: int) -> None:
            self._name = name
            self._count = count
            self._device_id = device_id

    def __init__(self, mode: Mode = None) -> None:
        self._mode = mode
        self._resources: Iterable[RateLimiterOptions.Resource] = None
        pass

    def add_resource(self, name: str, count: int, device_id: int) -> None:
        if self._resources is None:
            self._resources = []
        self._resources.append(
            RateLimiterOptions.Resource(name, count, device_id))


class Options:
    # in-process API doesn't provide proper getter,
    # can't reflect the default / current value.
    # NOTE: TRITONSERVER_ServerOptions creation will be in place
    # during server initialization
    def __init__(self) -> None:
        # List the set of options, 'None' implies that the value is not
        # specified and the default value will be used
        self.server_id: str = None
        self.minimum_compute_capability: float = None
        self.strict_readiness: bool = None
        self.exit_on_error: bool = None
        self.exit_timeout: int = None

        # [FIXME] this option is actually not in used in core
        # TRITONSERVER_ServerOptionsSetBufferManagerThreadCount
        self.buffer_manager_thread_count: int = None

        # Model stuff..
        self.model_repositories: Iterable[str] = None
        self.model_control_mode: ModelControlMode = None
        self.startup_models: Iterable[str] = None
        # [FIXME] turn to auto-complete-config?
        self.strict_model_config: bool = None
        self.model_load_thread_count: int = None
        self.model_namespacing: bool = None
        # [WIP] proper type specification
        self.model_load_device_limit: Any = None

        # Rate limiter
        self.rate_limiter: RateLimiterOptions = None

        # memory pool
        self.pinned_memory_pool_size: int = None
        self.cuda_memory_pool_size: dict[str, int] = None

        # cache
        self.cache_directory: str = None
        self.cache_config: dict[str, str] = None

        # Option for Server-provided part of the Metrics
        self.enable_metrics: bool = None
        self.enable_gpu_metrics: bool = None
        self.enable_cpu_metrics: bool = None
        self.metrics_interval: int = None
        self.metrics_config: dict[str, list[tuple[str, str]]] = None

        # Backends
        self.backend_directory: str = None
        self.backend_config: dict[str, list[tuple[str, str]]] = None

        self.host_policy: dict[str, list[tuple[str, str]]] = None

        self.repoagent_directory: str = None

        pass


class TritonCore:

    def __init__(self, option: Options = None) -> None:
        self._stop = True
        # Create TRITONSERVER_ServerOptions
        # TRITONSERVER_ServerNew
        # finally: delete TRITONSERVER_ServerOptions
        self._stop = False
        pass

    def __del__(self) -> None:
        self.stop()

    def stop(self) -> None:
        if not self._stop:
            self._stop = True
            # TRITONSERVER_ServerStop

    # [FIXME] context manager...

    def ready(self) -> bool:
        # TRITONSERVER_ServerIsReady
        raise "Not Implemented"

    def live(self) -> bool:
        # TRITONSERVER_ServerIsLive
        raise "Not Implemented"

    def metadata(self) -> dict:
        # TRITONSERVER_ServerMetadata
        # TRITONSERVER_MessageSerializeToJson
        # json -> Python dict
        # (general procedure to present Triton message to the user)
        raise "Not Implemented"

    # Repository APIs
    def register_repository(self,
                            path: str,
                            name_mapping: Mapping[str, str] = None):
        # TRITONSERVER_ServerRegisterModelRepository
        raise "Not Implemented"

    def unregister_repository(self, path: str):
        # TRITONSERVER_ServerUnregisterModelRepository
        raise "Not Implemented"

    def poll_repository(self):
        # TRITONSERVER_ServerPollModelRepository
        raise "Not Implemented"

    def model_index(self, flags: Union[ModelIndexFlag, int] = None):
        # TRITONSERVER_ServerModelIndex
        raise "Not Implemented"

    def load_model(self, name: str, parameters: Mapping[str, str] = None):
        # TRITONSERVER_ServerLoadModelWithParameters
        raise "Not Implemented"

    def unload_model(self, name: str, recursive_unload=False):
        if recursive_unload:
            # TRITONSERVER_ServerUnloadModelAndDependents
            raise "Not Implemented"
        else:
            # TRITONSERVER_ServerUnloadModel
            raise "Not Implemented"

    # Models [FIXME] another abstraction?
    def model_ready(self, name: str, version: int = -1) -> bool:
        # TRITONSERVER_ServerModelIsReady
        raise "Not Implemented"

    def model_batch_properties(
            self,
            name: str,
            version: int = -1) -> Tuple[Iterable[BatchProperty], Any]:
        # TRITONSERVER_ServerModelBatchProperties
        # demultiplex the properties
        raise "Not Implemented"

    def model_transaction_properties(
            self,
            name: str,
            version: int = -1) -> Tuple[Iterable[TransactionProperty], Any]:
        # TRITONSERVER_ServerModelTransactionProperties
        # demultiplex the properties
        raise "Not Implemented"

    def model_metadata(self, name: str, version: int = -1) -> dict:
        # TRITONSERVER_ServerModelMetadata
        # TRITONSERVER_MessageSerializeToJson
        # json -> Python dict
        # (general procedure to present Triton message to the user)
        raise "Not Implemented"

    def model_statistics(self, name: str, version: int = -1) -> dict:
        # TRITONSERVER_ServerModelStatistics
        # TRITONSERVER_MessageSerializeToJson
        # json -> Python dict
        # (general procedure to present Triton message to the user)
        raise "Not Implemented"

    def model_config(self,
                     name: str,
                     version: int = -1,
                     config_version: int = 1) -> dict:
        # TRITONSERVER_ServerModelConfig
        # TRITONSERVER_MessageSerializeToJson
        # json -> Python dict
        # (general procedure to present Triton message to the user)
        raise "Not Implemented"

    def infer_async(self,
                    request: InferenceRequest,
                    allocator: ResponseAllocator = None,
                    trace_reportor: TraceReportor = None) -> ResponseHandle:
        # TRITONSERVER_InferenceRequestNew and initialization
        # If 'allocator' not provided, use DefaultAllocator
        # set with pre-define callbacks..
        raise "Not Implemented"
