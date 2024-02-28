from enum import Enum
from collections.abc import Iterable


class Format(Enum):
    PROMETHEUS: int = 0


class Kind(Enum):
    COUNTER: int = 0
    GAUGE: int = 1


class Metric:

    def __init__(self, family, labels) -> None:
        self._closed = False
        # labels -> TRITONSERVER_Parameter
        # TRITONSERVER_MetricNew
        # finally: delete TRITONSERVER_Parameter
        pass

    def close(self):
        if self._closed:
            return

        self._closed = True
        # if self._metric is not None: TRITONSERVER_MetricDelete
        raise "Not Implemented"

    def __del__(self):
        self.close()

    @property
    def kind(self) -> Kind:
        # TRITONSERVER_GetMetricKind
        raise "Not Implemented"
        return Kind.COUNTER

    @property
    def value(self) -> float:
        # TRITONSERVER_MetricValue
        raise "Not Implemented"
        return 0.0

    @value.setter
    def value(self, v: float):
        # TRITONSERVER_MetricSet
        raise "Not Implemented"

    def increment(self, v: float):
        # TRITONSERVER_MetricIncrement
        raise "Not Implemented"


class Family:

    def __init__(self, kind: Kind, name: str, description: str) -> None:
        # TRITONSERVER_MetricFamilyNew
        self._metrics = []
        pass

    def __del__(self):
        # Make sure all associated metrics are deleted
        for metric in self._metrics:
            metric.close()
        # TRITONSERVER_MetricFamilyDelete

    def add_metric(self, labels: Iterable[str]):
        # self._metrics.append(Metric(self, labels))
        pass


# [WIP] figure out "singleton"
class GlobalMetrics:

    def __init__(self, format: Format = Format.Prometheus) -> None:
        # Below will raise exception if 'format' is not the proper class
        format in Format
        self._format = format
        self._metric_families = {}

    def report(self) -> str:
        # TRITONSERVER_ServerMetrics
        # TRITONSERVER_MetricsFormatted
        # convert to string
        # TRITONSERVER_MetricsDelete
        raise "Not Implemented"
        return ""

    # custom metric..
    def add_family(self, kind: Kind, name: str, description: str) -> Family:
        # self._metric_families[name] = Family(kind, name, description)
        # return newed object
        raise "Not Implemented"
