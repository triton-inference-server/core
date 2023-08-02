from enum import Enum
from collections.abc import Iterable


class LogLevel(Enum):
    INFO: int = 0
    WARNING: int = 1  # Warn in Triton in-process API
    ERROR: int = 2
    VERBOSE: int = 3


class LogFormat(Enum):
    DEFAULT: int = 0
    ISO8601: int = 1


# [WIP] figure out "singleton"
class GlobalLogger:

    def __init__(self,
                 format: LogFormat = LogFormat.DEFAULT,
                 file: str = "") -> None:
        # Below will raise exception if 'format' is not the proper class
        format in LogFormat
        self.format(format)
        self.file(file)

    # Log APIs
    def info(self, message: str) -> None:
        self.log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        self.log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        self.log(LogLevel.ERROR, message)

    def verbose(self, message: str) -> None:
        self.log(LogLevel.VERBOSE, message)

    # use by handler to "emit" the LogRecord
    def log(self, level, message) -> None:
        # [FIXME] get line and file name
        # TRITONSERVER_LogMessage
        raise "Not Implemented"

    # property getters / setters
    @property
    def format(self) -> LogFormat:
        return self._format

    @format.setter
    def format(self, f: LogFormat) -> None:
        # TRITONSERVER_ServerOptionsSetLogFormat
        self._format = format

    @property
    def file(self) -> str:
        return self._file

    @format.setter
    def file(self, f: str) -> None:
        # TRITONSERVER_ServerOptionsSetLogFile
        self._file = f

    @property
    def enabled_info(self) -> bool:
        # TRITONSERVER_LogIsEnabled(INFO, ...)
        raise "Not Implemented"
        return False

    @enabled_info.setter
    def enable_info(self, v: bool) -> None:
        # TRITONSERVER_ServerOptionsSetLogInfo
        raise "Not Implemented"

    @property
    def enabled_warning(self) -> bool:
        # TRITONSERVER_LogIsEnabled(WARN, ...)
        raise "Not Implemented"
        return False

    @enabled_warning.setter
    def enable_warning(self, v: bool) -> None:
        # TRITONSERVER_ServerOptionsSetLogWarn
        raise "Not Implemented"

    @property
    def enabled_error(self) -> bool:
        # TRITONSERVER_LogIsEnabled(ERROR, ...)
        raise "Not Implemented"
        return False

    @enabled_error.setter
    def enable_error(self, v: bool) -> None:
        # TRITONSERVER_ServerOptionsSetLogError
        raise "Not Implemented"

    @property
    def enabled_verbose(self) -> bool:
        # TRITONSERVER_LogIsEnabled(VERBOSE, ...)
        raise "Not Implemented"
        return False

    @enabled_verbose.setter
    def enable_verbose(self, v: int) -> None:
        # TRITONSERVER_ServerOptionsSetLogVerbose
        raise "Not Implemented"
