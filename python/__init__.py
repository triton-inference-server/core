# Import higher-level Python wrappers
from ._common import *
from ._infer import *
from ._logger import *
from ._metrics import *
from ._server import *
from ._trace import *

# [FIXME] exposing the actual C API binding below for users who want
# more customization over what the Python wrapper provides.
# one may invoke the API using bindings.xxx()
import _pybind as bindings

TARGET_TRITONSERVER_API_VERSION_MAJOR: int = 1
TARGET_TRITONSERVER_API_VERSION_MINOR: int = 23

# static check of the version, use to make sure the Python binding
# aligns with the shipped in-process API
major, minor = version()
if (major != TARGET_TRITONSERVER_API_VERSION_MAJOR) or (
        minor < TARGET_TRITONSERVER_API_VERSION_MINOR):
    raise "The Python binding is generated for incompatible version of the API, target '{}.{}', got '{}.{}'".format(
        TARGET_TRITONSERVER_API_VERSION_MAJOR,
        TARGET_TRITONSERVER_API_VERSION_MINOR, major, minor)

# Sanity check for keeping the Python binding up-to-date, should look for
# this message as part of the pipeline
if (minor > TARGET_TRITONSERVER_API_VERSION_MINOR):
    print(
        "WARNING: The Python binding is generated for an older minor version of the API, target '{}.{}', got '{}.{}'"
        .format(TARGET_TRITONSERVER_API_VERSION_MAJOR,
                TARGET_TRITONSERVER_API_VERSION_MINOR, major, minor))
