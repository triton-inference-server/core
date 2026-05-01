# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Logging Utilities"""

import inspect
import os

from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel
from tritonserver._c.triton_bindings import TRITONSERVER_LogMessage


def LogMessage(level: LogLevel, message: str):
    """Log Message using Triton Inference Server Logger

    Parameters
    ----------
    level : LogLevel
        log level one of LogLevel.WARN, LogLevel.ERROR, LogLevel.INFO
    message : str
        message

    Examples
    --------

    LogMessage(LogLevel.ERROR,"I've got a bad feeling about this ...")

    """

    filename, line_number = "unknown", -1
    try:
        current_frame = inspect.stack()[-1]
        filename, line_number = (
            os.path.basename(current_frame.filename),
            current_frame.lineno,
        )
    except Exception as e:
        pass
    TRITONSERVER_LogMessage(level, filename, line_number, message)
