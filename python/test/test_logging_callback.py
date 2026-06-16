# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the structured log callback added via
TRITONSERVER_ServerOptionsSetLogCallback and exposed as Options.log_callback."""

import pytest
import tritonserver
from tritonserver import _c as triton_bindings

# Logging configuration is process-global in Triton. The autouse fixture below
# clears any registered callback after every test so a test's callback cannot
# leak into later tests.


@pytest.fixture(autouse=True)
def _clear_log_callback():
    yield
    triton_bindings.TRITONSERVER_ServerOptions().set_log_callback(None)


def _emit(level, message, filename="logcb_test.py", line=1):
    triton_bindings.TRITONSERVER_LogMessage(level, filename, line, message)


class TestLogCallbackBinding:
    """Low-level binding: TRITONSERVER_ServerOptions.set_log_callback."""

    def test_receives_structured_record(self):
        records = []
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_log_callback(
            lambda level, filename, line, ts, msg: records.append(
                (level, filename, line, ts, msg)
            )
        )

        _emit(
            triton_bindings.TRITONSERVER_LogLevel.ERROR,
            "callback-record",
            filename="model.cc",
            line=42,
        )

        assert records, "log callback was not invoked"
        level, filename, line, _ts, msg = records[-1]
        assert level == triton_bindings.TRITONSERVER_LogLevel.ERROR
        assert filename == "model.cc"
        assert line == 42
        assert msg == "callback-record"

    def test_clear_stops_delivery(self):
        records = []
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_log_callback(lambda *args: records.append(args))
        options.set_log_callback(None)  # clear

        _emit(triton_bindings.TRITONSERVER_LogLevel.ERROR, "should-be-dropped")
        assert not records

    def test_callback_exceptions_do_not_propagate(self):
        # A throwing callback must not crash logging.
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_log_callback(
            lambda *args: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        # Should not raise.
        _emit(triton_bindings.TRITONSERVER_LogLevel.ERROR, "trigger-throwing-callback")


class TestLogCallbackOption:
    """High-level surface: tritonserver.Options(log_callback=...)."""

    def test_option_applies_callback(self):
        records = []
        options = tritonserver.Options(
            # Not started, so the repository path is only stored, never read.
            model_repository="/tmp/triton-log-callback-test",
            # log_info cascades to enable WARN/ERROR too, so this leaves all
            # levels enabled (the logger's default state) and does not disable
            # any level process-wide for other tests in this process.
            log_info=True,
            log_callback=lambda level, filename, line, ts, msg: records.append(msg),
        )
        # Translate the dataclass to C-API options; this applies the callback to
        # the (global) logger without starting a server.
        options._create_tritonserver_server_options()

        _emit(triton_bindings.TRITONSERVER_LogLevel.INFO, "via-options")
        assert "via-options" in records
