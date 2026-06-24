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


import pytest
import tritonserver
from tritonserver import _c as triton_bindings


@pytest.fixture(autouse=True)
def _clear_log_callback():
    # Logging is process-global in Triton, clear the callback after each test so
    # one test's callback cannot leak into the next.
    yield
    triton_bindings.TRITONSERVER_ServerOptions().set_log_callback(None)


@pytest.fixture
def options():
    return triton_bindings.TRITONSERVER_ServerOptions()


def _create_callback():
    """Return (callback, records); the callback appends each record to records."""
    records = []

    def _callback(level, filename, line, timestamp_us, message):
        records.append((level, filename, line, timestamp_us, message))

    return _callback, records


def _emit(level, message, filename="logcb_test.py", line=1):
    triton_bindings.TRITONSERVER_LogMessage(level, filename, line, message)


class TestLogCallback:
    """Tests for the structured log callback added via
    TRITONSERVER_ServerOptionsSetLogCallback and exposed as Options.log_callback.
    """

    def test_binding_receives_structured_record(options):
        callback, records = _create_callback()
        options.set_log_callback(callback)

        _emit(
            triton_bindings.TRITONSERVER_LogLevel.ERROR,
            "callback-record",
            filename="model.cc",
            line=42,
        )

        assert records, "log callback was not invoked"
        level, filename, line, _ts, message = records[-1]
        assert level == triton_bindings.TRITONSERVER_LogLevel.ERROR
        assert filename == "model.cc"
        assert line == 42
        assert message == "callback-record"

    def test_binding_clear_stops_delivery(options):
        callback, records = _create_callback()
        options.set_log_callback(callback)
        options.set_log_callback(None)  # clear

        _emit(triton_bindings.TRITONSERVER_LogLevel.ERROR, "should-be-dropped")
        assert not records

    def test_binding_callback_exceptions_do_not_propagate(options):
        # A throwing callback must not crash logging or raise to the caller.
        def _raise(*args):
            raise RuntimeError("error in callback")

        options.set_log_callback(_raise)
        _emit(triton_bindings.TRITONSERVER_LogLevel.ERROR, "trigger-throwing-callback")

    def test_option_applies_callback():
        callback, records = _create_callback()
        options = tritonserver.Options(
            # Not started, so the repository path is only stored, never read.
            model_repository="/tmp/triton-log-callback-test",
            # log_info keeps all levels enabled (the logger default), so this
            # does not disable any level process-wide for other tests.
            log_info=True,
            log_callback=callback,
        )
        # Apply the dataclass options to the global logger without starting a server.
        options._create_tritonserver_server_options()

        _emit(triton_bindings.TRITONSERVER_LogLevel.INFO, "via-options")
        assert any(message == "via-options" for *_, message in records)
