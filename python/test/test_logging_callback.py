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


import os

import pytest
import tritonserver
from tritonserver import _c as triton_bindings

LogLevel = triton_bindings.TRITONSERVER_LogLevel


def _make_repo(path):
    # An existing (empty) model repository is enough for an EXPLICIT-mode server:
    # no models are loaded, so no backend is needed.
    os.makedirs(path, exist_ok=True)
    return path


def _make_server(repo_dir, callback):
    """Create a minimal in-process server. Construction runs
    TRITONSERVER_ServerNew, which installs `callback` (if any) on the global
    logger before any worker/logging thread starts."""
    options = triton_bindings.TRITONSERVER_ServerOptions()
    options.set_model_repository_path(repo_dir)
    options.set_model_control_mode(
        triton_bindings.TRITONSERVER_ModelControlMode.EXPLICIT
    )
    options.set_strict_model_config(False)
    options.set_exit_timeout(5)
    if callback is not None:
        options.set_log_callback(callback)
    return triton_bindings.TRITONSERVER_Server(options)


def _create_callback():
    """Return (callback, records); the callback appends each record to records."""
    records = []

    def _callback(level, filename, line, timestamp_us, message):
        records.append((level, filename, line, timestamp_us, message))

    return _callback, records


def _emit(level, message, filename="logcb_test.py", line=1):
    triton_bindings.TRITONSERVER_LogMessage(level, filename, line, message)


def _messages(records):
    return [message for *_, message in records]


@pytest.fixture
def repo_dir(tmp_path):
    return _make_repo(str(tmp_path / "models"))


class TestLogCallback:
    def test_server_install_forwards_structured_record(self, repo_dir):
        callback, records = _create_callback()
        _make_server(repo_dir, callback)

        _emit(LogLevel.ERROR, "callback-record", filename="model.cc", line=42)

        matches = [r for r in records if r[4] == "callback-record"]
        assert matches, "log callback was not invoked"
        level, filename, line, _ts, message = matches[-1]
        assert level == LogLevel.ERROR
        assert filename == "model.cc"
        assert line == 42
        assert message == "callback-record"

    def test_server_without_callback_uses_default_sink(self, tmp_path):
        callback, records = _create_callback()
        # First server installs the recording callback.
        _make_server(_make_repo(str(tmp_path / "a")), callback)
        _emit(LogLevel.ERROR, "before-clear")
        assert "before-clear" in _messages(records)

        # A later server with no callback overwrites it with the default sink.
        _make_server(_make_repo(str(tmp_path / "b")), None)
        records.clear()
        _emit(LogLevel.ERROR, "after-clear")
        assert "after-clear" not in _messages(records)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_throwing_callback_does_not_propagate(self, repo_dir):
        # A throwing callback must not crash logging or raise to the caller,
        # including for the server's own startup logs.
        def _raise(*args):
            raise RuntimeError("error in callback")

        _make_server(repo_dir, _raise)
        _emit(LogLevel.ERROR, "trigger-throwing-callback")

    def test_high_level_option_installs_callback(self, repo_dir):
        callback, records = _create_callback()
        server = tritonserver.Server(
            model_repository=repo_dir,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=True,  # enable INFO so the emitted record is not filtered
            log_callback=callback,
        ).start()
        try:
            _emit(LogLevel.INFO, "via-options")
            assert "via-options" in _messages(records)
        finally:
            server.stop()
