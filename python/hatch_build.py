# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Tag the tritonserver wheel as platform-specific and emit type stubs.

    The wheel ships an arch-specific CPython extension
    (tritonserver/_c/triton_bindings.cpython-<xy>-<arch>-linux-gnu.so) that
    build_wheel.py stages into the package rather than having the backend
    compile it. Hatchling therefore sees only data files and would emit a
    pure-Python "py3-none-any" wheel, which auditwheel rejects.

    pure_python=False plus infer_tag=True makes hatchling derive the
    cp<XY>-cp<XY>-linux_<arch> tag from the running interpreter, reproducing
    what the setuptools BinaryDistribution shim did. See TRI-983.

    stubgen regenerates the _c type stubs next to the extension so they are
    collected into the wheel, replacing the setuptools build_py subclass.
    """

    def initialize(self, version, build_data):
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        # Written into the package tree (self.root is the wheel build root)
        # so hatchling collects the stubs alongside the extension.
        subprocess.run(
            ["stubgen", "-p", "tritonserver._c", "-o", self.root],
            check=True,
        )
