#!/usr/bin/env python3
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


import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    def run(self):
        build_py.run(self)
        # Generate stub files:
        package_name = self.distribution.metadata.name
        subprocess.run(
            ["stubgen", "-p", f"{package_name}._c", "-o", f"{self.build_lib}"],
            check=True,
        )


# The wheel ships an arch-specific CPython extension
# (tritonserver/_c/triton_bindings.*.so). Mark root as impure so
# setuptools/wheel tags the produced wheel with the current platform
# (e.g. linux_x86_64 / linux_aarch64) instead of the misleading
# "none-any" that violates PEP 425 for wheels with arch-specific content.
#
# NOTE: the embedded .so is also CPython-ABI-specific (filename encodes
# "cpython-312-..." etc.), which means it is only loadable under the
# matching interpreter. The current override keeps the existing
# "py3-none-<plat>" shape for backwards compatibility with consumers;
# promote the `get_tag` override to emit "cp<XY>-cp<XY>" when we are
# ready to gate installs on the exact CPython version (see TRI-983).
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    bdist_wheel = None


if __name__ == "__main__":
    cmdclass = {"build_py": BuildPyCommand}
    if bdist_wheel is not None:
        cmdclass["bdist_wheel"] = bdist_wheel
    setup(cmdclass=cmdclass)
