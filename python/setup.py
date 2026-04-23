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

from setuptools import Distribution, setup
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
# (tritonserver/_c/triton_bindings.cpython-<xy>-<arch>-linux-gnu.so)
# that is copied into the package_data at build time rather than
# declared via setup(ext_modules=...). Without a declared ext_module
# setuptools treats the distribution as pure-Python and emits
# "Root-Is-Purelib: true" in the WHEEL metadata + a "py3-none-any"
# tag, which auditwheel rightly rejects.
#
# Signaling has_ext_modules()=True via a custom Distribution subclass
# is the canonical way to tell setuptools the wheel is binary without
# triggering a fake compilation step. setuptools then:
#   - sets Root-Is-Purelib to false (required for auditwheel repair),
#   - auto-derives the correct cp<XY>-cp<XY>-linux_<arch> tag from
#     the current interpreter and sysconfig.get_platform().
# See TRI-983.
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPyCommand})
