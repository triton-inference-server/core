#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


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


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPyCommand})
