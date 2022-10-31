<!--
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Inference Server Core

This repository holds the source code and headers for the library that
implements the core functionality of Triton. The *core* library can be
built as described below and used directly via its [C
API](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#in-process-triton-server-api). To
be useful the core library must be paired with one or more backends.
You can learn more about backends in the [backend
repo](https://github.com/triton-inference-server/backend).

Typically you do not build or use the core library on its own, but as
part of the *tritonserver* executable. The *tritonserver* executable
is built in the [server
repo](https://github.com/triton-inference-server/server) as described
in the [server build
documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md).

Ask questions or report problems in the main Triton [issues
page](https://github.com/triton-inference-server/server/issues).

## Build the Triton Core Library

Before building the Triton core library, your build system must
install the required dependencies described in the [build
documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md). For
example, if you are building the core library with GPU support
(-DTRITON_ENABLE_GPU=ON), then you must install the CUDA, cuDNN, and
TensorRT dependencies required for the version of Triton you are
building.

To build, first clone the release branch matching the Triton release
you are interest in (*rxx.yy*), or the *main* branch to build the
top-of-tree. The Triton core library is built with CMake.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_CORE_HEADERS_ONLY=OFF ..
$ make install
```

When the build completes, the install directory will contain the
Triton core shared library (install/lib/libtritonserver.so on Linux,
install/bin/tritonserver.dll on Windows), and the core library headers
files in install/include/triton/core.

### Build a Release Branch

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/third_party: -DTRITON_THIRD_PARTY_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

You will need to override if you are building from a release
branch. For example, if you are building the r22.03 version of Triton,
you would clone the r22.03 branch of the core repo and use the
following cmake command.

```
$ cmake -DTRITON_THIRD_PARTY_REPO_TAG=r22.03 -DTRITON_COMMON_REPO_TAG=r22.03 -DTRITON_CORE_HEADERS_ONLY=OFF ..
```

### Build Options

The [CMakeLists.txt](CMakeLists.txt) file contains the options
available when build the core library. For example, to build the core
library with the default settings plus S3 cloud storage and ensembling
support use the following command.

```
$ cmake -DTRITON_CORE_HEADERS_ONLY=OFF -DTRITON_ENABLE_S3=ON -DTRITON_ENABLE_ENSEMBLE=ON ..
```
