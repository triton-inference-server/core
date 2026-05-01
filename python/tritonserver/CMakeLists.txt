# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.31.8)

project(triton-bindings LANGUAGES C CXX)

# Top level module entry point and typed marker
file(COPY __init__.py DESTINATION .)
file(COPY py.typed DESTINATION .)
# Copy the '__init__.py' for the '_c' module
file(COPY _c/__init__.py DESTINATION ./_c/.)
# Find and copy _api modules
file(GLOB PYTHON_MODULE_FILES ./_api/*.py)
file(COPY ${PYTHON_MODULE_FILES} DESTINATION ./_api/.)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11"
  # COMMIT ID for v2.12.0
  GIT_TAG "3e9dfa2866941655c56877882565e7577de6fc7b"
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pybind11)
set(
  PYTHON_BINDING_SRCS
  _c/tritonserver_pybind.cc
)

pybind11_add_module(python-bindings SHARED ${PYTHON_BINDING_SRCS})
target_link_libraries(
  python-bindings
  PRIVATE
  triton-core-serverapi           # from repo-core
  triton-core-serverstub          # from repo-core
)
target_compile_features(python-bindings PRIVATE cxx_std_17)

set_property(TARGET python-bindings PROPERTY OUTPUT_NAME triton_bindings)
# Add Triton library default path in 'rpath' for runtime library lookup
set_target_properties(python-bindings PROPERTIES BUILD_RPATH "$ORIGIN:/opt/tritonserver/lib")
set_target_properties(python-bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/tritonserver/_c/)
