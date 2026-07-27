# Copyright 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.31.8)

project(tritoncore LANGUAGES C CXX)

# Control building of shared library vs. only headers and stub. By
# default only the headers and library stub is built. Set
# TRITON_CORE_HEADERS_ONLY=OFF to also build libtritonserver.so.
option(TRITON_CORE_HEADERS_ONLY "Build only headers and stub" ON)

#
# Specifying min required C++ standard
#
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

#
# Triton Server API
#
add_library(
  triton-core-serverapi INTERFACE
)

add_library(
  TritonCore::triton-core-serverapi ALIAS triton-core-serverapi
)

target_include_directories(
  triton-core-serverapi
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

#
# Triton Backend API
#
add_library(
  triton-core-backendapi INTERFACE
)

add_library(
  TritonCore::triton-core-backendapi ALIAS triton-core-backendapi
)

target_include_directories(
  triton-core-backendapi
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

#
# Triton RepoAgent API
#
add_library(
  triton-core-repoagentapi INTERFACE
)

add_library(
  TritonCore::triton-core-repoagentapi ALIAS triton-core-repoagentapi
)

target_include_directories(
  triton-core-repoagentapi
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

#
# Triton Cache API
#
add_library(
  triton-core-cacheapi INTERFACE
)

add_library(
  TritonCore::triton-core-cacheapi ALIAS triton-core-cacheapi
)

target_include_directories(
  triton-core-cacheapi
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)



#
# Stub library for libtritonserver.so that stubs Triton Server API and
# Triton Backend API
#
add_library(
  triton-core-serverstub SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tritonserver_stub.cc
)

add_library(
  TritonCore::triton-core-serverstub ALIAS triton-core-serverstub
)
target_compile_features(triton-core-serverstub PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})

target_compile_options(
  triton-core-serverstub
  PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Werror
)

set_target_properties(
  triton-core-serverstub
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME tritonserver
)

#
# Shared library implementing Triton Server API
#
if(NOT TRITON_CORE_HEADERS_ONLY)
  include(CMakeDependentOption)

  set(TRITON_VERSION "0.0.0" CACHE STRING "The version of the Triton shared library" )

  option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
  option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
  option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
  option(TRITON_ENABLE_NVTX "Include NVTX support in server" OFF)
  option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
  option(TRITON_ENABLE_MALI_GPU "Enable Arm Mali GPU support in server" OFF)
  set(TRITON_MIN_COMPUTE_CAPABILITY "7.5" CACHE STRING
      "The minimum CUDA compute capability supported by Triton" )
  set(TRITON_EXTRA_LIB_PATHS "" CACHE PATH "Extra library paths for Triton Server build")

  # Ensemble
  option(TRITON_ENABLE_ENSEMBLE "Include ensemble support in server" OFF)

  # Metrics
  option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
  option(TRITON_ENABLE_METRICS_GPU "Include GPU metrics support in server" ON)
  option(TRITON_ENABLE_METRICS_CPU "Include CPU metrics support in server" ON)

  # Cloud storage
  option(TRITON_ENABLE_GCS "Include GCS Filesystem support in server" OFF)
  option(TRITON_ENABLE_S3 "Include S3 Filesystem support in server" OFF)
  option(TRITON_ENABLE_AZURE_STORAGE "Include Azure Storage Filesystem support in server" OFF)

  # Repo tags
  set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
  set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
  set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/third_party repo")

  # Third-party location
  set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party" CACHE STRING "Location of third-party build")
  set(TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party-src" CACHE STRING "Location of third-party source")

  if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
    message(FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
  endif()

  if(TRITON_ENABLE_TRACING AND NOT TRITON_ENABLE_STATS)
    message(FATAL_ERROR "TRITON_ENABLE_TRACING=ON requires TRITON_ENABLE_STATS=ON")
  endif()

  if (TRITON_ENABLE_METRICS_CPU AND NOT TRITON_ENABLE_METRICS)
    message(FATAL_ERROR "TRITON_ENABLE_METRICS_CPU=ON requires TRITON_ENABLE_METRICS=ON")
  endif()

  if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_METRICS)
    message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_METRICS=ON")
  endif()

  if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_GPU)
    message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_GPU=ON")
  endif()

  # core/src is built in this scope rather than as an ExternalProject.
  #
  # It used to be an ExternalProject fed by third_party: a separate CMake cache
  # meant every dependency path had to be forwarded by hand
  # (-DProtobuf_DIR, -DgRPC_DIR, -Dabsl_DIR, ...), each pointing into
  # TRITON_THIRD_PARTY_INSTALL_PREFIX. As a subdirectory it inherits
  # CMAKE_PREFIX_PATH and the <Pkg>_DIR cache entries the caller already
  # published, so those forwards are unnecessary rather than merely updated --
  # and third_party is no longer fetched at all.
  #
  # Building standalone still works: whoever configures this tree is responsible
  # for making the packages in src/CMakeLists.txt findable, exactly as for any
  # other CMake project.
  add_subdirectory(src)
endif() # NOT TRITON_CORE_HEADERS_ONLY

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonCore)

install(
  TARGETS
    triton-core-backendapi
    triton-core-repoagentapi
    triton-core-cacheapi
    triton-core-serverapi
  EXPORT
    triton-core-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  TARGETS
    triton-core-serverstub
  EXPORT
    triton-core-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/stubs
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/stubs
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/stubs
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT
    triton-core-targets
  FILE
    TritonCoreTargets.cmake
  NAMESPACE
    TritonCore::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonCoreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonCoreConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TritonCoreConfig.cmake
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT
    triton-core-targets
  FILE
    ${CMAKE_CURRENT_BINARY_DIR}/TritonCoreTargets.cmake
  NAMESPACE
    TritonCore::
)

export(PACKAGE TritonCore)

if(NOT TRITON_CORE_HEADERS_ONLY)
  add_subdirectory(python python)
endif()
