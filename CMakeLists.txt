# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
		  "If the target system is not Windows 10, please update _WIN32_WINNT "
		  "to corresponding value.")
  target_compile_options(
    triton-core-serverstub
    PRIVATE
      /Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
  )
else()
  target_compile_options(
    triton-core-serverstub
    PRIVATE
      -Wall -Wextra -Wno-unused-parameter -Werror
  )
endif()

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

  include(FetchContent)
  FetchContent_Declare(
    repo-third-party
    GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/third_party.git
    GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
  )
  FetchContent_MakeAvailable(repo-third-party)

  # Some libs are installed to ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib64 instead
  # of ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib on Centos
  set (LIB_DIR "lib")
  if(LINUX)
    file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")
    if(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
      set (LIB_DIR "lib64")
    endif(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
  endif(LINUX)
  set(TRITON_CORE_HEADERS_ONLY OFF)

  # Need to use ExternalProject for our builds so that we can get the
  # correct dependencies between Triton shared library components and
  # the ExternalProject dependencies (found in the third_party repo)
  include(ExternalProject)

  # If CMAKE_TOOLCHAIN_FILE is set, propagate that hint path to the external
  # projects.
  set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "")
  if (CMAKE_TOOLCHAIN_FILE)
    set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}")
  endif()

  # If VCPKG_TARGET_TRIPLET is set, propagate that hint path to the external
  # projects.
  set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "")
  if (VCPKG_TARGET_TRIPLET)
    set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "-DVCPKG_TARGET_TRIPLET:STRING=${VCPKG_TARGET_TRIPLET}")
  endif()

  # If OPENSSL_ROOT_DIR is set, propagate that hint path to the external
  # projects with OpenSSL dependency.
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
  if (OPENSSL_ROOT_DIR)
    set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
  endif()

  # Location where protobuf-config.cmake will be installed varies by
  # platform
  if (WIN32)
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/cmake")
  else()
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/${LIB_DIR}/cmake/protobuf")
  endif()

  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
  else()
    set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  endif()

  set(TRITON_DEPENDS googletest protobuf re2)
  if(${TRITON_ENABLE_GCS})
    set(TRITON_DEPENDS ${TRITON_DEPENDS} google-cloud-cpp)
  endif() # TRITON_ENABLE_GCS
  if(${TRITON_ENABLE_S3})
    set(TRITON_DEPENDS ${TRITON_DEPENDS} aws-sdk-cpp)
    # Add where to find all S3 dependencies to CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
      ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/lib/cmake
      ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/lib)
  endif() # TRITON_ENABLE_S3
  if(${TRITON_ENABLE_AZURE_STORAGE})
    set(TRITON_DEPENDS ${TRITON_DEPENDS} azure-sdk)
    set(TRITON_AZURE_STORAGE_PACKAGE_DIRS
      -Dazure-storage-blobs-cpp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/azure-sdk/share/azure-storage-blobs-cpp
      -Dazure-storage-common-cpp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/azure-sdk/share/azure-storage-common-cpp
      -Dazure-core-cpp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/azure-sdk/share/azure-core-cpp
    )
  endif() # TRITON_ENABLE_AZURE_STORAGE
  if(${TRITON_ENABLE_METRICS})
    set(TRITON_DEPENDS ${TRITON_DEPENDS} prometheus-cpp)
  endif() # TRITON_ENABLE_METRICS
  if(${TRITON_ENABLE_GPU})
    set(TRITON_DEPENDS ${TRITON_DEPENDS} cnmem)
  endif() # TRITON_ENABLE_GPU

  ExternalProject_Add(triton-core
    PREFIX triton-core
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton-core"
    CMAKE_CACHE_ARGS
      -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
      ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DGTEST_ROOT:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
      -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
      -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
      -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
      -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
      -Dnlohmann_json_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/${LIB_DIR}/cmake/nlohmann_json
      -Dprometheus-cpp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/prometheus-cpp/${LIB_DIR}/cmake/prometheus-cpp
      -Dgoogle_cloud_cpp_storage_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/google_cloud_cpp_storage
      -Dgoogle_cloud_cpp_rest_internal_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/google_cloud_cpp_rest_internal
      -Dgoogle_cloud_cpp_common_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/google_cloud_cpp_common
      ${TRITON_AZURE_STORAGE_PACKAGE_DIRS}
      -DCrc32c_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/crc32c/${LIB_DIR}/cmake/Crc32c
      -DAWSSDK_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/AWSSDK
      -Daws-cpp-sdk-core_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/aws-cpp-sdk-core
      -Daws-cpp-sdk-s3_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/aws-cpp-sdk-s3
      -Daws-c-event-stream_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-event-stream/cmake
      -Daws-c-common_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-common/cmake
      -Daws-checksums_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-checksums/cmake
      -Daws-crt-cpp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-crt-cpp/cmake
      -Daws-c-http_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-http/cmake
      -Daws-c-io_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-io/cmake
      -Ds2n_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/s2n/cmake
      -Daws-c-cal_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-cal/cmake
      -Daws-c-s3_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-s3/cmake
      -Daws-c-auth_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-auth/cmake
      -Daws-c-compression_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-compression/cmake
      -Daws-c-mqtt_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-mqtt/cmake
      -Daws-c-sdkutils_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-sdkutils/cmake
      -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
      -DCNMEM_PATH:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/cnmem
      -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
      -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
      -DTRITON_EXTRA_LIB_PATHS:PATH=${TRITON_EXTRA_LIB_PATHS}
      -DTRITON_ENABLE_NVTX:BOOL=${TRITON_ENABLE_NVTX}
      -DTRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
      -DTRITON_ENABLE_LOGGING:BOOL=${TRITON_ENABLE_LOGGING}
      -DTRITON_ENABLE_STATS:BOOL=${TRITON_ENABLE_STATS}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DTRITON_ENABLE_MALI_GPU:BOOL=${TRITON_ENABLE_MALI_GPU}
      -DTRITON_MIN_COMPUTE_CAPABILITY:STRING=${TRITON_MIN_COMPUTE_CAPABILITY}
      -DTRITON_ENABLE_METRICS:BOOL=${TRITON_ENABLE_METRICS}
      -DTRITON_ENABLE_METRICS_GPU:BOOL=${TRITON_ENABLE_METRICS_GPU}
      -DTRITON_ENABLE_METRICS_CPU:BOOL=${TRITON_ENABLE_METRICS_CPU}
      -DTRITON_ENABLE_GCS:BOOL=${TRITON_ENABLE_GCS}
      -DTRITON_ENABLE_AZURE_STORAGE:BOOL=${TRITON_ENABLE_AZURE_STORAGE}
      -DTRITON_ENABLE_S3:BOOL=${TRITON_ENABLE_S3}
      -DTRITON_ENABLE_ENSEMBLE:BOOL=${TRITON_ENABLE_ENSEMBLE}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
      -DTRITON_VERSION:STRING=${TRITON_VERSION}
    DEPENDS ${TRITON_DEPENDS}
  )
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
