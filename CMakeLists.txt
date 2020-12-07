# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

include(ExternalProject)

#
# Get the protobuf and grpc source
#
ExternalProject_Add(grpc-repo
  PREFIX grpc-repo
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.25.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# Build protobuf project from grpc-repo
#
# Use -DCMAKE_BUILD_TYPE:STRING=RELEASE to workaround the protobuf build issue
# described in https://stackoverflow.com/a/60218066/2091555
ExternalProject_Add(protobuf
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/protobuf/cmake"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -Dprotobuf_BUILD_TESTS:BOOL=OFF
    -Dprotobuf_WITH_ZLIB:BOOL=OFF
    -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -DCMAKE_BUILD_TYPE:STRING=RELEASE
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf
  DEPENDS grpc-repo
)

SET(DEPENDENCIES ${DEPENDENCIES} protobuf)

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/${CMAKE_INSTALL_LIBDIR}/cmake/protobuf")
endif()
set(Protobuf_DIR ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})
message(STATUS "Using Protobuf_DIR:  ${Protobuf_DIR}")

#
# Protobuf
#
set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS model_config.proto)

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

target_compile_features(triton-core-serverstub PRIVATE cxx_std_11)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
		  "If the target system is not Windows 10, please update _WIN32_WINNT "
		  "to corresponding value.")
  target_compile_options(
    triton-core-serverstub
    PRIVATE
      /Wall /D_WIN32_WINNT=0x0A00 /EHsc
  )

  set_target_properties(
    triton-core-serverstub
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      NO_SONAME ON
      OUTPUT_NAME tritonserver
  )
else()
  target_compile_options(
    triton-core-serverstub
    PRIVATE
      -Wall -Wextra -Wno-unused-parameter -Werror
  )
  set_target_properties(
    triton-core-serverstub
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      NO_SONAME ON
      OUTPUT_NAME tritonserver_stub
  )

  target_link_libraries(
    triton-core-serverstub
    PRIVATE
      -Wl,-soname=libtritonserver.so
  )
endif()


#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonCore)

install(
  TARGETS
    triton-core-backendapi
    triton-core-serverapi
  EXPORT
    triton-core-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  TARGETS
    triton-core-serverstub
  EXPORT
    triton-core-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
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
