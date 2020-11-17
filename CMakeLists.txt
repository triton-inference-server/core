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

cmake_minimum_required(VERSION 3.17)

project(tritoncore LANGUAGES C CXX)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/tritonserver_stub.cc
)

add_library(
  TritonCore::triton-core-serverstub ALIAS triton-core-serverstub
)

target_compile_features(triton-core-serverstub PRIVATE cxx_std_11)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
		  "If the target system is not Windows 10, please update _WIN32_WINNT "
		  "to corresponding value.")
  set(CMAKE_CXX_FLAGS "/W4 /D_WIN32_WINNT=0x0A00 /EHsc")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

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
