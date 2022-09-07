# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# Unit tests
#
find_package(GTest REQUIRED)

#
# CudaMemoryManger
#
set(
  CUDA_MEMORY_MANAGER_SRCS
  ../cuda_memory_manager.cc
  ../cuda_utils.cc
  ../status.cc
)

set(
  CUDA_MEMORY_MANAGER_HDRS
  ../cuda_memory_manager.h
  ../cuda_utils.h
  ../status.h
)

#
# PinnedMemoryManger
#
set(
  PINNED_MEMORY_MANAGER_SRCS
  ../cuda_utils.cc
  ../numa_utils.cc
  ../pinned_memory_manager.cc
  ../status.cc
)

set(
  PINNED_MEMORY_MANAGER_HDRS
  ../cuda_utils.h
  ../numa_utils.h
  ../pinned_memory_manager.h
  ../status.h
)

#
# Memory
#
set(
  MEMORY_SRCS
  ../buffer_attributes.cc
  ../memory.cc
)

set(
  MEMORY_HDRS
  ../buffer_attributes.h
  ../memory.h
)

#
# Unit test for RequestResponseCache
#
if(${TRITON_ENABLE_GPU})
  add_executable(
    response_cache_test
    response_cache_test.cc
    ../response_cache.cc
    ../response_cache.h
    ../status.cc
    ../status.h
    ../constants.h
    ${MEMORY_SRCS}
    ${CUDA_MEMORY_MANAGER_SRCS}
    ${PINNED_MEMORY_MANAGER_SRCS}
    ${MEMORY_HDRS}
    ${CUDA_MEMORY_MANAGER_HDRS}
    ${PINNED_MEMORY_MANAGER_HDRS}
  )

  set_target_properties(
    response_cache_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    response_cache_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${GTEST_INCLUDE_DIRS}
      ${CNMEM_PATH}/include
  )

  target_compile_definitions(
    response_cache_test
    PRIVATE
      TRITON_ENABLE_LOGGING=1
      TRITON_ENABLE_GPU=1
      TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)

  target_link_libraries(
    response_cache_test
    PRIVATE
      triton-common-error        # from repo-common
      triton-common-logging      # from repo-common
      proto-library              # from repo-common
      GTest::gtest
      GTest::gtest_main
      protobuf::libprotobuf
      ${CNMEM_LIBRARY}
      CUDA::cudart
  )

  if (NOT WIN32)
    target_link_libraries(
      response_cache_test
      PRIVATE
        dl
        numa
    )
  endif()

  install(
    TARGETS response_cache_test
    RUNTIME DESTINATION bin
  )
endif() # TRITON_ENABLE_GPU

#
# Unit test for Query
#
add_executable(
  query_test
  query_test.cc
  )

set_target_properties(
  query_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  query_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  query_test
  PRIVATE
    triton-common-error # from repo-common
    triton-core
    GTest::gtest
    GTest::gtest_main
)

install(
  TARGETS query_test
  RUNTIME DESTINATION bin
)

#
# Unit test for Memory
#
if(${TRITON_ENABLE_GPU})
  add_executable(
    memory_test
    memory_test.cc
    ../constants.h
    ${MEMORY_SRCS}
    ${CUDA_MEMORY_MANAGER_SRCS}
    ${PINNED_MEMORY_MANAGER_SRCS}
    ${MEMORY_HDRS}
    ${CUDA_MEMORY_MANAGER_HDRS}
    ${PINNED_MEMORY_MANAGER_HDRS}
  )

  set_target_properties(
    memory_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    memory_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${GTEST_INCLUDE_DIRS}
      ${CNMEM_PATH}/include
  )

  target_compile_definitions(
    memory_test
    PRIVATE
      TRITON_ENABLE_LOGGING=1
      TRITON_ENABLE_GPU=1
      TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)

  target_link_libraries(
    memory_test
    PRIVATE
      triton-common-error        # from repo-common
      triton-common-logging      # from repo-common
      proto-library              # from repo-common
      GTest::gtest
      GTest::gtest_main
      protobuf::libprotobuf
      ${CNMEM_LIBRARY}
      CUDA::cudart
  )

  if (NOT WIN32)
    target_link_libraries(
      memory_test
      PRIVATE
        dl
        numa
    )
  endif()

  install(
    TARGETS memory_test
    RUNTIME DESTINATION bin
  )
endif() # TRITON_ENABLE_GPU

#
# Unit test for PinnedMemoryManager
#
if(${TRITON_ENABLE_GPU})
  add_executable(
    pinned_memory_manager_test
    pinned_memory_manager_test.cc
    ${PINNED_MEMORY_MANAGER_SRCS}
    ${PINNED_MEMORY_MANAGER_HDRS}
  )

  set_target_properties(
    pinned_memory_manager_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    pinned_memory_manager_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${GTEST_INCLUDE_DIRS}
      ${CNMEM_PATH}/include
  )

  target_compile_definitions(
    pinned_memory_manager_test
    PRIVATE
      TRITON_ENABLE_LOGGING=1
      TRITON_ENABLE_GPU=1
      TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)

  target_link_libraries(
    pinned_memory_manager_test
    PRIVATE
      triton-common-error        # from repo-common
      triton-common-logging      # from repo-common
      proto-library              # from repo-common
      GTest::gtest
      GTest::gtest_main
      protobuf::libprotobuf
      ${CNMEM_LIBRARY}
      CUDA::cudart
  )

  if (NOT WIN32)
    target_link_libraries(
      pinned_memory_manager_test
      PRIVATE
        dl
        numa
    )
  endif()

  install(
    TARGETS pinned_memory_manager_test
    RUNTIME DESTINATION bin
  )
endif() # TRITON_ENABLE_GPU

#
# Unit test for AsycWorkQueue
#
add_executable(
  async_work_queue_test
  async_work_queue_test.cc
)

set_target_properties(
  async_work_queue_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  async_work_queue_test
  PRIVATE
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  async_work_queue_test
  PRIVATE
    triton-common-thread-pool        # from repo-common
    triton-common-async-work-queue   # from repo-common
    triton-common-error              # from repo-common
    GTest::gtest
    GTest::gtest_main
)

install(
  TARGETS async_work_queue_test
  RUNTIME DESTINATION bin
)

#
# Unit test for TritonRepoAgent ... (TODO specify the other classes)
#
add_executable(
  repo_agent_test
  repo_agent_test.cc
  ../repo_agent.cc
  ../status.cc
  ../filesystem.cc
  ../model_config_utils.cc
  ../repo_agent.h
  ../shared_library.h
  ../status.h
  ../filesystem.h
  ../model_config_utils.h
)

set_target_properties(
  repo_agent_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  repo_agent_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  repo_agent_test
  PRIVATE
    triton-common-error        # from repo-common
    triton-common-model-config # from repo-common
    triton-common-json         # from repo-common
    triton-common-logging      # from repo-common
    proto-library              # from repo-common
    GTest::gtest
    GTest::gtest_main
    protobuf::libprotobuf
)

install(
  TARGETS repo_agent_test
  RUNTIME DESTINATION bin
)


if(${TRITON_ENABLE_METRICS})
  #
  # Unit test for Generic Metrics API
  #
  add_executable(
    metrics_api_test
    metrics_api_test.cc
    ../metric_family.cc
    ../metric_family.h
    ../metrics.cc
    ../metrics.h
    ../infer_parameter.cc
    ../infer_parameter.h
  )

  set_target_properties(
    metrics_api_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    metrics_api_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${GTEST_INCLUDE_DIRS}
  )

  target_compile_definitions(
    metrics_api_test 
    PRIVATE
      TRITON_ENABLE_LOGGING=1
      TRITON_ENABLE_METRICS=1
  )

  target_link_libraries(
    metrics_api_test
    PRIVATE
      triton-common-error   # from repo-common
      triton-common-logging # from repo-common
      proto-library         # from repo-common
      triton-core
      GTest::gtest
      GTest::gtest_main
      prometheus-cpp::core
      protobuf::libprotobuf
  )

  install(
    TARGETS metrics_api_test
    RUNTIME DESTINATION bin
  )
endif() # TRITON_ENABLE_METRICS

#
# Unit test for Model Repository Register API
#
add_executable(
  register_api_test
  register_api_test.cc
)

set_target_properties(
  register_api_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  register_api_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  register_api_test
  PRIVATE
    triton-common-error   # from repo-common
    triton-common-logging # from repo-common
    triton-core
    GTest::gtest
    GTest::gtest_main
)

install(
  TARGETS register_api_test
  RUNTIME DESTINATION bin
)
