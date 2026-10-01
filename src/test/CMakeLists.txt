# Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# Unit tests
#
find_package(GTest REQUIRED)

#
# CudaMemoryManger
#
set(
  CUDA_MEMORY_MANAGER_SRCS
  ../cuda_block_manager.cc
  ../cuda_memory_manager.cc
  ../cuda_utils.cc
  ../status.cc
)

set(
  CUDA_MEMORY_MANAGER_HDRS
  ../cuda_block_manager.h
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
  ../triton_memory.cc
)

set(
  MEMORY_HDRS
  ../buffer_attributes.h
  ../triton_memory.h
)

#
# Unit test for TritonCache
#
if(${TRITON_ENABLE_GPU})
  add_executable(
    response_cache_test
    response_cache_test.cc
    ../cache_manager.cc
    ../cache_manager.h
    ../cache_entry.cc
    ../cache_entry.h
    ../filesystem/api.cc
    ../filesystem/api.h
    ../shared_library.cc
    ../shared_library.h
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
      ${Boost_INCLUDE_DIRS}
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
      triton-common-model-config # from repo-common
      proto-library              # from repo-common
      triton-core
      GTest::gtest
      GTest::gtest_main
      protobuf::libprotobuf
      ${CNMEM_LIBRARY}
      CUDA::cudart
  )

  target_link_libraries(
    response_cache_test
    PRIVATE
      dl
      numa
  )

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
    GTest::gmock
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

  target_link_libraries(
    memory_test
    PRIVATE
      dl
      numa
  )

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

  target_link_libraries(
    pinned_memory_manager_test
    PRIVATE
      dl
      numa
  )

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
  ../filesystem/api.cc
  ../model_config_utils.cc
  ../repo_agent.h
  ../shared_library.h
  ../status.h
  ../filesystem/api.h
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
    ${Boost_INCLUDE_DIRS}
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
    ${PINNED_MEMORY_MANAGER_SRCS}
    ${PINNED_MEMORY_MANAGER_HDRS}
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
      GTest::gmock
      prometheus-cpp::core
      protobuf::libprotobuf
  )

  if (TRITON_ENABLE_GPU)
    target_link_libraries(
      metrics_api_test
      PRIVATE
        ${CNMEM_LIBRARY}
        CUDA::cudart
    )
  endif()

  target_link_libraries(
    metrics_api_test
    PRIVATE
      dl
      numa
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

#
# Unit test for InstanceQueue
#
add_executable(
  instance_queue_test
  instance_queue_test.cc
  ../instance_queue.cc
)

set_target_properties(
  instance_queue_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  instance_queue_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  instance_queue_test
  PRIVATE
    triton-common-error        # from repo-common
    triton-common-logging      # from repo-common
    triton-common-model-config # from repo-common
    proto-library              # from repo-common
    GTest::gtest
    GTest::gtest_main
    protobuf::libprotobuf
)

install(
  TARGETS instance_queue_test
  RUNTIME DESTINATION bin
)

#
# Backend Output Detail Unittest
#
add_executable(
  backend_output_detail_test
  backend_output_detail_test.cc
)

target_include_directories(
  backend_output_detail_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${GTEST_INCLUDE_DIRS}
)

set_target_properties(
  backend_output_detail_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_link_libraries(
  backend_output_detail_test
  PRIVATE
    triton-common-error   # from repo-common
    triton-common-logging # from repo-common
    triton-core
    GTest::gtest
    GTest::gtest_main
)

install(
  TARGETS backend_output_detail_test
  RUNTIME DESTINATION bin
)

#
# Request Cancellation Unittest
#
add_executable(
  request_cancellation_test
  request_cancellation_test.cc
)

set_target_properties(
  request_cancellation_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  request_cancellation_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  request_cancellation_test
  PRIVATE
    triton-common-error   # from repo-common
    triton-common-logging # from repo-common
    triton-core
    GTest::gtest
    GTest::gtest_main
)

install(
  TARGETS request_cancellation_test
  RUNTIME DESTINATION bin
)

#
# Unit test for Input Byte Size Validation
#
if(${TRITON_ENABLE_GPU})
  add_executable(
    input_byte_size_test
    input_byte_size_test.cc
    ../constants.h
    ${CUDA_MEMORY_MANAGER_SRCS}
    ${CUDA_MEMORY_MANAGER_HDRS}
  )
  target_include_directories(
    input_byte_size_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${GTEST_INCLUDE_DIRS}
      ${CNMEM_PATH}/include
  )

  target_compile_definitions(
    input_byte_size_test
    PRIVATE
      TRITON_ENABLE_LOGGING=1
      TRITON_ENABLE_GPU=1
      TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)

  target_link_libraries(
    input_byte_size_test
    PRIVATE
      dl
      triton-common-error        # from repo-common
      triton-core
      triton-common-logging      # from repo-common
      proto-library              # from repo-common
      GTest::gtest
      GTest::gtest_main
      GTest::gmock
      protobuf::libprotobuf
      CUDA::cudart
      ${CNMEM_LIBRARY}
  )
else()
  add_executable(
    input_byte_size_test
    input_byte_size_test.cc
  )
  target_include_directories(
    input_byte_size_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${GTEST_INCLUDE_DIRS}
  )
  target_link_libraries(
    input_byte_size_test
    PRIVATE
      triton-common-error # from repo-common
      triton-core
      GTest::gtest
      GTest::gtest_main
      GTest::gmock
  )
endif() # TRITON_ENABLE_GPU

set_target_properties(
  input_byte_size_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

install(
  TARGETS input_byte_size_test
  RUNTIME DESTINATION bin
)

# Internal C++ symbols are hidden by triton-core's export script. Compile the
# same sources and reuse its build settings for the scheduler regression probe.
get_target_property(scheduler_test_sources triton-core SOURCES)
get_target_property(scheduler_test_source_dir triton-core SOURCE_DIR)
list(TRANSFORM scheduler_test_sources PREPEND "${scheduler_test_source_dir}/")
add_executable(scheduler_regression_test EXCLUDE_FROM_ALL
  scheduler_regression_test.cc ${scheduler_test_sources})
target_compile_features(scheduler_regression_test
  PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(scheduler_regression_test PRIVATE
  $<TARGET_PROPERTY:triton-core,COMPILE_OPTIONS> -g -O1)
target_compile_definitions(scheduler_regression_test
  PRIVATE $<TARGET_PROPERTY:triton-core,COMPILE_DEFINITIONS>)
target_include_directories(scheduler_regression_test
  PRIVATE $<TARGET_PROPERTY:triton-core,INCLUDE_DIRECTORIES>)
target_link_directories(scheduler_regression_test
  PRIVATE $<TARGET_PROPERTY:triton-core,LINK_DIRECTORIES>)
target_link_libraries(scheduler_regression_test
  PRIVATE $<TARGET_PROPERTY:triton-core,LINK_LIBRARIES>)

set(scheduler_test_models "${CMAKE_CURRENT_BINARY_DIR}/scheduler_test_models")
file(MAKE_DIRECTORY "${scheduler_test_models}/empty"
  "${scheduler_test_models}/scheduler_test/1")
add_library(scheduler_test_backend SHARED EXCLUDE_FROM_ALL scheduler_test_backend.cc)
target_include_directories(scheduler_test_backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set_target_properties(scheduler_test_backend PROPERTIES
  OUTPUT_NAME triton_scheduler_test
  LIBRARY_OUTPUT_DIRECTORY "${scheduler_test_models}/scheduler_test")
add_dependencies(scheduler_regression_test scheduler_test_backend)

enable_testing()
foreach(mode instance-removal model-removal capacity lookup)
  add_test(NAME SchedulerRegression.${mode}
    COMMAND scheduler_regression_test "${scheduler_test_models}" ${mode})
  set_tests_properties(SchedulerRegression.${mode} PROPERTIES TIMEOUT 30)
endforeach()
find_program(SCHEDULER_TEST_GDB gdb)
if(SCHEDULER_TEST_GDB)
  add_test(NAME SchedulerRegression.accounting
    COMMAND ${SCHEDULER_TEST_GDB} --batch
      -x "${CMAKE_CURRENT_SOURCE_DIR}/scheduler_regression_test.gdb"
      --args $<TARGET_FILE:scheduler_regression_test>
      "${scheduler_test_models}" accounting)
  set_tests_properties(SchedulerRegression.accounting PROPERTIES TIMEOUT 30)
else()
  message(STATUS "Install GDB to run SchedulerRegression.accounting")
endif()
