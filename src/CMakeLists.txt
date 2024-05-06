# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(libtritonserver LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# We must include the transitive closure of all repos so that we can
# override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)

set(TRITON_COMMON_ENABLE_PROTOBUF ON)

FetchContent_MakeAvailable(repo-common)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
endif() # TRITON_ENABLE_GPU

#
# Boost
#
# Minimum of 1.78 required for use of boost::span. This can eventually be
# relaxed and replaced with std::span in C++20.
find_package(Boost 1.78 REQUIRED COMPONENTS)
message(STATUS "Using Boost ${Boost_VERSION}")

#
# Protobuf
#
set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

#
# Prometheus
#
if(${TRITON_ENABLE_METRICS})
  find_package(prometheus-cpp CONFIG REQUIRED)
  message(STATUS "Using prometheus-cpp ${prometheus-cpp_VERSION}")
endif() # TRITON_ENABLE_METRICS

#
# GCS
#
if(${TRITON_ENABLE_GCS})
  find_package(google_cloud_cpp_storage REQUIRED)
  message(STATUS "Using google-cloud-cpp ${google_cloud_cpp_storage_VERSION}")
endif() # TRITON_ENABLE_GCS

#
# S3
#
if(${TRITON_ENABLE_S3})
  find_package(ZLIB REQUIRED)
  find_package(AWSSDK REQUIRED COMPONENTS s3)
  message(STATUS "Using aws-sdk-cpp ${AWSSDK_VERSION}")
endif()

#
# Azure Storage
#
if(${TRITON_ENABLE_AZURE_STORAGE})
  find_package(azure-storage-blobs-cpp CONFIG REQUIRED)
  message(STATUS "Using Azure storage blobs ${azure-storage-blobs-cpp_VERSION}")
endif()

configure_file(libtritonserver.ldscript libtritonserver.ldscript COPYONLY)

set(
  SERVER_SRCS
  backend_config.cc
  backend_manager.cc
  backend_memory_manager.cc
  backend_model.cc
  backend_model_instance.cc
  buffer_attributes.cc
  cache_entry.cc
  cache_manager.cc
  cuda_utils.cc
  dynamic_batch_scheduler.cc
  ensemble_scheduler/ensemble_scheduler.cc
  ensemble_scheduler/ensemble_utils.cc
  filesystem/api.cc
  infer_parameter.cc
  infer_request.cc
  infer_response.cc
  infer_stats.cc
  infer_trace.cc
  instance_queue.cc
  label_provider.cc
  memory.cc
  metric_model_reporter.cc
  metrics.cc
  metric_family.cc
  model.cc
  model_config_utils.cc
  model_repository_manager/model_lifecycle.cc
  model_repository_manager/model_repository_manager.cc
  numa_utils.cc
  payload.cc
  pinned_memory_manager.cc
  rate_limiter.cc
  repo_agent.cc
  scheduler_utils.cc
  sequence_batch_scheduler/sequence_batch_scheduler.cc
  sequence_batch_scheduler/sequence_utils.cc
  sequence_state.cc
  server.cc
  shared_library.cc
  status.cc
  tritoncache.cc
  tritonserver.cc
)

set(
  SERVER_HDRS
  backend_config.h
  backend_manager.h
  backend_memory_manager.h
  backend_model.h
  backend_model_instance.h
  buffer_attributes.h
  cache_entry.h
  cache_manager.h
  constants.h
  cuda_utils.h
  dynamic_batch_scheduler.h
  ensemble_scheduler/ensemble_scheduler.h
  ensemble_scheduler/ensemble_utils.h
  filesystem/api.h
  infer_parameter.h
  infer_request.h
  infer_response.h
  infer_stats.h
  infer_trace.h
  instance_queue.h
  label_provider.h
  memory.h
  metric_model_reporter.h
  metrics.h
  metric_family.h
  model_config_utils.h
  model.h
  model_repository_manager/model_lifecycle.h
  model_repository_manager/model_repository_manager.h
  numa_utils.h
  payload.h
  pinned_memory_manager.h
  rate_limiter.h
  repo_agent.h
  response_allocator.h
  scheduler.h
  scheduler_utils.h
  sequence_batch_scheduler/sequence_batch_scheduler.h
  sequence_batch_scheduler/sequence_utils.h
  sequence_state.h
  server.h
  server_message.h
  shared_library.h
  status.h
  tritonserver_apis.h
)

if(${TRITON_ENABLE_GPU})
  set(
    SERVER_SRCS
    ${SERVER_SRCS}
    cuda_memory_manager.cc
    cuda_block_manager.h
    model_config_cuda.cc
  )
  set(
    SERVER_HDRS
    ${SERVER_HDRS}
    cuda_memory_manager.h
    cuda_block_manager.cc
    model_config_cuda.h
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ENSEMBLE})
  set(
    SERVER_SRCS
    ${SERVER_SRCS}
    ensemble_scheduler/ensemble_model.cc
  )
  set(
    SERVER_HDRS
    ${SERVER_HDRS}
    ensemble_scheduler/ensemble_model.h
  )
endif() # TRITON_ENABLE_ENSEMBLE

add_library(
  triton-core SHARED
  ${SERVER_SRCS} ${SERVER_HDRS}
)

add_library(
  TritonCore::triton-core ALIAS triton-core
)

target_compile_features(triton-core PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
		  "If the target system is not Windows 10, please update _WIN32_WINNT "
		  "to corresponding value.")
  target_compile_options(
    triton-core
    PRIVATE
      /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
  )
else()
  target_compile_options(
    triton-core
    PRIVATE
      -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -Werror
  )
  set_target_properties(
    triton-core
    PROPERTIES
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtritonserver.ldscript
      LINK_FLAGS "-Wl,--version-script libtritonserver.ldscript"
  )
endif()

set_target_properties(
  triton-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME tritonserver
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}:$\{ORIGIN\}/pytorch"
)

target_include_directories(
  triton-core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
)

if(${TRITON_ENABLE_GPU})
  target_include_directories(
    triton-core
    PRIVATE
      ${CNMEM_PATH}/include
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_METRICS})
  target_include_directories(
    triton-core
    PRIVATE $<TARGET_PROPERTY:prometheus-cpp::core,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_GCS})
  target_include_directories(
    triton-core
    PRIVATE $<TARGET_PROPERTY:google-cloud-cpp::storage,INTERFACE_INCLUDE_DIRECTORIES>
  )
  if (NOT WIN32)
    # [WIP] still needed?
    set_source_files_properties(
      filesystem/api.cc
      PROPERTIES
        COMPILE_FLAGS -Wno-missing-field-initializers
    )
  endif() # NOT WIN32
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_S3})
  target_include_directories(
    triton-core
    PRIVATE $<TARGET_PROPERTY:aws-cpp-sdk-s3,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

if(${TRITON_ENABLE_AZURE_STORAGE})
  target_include_directories(
    triton-core
    PRIVATE $<TARGET_PROPERTY:Azure::azure-storage-blobs,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRITON_ENABLE_AZURE_STORAGE

target_compile_definitions(
  triton-core
  PRIVATE TRITON_VERSION="${TRITON_VERSION}"
)

if(${TRITON_ENABLE_NVTX})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_NVTX=1
  )
endif() # TRITON_ENABLE_NVTX

if(${TRITON_ENABLE_TRACING})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_TRACING=1
  )
endif() # TRITON_ENABLE_TRACING

if(${TRITON_ENABLE_LOGGING})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_LOGGING=1
  )
endif() # TRITON_ENABLE_LOGGING

if(${TRITON_ENABLE_STATS})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_STATS=1
  )
endif() # TRITON_ENABLE_STATS

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  if(CUDA_VERSION VERSION_GREATER "10.1" OR CUDA_VERSION VERSION_EQUAL "10.1")
    target_compile_definitions(
      triton-core
      PRIVATE TRITON_ENABLE_CUDA_GRAPH=1
    )
  else()
    message(WARNING "CUDA ${CUDA_VERSION} does not support CUDA graphs.")
  endif()
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_MALI_GPU})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_MALI_GPU=1
  )
endif() # TRITON_ENABLE_MALI_GPU

if(${TRITON_ENABLE_METRICS})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_METRICS=1
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_METRICS_GPU})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_METRICS_GPU=1
  )
endif() # TRITON_ENABLE_METRICS_GPU

if(${TRITON_ENABLE_METRICS_CPU})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_METRICS_CPU=1
  )
endif() # TRITON_ENABLE_METRICS_CPU

if(${TRITON_ENABLE_GCS})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_GCS=1
  )
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_AZURE_STORAGE})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_AZURE_STORAGE=1
  )
endif() # TRITON_ENABLE_AZURE_STORAGE

if(${TRITON_ENABLE_S3})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_S3=1
  )
endif() # TRITON_ENABLE_S3

if(${TRITON_ENABLE_ENSEMBLE})
  target_compile_definitions(
    triton-core
    PRIVATE TRITON_ENABLE_ENSEMBLE=1
  )
endif() # TRITON_ENABLE_ENSEMBLE

FOREACH(p ${TRITON_EXTRA_LIB_PATHS})
  target_link_directories(
    triton-core
    PRIVATE ${p}
  )
ENDFOREACH(p)

# re2 requires Threads::Threads to be available in the
# list of targets
find_package(Threads)
find_package(re2 REQUIRED)

target_link_libraries(
  triton-core
  PRIVATE
    proto-library                    # from repo-common
    triton-common-async-work-queue   # from repo-common
    triton-common-thread-pool        # from repo-common
    triton-common-error              # from repo-common
    triton-common-model-config       # from repo-common
    triton-common-logging            # from repo-common
    triton-common-json               # from repo-common
    triton-common-table-printer      # from repo-common
    protobuf::libprotobuf
    re2::re2
)

if (NOT WIN32)
  target_link_libraries(
    triton-core
    PRIVATE
      dl
      numa
  )
endif()

if(${TRITON_ENABLE_METRICS})
  target_link_libraries(
    triton-core
    PRIVATE prometheus-cpp::core
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_GCS})
  target_link_libraries(
    triton-core
    PRIVATE
      google-cloud-cpp::storage
  )
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_S3})
  target_link_libraries(
    triton-core
    PRIVATE
      aws-cpp-sdk-s3
  )
endif() # TRITON_ENABLE_S3

if(${TRITON_ENABLE_AZURE_STORAGE})
  target_link_libraries(
    triton-core
    PRIVATE
      Azure::azure-storage-blobs
  )
endif() # TRITON_ENABLE_AZURE_STORAGE

if(${TRITON_ENABLE_GPU})
  find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)
  target_link_libraries(
    triton-core
    PRIVATE
      ${CNMEM_LIBRARY}
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_METRICS_GPU})
  target_link_libraries(
    triton-core
    PRIVATE
      dcgm
  )
endif() # TRITON_ENABLE_METRICS_GPU

install(
  TARGETS
    triton-core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Currently unit tests do not build for windows...
if (NOT WIN32)
  add_subdirectory(test test)
endif() # NOT WIN32
