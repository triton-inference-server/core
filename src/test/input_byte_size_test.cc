// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "triton/core/tritonserver.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace {

#define FAIL_TEST_IF_ERR(X, MSG)                                              \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
        << "error: " << (MSG) << ": "                                         \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

#define FAIL_TEST_IF_SUCCESS(X, MSG, CONDITION)                               \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_FALSE((err__ == nullptr)) << "error: " << (MSG) << ": ";           \
    ASSERT_THAT(TRITONSERVER_ErrorMessage(err__.get()), (CONDITION))          \
        << "error: "                                                          \
        << "Unexpected error message: "                                       \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

#ifdef TRITON_ENABLE_GPU
#define FAIL_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                     \
    cudaError_t err__ = (X);                                               \
    if (err__ != cudaSuccess) {                                            \
      std::cerr << "error: " << (MSG) << ": " << cudaGetErrorString(err__) \
                << std::endl;                                              \
      exit(1);                                                             \
    }                                                                      \
  } while (false)
#endif  // TRITON_ENABLE_GPU

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = preferred_memory_type_id;

  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
  } else {
    void* allocated_ptr = nullptr;
    allocated_ptr = malloc(byte_size);

    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
    }
  }
  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Notify that the completion.
    auto p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
  }
  TRITONSERVER_InferenceResponseDelete(response);
}

void
SplitBytesInput(
    const char input_data[], const size_t input_data_size,
    size_t first_portion_size, TRITONSERVER_MemoryType memory_type,
    TRITONSERVER_InferenceRequest* irequest)
{
  // Append first buffer
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest, "INPUT0", input_data, first_portion_size, memory_type, 0),
      "assigning INPUT data");

  // Append second buffer
  const size_t second_portion_size = input_data_size - first_portion_size;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest, "INPUT0", input_data + first_portion_size,
          second_portion_size, memory_type, 0),
      "assigning INPUT data");
}

class InputByteSizeTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    // Prepare input array. Concatenate first 4 bytes representing the size
    // of input with the input string bytes.
    const size_t input_element_size = std::strlen(input_element_);
    std::memcpy(
        input_data_string_, reinterpret_cast<const char*>(&input_element_size),
        kElementSizeIndicator_);
    std::memcpy(
        input_data_string_ + kElementSizeIndicator_, input_element_,
        std::strlen(input_element_));

    // Create the server...
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsNew(&server_options),
        "creating server options");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(
            server_options, "./models"),
        "setting model repository path");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetBackendDirectory(
            server_options, "/opt/tritonserver/backends"),
        "setting backend directory");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
            server_options, "/opt/tritonserver/repoagents"),
        "setting repository agent directory");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
        "setting strict model configuration");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerNew(&server_, server_options), "creating server");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsDelete(server_options),
        "deleting server options");
  }

  static void TearDownTestSuite()
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_), "deleting server");
  }

  void SetUp() override
  {
    ASSERT_TRUE(server_ != nullptr) << "Server has not created";
    // Wait until the server is both live and ready.
    size_t health_iters = 0;
    while (true) {
      bool live, ready;
      FAIL_TEST_IF_ERR(
          TRITONSERVER_ServerIsLive(server_, &live),
          "unable to get server liveness");
      FAIL_TEST_IF_ERR(
          TRITONSERVER_ServerIsReady(server_, &ready),
          "unable to get server readiness");
      if (live && ready) {
        break;
      }

      if (++health_iters >= 10) {
        FAIL() << "failed to find healthy inference server";
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Create allocator with common callback
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, ResponseAlloc, ResponseRelease,
            nullptr /* start_fn */),
        "creating response allocator");
  }

  void TearDown() override
  {
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestDelete(irequest_),
        "deleting inference request");
  }

  static TRITONSERVER_Server* server_;
  TRITONSERVER_ResponseAllocator* allocator_ = nullptr;
  static constexpr char input_element_[] = "Example input string";
  static constexpr size_t kElementSizeIndicator_ = sizeof(uint32_t);
  static char input_data_string_[];
  TRITONSERVER_InferenceRequest* irequest_ = nullptr;
  std::promise<TRITONSERVER_Error*> completed_;
};

TRITONSERVER_Server* InputByteSizeTest::server_ = nullptr;
char InputByteSizeTest::input_data_string_
    [kElementSizeIndicator_ + std::strlen(input_element_)] = {};

TEST_F(InputByteSizeTest, ValidInputByteSize)
{
  // Create an inference request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest_, server_, "pt_identity", -1 /* model_version */),
      "creating inference request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest_, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Define input shape and data
  std::vector<int64_t> shape{1, 8};
  std::vector<float> input_data(8, 1);
  const auto input0_byte_size = sizeof(input_data[0]) * input_data.size();

  // Set input for the request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest_, "INPUT0", TRITONSERVER_TYPE_FP32, shape.data(),
          shape.size()),
      "setting input for the request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest_, "INPUT0", input_data.data(), input0_byte_size,
          TRITONSERVER_MEMORY_CPU, 0),
      "assigning INPUT data");

  std::promise<TRITONSERVER_InferenceResponse*> p;
  std::future<TRITONSERVER_InferenceResponse*> future = p.get_future();

  // Set response callback
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest_, allocator_, nullptr /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(&p)),
      "setting response callback");

  // Run inference
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  // Get the inference response
  TRITONSERVER_InferenceResponse* response = future.get();
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceResponseError(response),
      "error with inference response");
  ASSERT_TRUE(response != nullptr) << "Expect successful inference";
}

TEST_F(InputByteSizeTest, InputByteSizeMismatch)
{
  // Create an inference request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest_, server_, "pt_identity", -1 /* model_version */),
      "creating inference request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest_, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Define input shape and data
  std::vector<int64_t> shape{1, 8};
  std::vector<float> input_data(10, 1);
  const auto input0_byte_size = sizeof(input_data[0]) * input_data.size();

  // Set input for the request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest_, "INPUT0", TRITONSERVER_TYPE_FP32, shape.data(),
          shape.size()),
      "setting input for the request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest_, "INPUT0", input_data.data(), input0_byte_size,
          TRITONSERVER_MEMORY_CPU, 0),
      "assigning INPUT data");

  std::promise<TRITONSERVER_InferenceResponse*> p;
  std::future<TRITONSERVER_InferenceResponse*> future = p.get_future();

  // Set response callback
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest_, allocator_, nullptr /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(&p)),
      "setting response callback");

  // Run inference
  constexpr auto err_msg =
      "input byte size mismatch for input 'INPUT0' for model 'pt_identity'. "
      "Expected 32, got 40";
  FAIL_TEST_IF_SUCCESS(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "expect error with inference response", ::testing::HasSubstr(err_msg));
}

TEST_F(InputByteSizeTest, ValidStringInputByteSize)
{
  // Create an inference request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest_, server_, "string_identity", -1 /* model_version */),
      "creating inference request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest_, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Define input shape
  std::vector<int64_t> shape{3};
  const size_t input_data_size = sizeof(input_data_string_);
  const size_t input_element_size = std::strlen(input_element_);

  // Set input for the request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest_, "INPUT0", TRITONSERVER_TYPE_BYTES, shape.data(),
          shape.size()),
      "setting input for the request");
  SplitBytesInput(
      input_data_string_, input_data_size, kElementSizeIndicator_,
      TRITONSERVER_MEMORY_CPU, irequest_);
  SplitBytesInput(
      input_data_string_, input_data_size,
      kElementSizeIndicator_ + input_element_size, TRITONSERVER_MEMORY_CPU,
      irequest_);
  SplitBytesInput(
      input_data_string_, input_data_size,
      kElementSizeIndicator_ + input_element_size / 2, TRITONSERVER_MEMORY_CPU,
      irequest_);

  std::promise<TRITONSERVER_InferenceResponse*> p;
  std::future<TRITONSERVER_InferenceResponse*> future = p.get_future();

  // Set response callback
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest_, allocator_, nullptr /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(&p)),
      "setting response callback");

  // Run inference
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  // Get the inference response
  TRITONSERVER_InferenceResponse* response = future.get();
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceResponseError(response),
      "error with inference response");
  ASSERT_TRUE(response != nullptr) << "Expect successful inference";
}

TEST_F(InputByteSizeTest, StringElementsCountMismatch)
{
  // Create an inference request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest_, server_, "string_identity", -1 /* model_version */),
      "creating inference request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest_, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Define input shape
  std::vector<int64_t> shape{3};
  const size_t input_data_size = sizeof(input_data_string_);
  const size_t input_element_size = std::strlen(input_element_);

  // Set input for the request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest_, "INPUT0", TRITONSERVER_TYPE_BYTES, shape.data(),
          shape.size()),
      "setting input for the request");
  SplitBytesInput(
      input_data_string_, input_data_size, kElementSizeIndicator_,
      TRITONSERVER_MEMORY_CPU, irequest_);
  SplitBytesInput(
      input_data_string_, input_data_size,
      kElementSizeIndicator_ + input_element_size, TRITONSERVER_MEMORY_CPU,
      irequest_);

  // Run inference
  constexpr auto err_msg =
      "expected 3 string elements for inference input 'INPUT0', got 2";
  FAIL_TEST_IF_SUCCESS(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "expect error with inference response", ::testing::HasSubstr(err_msg));
}

TEST_F(InputByteSizeTest, StringElementSizeMisalign)
{
  // Create an inference request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest_, server_, "string_identity", -1 /* model_version */),
      "creating inference request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest_, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Define input shape
  std::vector<int64_t> shape{1};
  const size_t input_data_size = sizeof(input_data_string_);

  // Set input for the request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest_, "INPUT0", TRITONSERVER_TYPE_BYTES, shape.data(),
          shape.size()),
      "setting input for the request");
  // Split element size indicator into two buffers
  SplitBytesInput(
      input_data_string_, input_data_size, 2, TRITONSERVER_MEMORY_CPU,
      irequest_);

  // Run inference
  constexpr auto err_msg =
      "element byte size indicator exceeds the end of the buffer";
  FAIL_TEST_IF_SUCCESS(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "expect error with inference response", ::testing::HasSubstr(err_msg));
}

#ifdef TRITON_ENABLE_GPU
TEST_F(InputByteSizeTest, SkipCUDASharedMemoryChecks)
{
  // Create an inference request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest_, server_, "string_identity", -1 /* model_version */),
      "creating inference request");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest_, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");

  // Define input shape
  std::vector<int64_t> shape{3};
  const size_t input_data_size = sizeof(input_data_string_);
  const size_t input_element_size = std::strlen(input_element_);

  // Allocate CUDA shared memory
  char* d_input_data;
  FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
  FAIL_IF_CUDA_ERR(
      cudaMalloc((void**)&d_input_data, input_data_size),
      "allocating GPU memory for INPUT0 data");
  FAIL_IF_CUDA_ERR(
      cudaMemcpy(
          (void*)d_input_data, input_data_string_, input_data_size,
          cudaMemcpyHostToDevice),
      "setting INPUT0 data in GPU memory");

  // Set input for the request
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest_, "INPUT0", TRITONSERVER_TYPE_BYTES, shape.data(),
          shape.size()),
      "setting input for the request");

  // Assign input data using CUDA shared memory
  SplitBytesInput(
      d_input_data, input_data_size, kElementSizeIndicator_,
      TRITONSERVER_MEMORY_GPU, irequest_);
  SplitBytesInput(
      d_input_data, input_data_size,
      kElementSizeIndicator_ + input_element_size, TRITONSERVER_MEMORY_GPU,
      irequest_);

  std::promise<TRITONSERVER_InferenceResponse*> p;
  std::future<TRITONSERVER_InferenceResponse*> future = p.get_future();

  // Set response callback
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest_, allocator_, nullptr /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(&p)),
      "setting response callback");

  // Run inference
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  // Get the inference response
  TRITONSERVER_InferenceResponse* response = future.get();
  constexpr auto err_msg =
      "expected 3 string elements for inference input 'INPUT0', got 2";
  // Currently byte_size check for string input is skipped at core level thus
  // should not receive the above error message. See details in
  // InferenceRequest::ValidateBytesInputs in infer_request.cc.
  FAIL_TEST_IF_SUCCESS(
      TRITONSERVER_InferenceResponseError(response),
      "error with inference response", Not(::testing::HasSubstr(err_msg)));
  ASSERT_TRUE(response != nullptr) << "Expect successful inference";

  // Clean up CUDA resources
  FAIL_IF_CUDA_ERR(cudaFree(d_input_data), "releasing GPU memory");
}
#endif  // TRITON_ENABLE_GPU

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
