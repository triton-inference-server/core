// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace {

#define FAIL_TEST_IF_ERR(X, MSG)                                              \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
        << "error: " << (MSG) << ": "                                         \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

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
  TRITONSERVER_InferenceRequestDelete(request);
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Notify that the completion.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

class BackendOutputDetailTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
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

    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestNew(
            &irequest_, server_, "add_sub", -1 /* model_version */),
        "creating inference request");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestSetReleaseCallback(
            irequest_, InferRequestComplete,
            nullptr /* request_release_userp */),
        "setting request release callback");

    std::vector<int64_t> input0_shape({16});
    std::vector<int64_t> input1_shape({16});
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAddInput(
            irequest_, "INPUT0", TRITONSERVER_TYPE_FP32, &input0_shape[0],
            input0_shape.size()),
        "setting input0 for the request");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest_, "INPUT0", &input0_data_[0], input0_data_.size(),
            TRITONSERVER_MEMORY_CPU, 0),
        "assigning INPUT data");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAddInput(
            irequest_, "INPUT1", TRITONSERVER_TYPE_FP32, &input1_shape[0],
            input1_shape.size()),
        "setting input1 for the request");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAppendInputData(
            irequest_, "INPUT1", &input1_data_[0], input1_data_.size(),
            TRITONSERVER_MEMORY_CPU, 0),
        "assigning INPUT1 data");
  }

  void TearDown() override
  {
    unsetenv("TEST_ANONYMOUS");
    unsetenv("TEST_BYTE_SIZE");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
  }

  static TRITONSERVER_Server* server_;
  TRITONSERVER_ResponseAllocator* allocator_ = nullptr;
  static std::vector<float> input0_data_;
  static std::vector<float> input1_data_;
  TRITONSERVER_InferenceRequest* irequest_ = nullptr;
};

TRITONSERVER_Server* BackendOutputDetailTest::server_ = nullptr;
std::vector<float> BackendOutputDetailTest::input0_data_(16, 1);
std::vector<float> BackendOutputDetailTest::input1_data_(16, 1);

TEST_F(BackendOutputDetailTest, DefaultInference)
{
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  std::future<TRITONSERVER_InferenceResponse*> future = p->get_future();

  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest_, allocator_, nullptr /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(p)),
      "setting response callback");

  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerInferAsync(server_, irequest_, nullptr /* trace */),
      "running inference");

  uint32_t output_count;
  const char* output_name;
  TRITONSERVER_DataType output_datatype;
  const int64_t* output_shape;
  uint64_t dims_count;
  std::vector<const char*> names = {"OUTPUT0", "OUTPUT1"};

  TRITONSERVER_InferenceResponse* response = future.get();
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceResponseError(response),
      "error with inference response");
  ASSERT_TRUE(response != nullptr) << "Expect successful inference";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
      "getting output count");
  ASSERT_EQ(output_count, size_t(2));


  for (size_t idx = 0; idx < output_count; idx++) {
    // Cast the response from a TRITONSERVER_Response to a
    // TRITONBACKEND_Response. This is not recommended and not allowed for
    // backend developers as this cast is unsupported. However, for the purposes
    // of our own internal testing we do so here in order to validate the
    // functionality of our backend APIs.
    TRITONBACKEND_Response* backend_response =
        reinterpret_cast<TRITONBACKEND_Response*>(response);

    FAIL_TEST_IF_ERR(
        TRITONBACKEND_InferenceResponseOutput(
            backend_response, idx, &output_name, &output_datatype,
            &output_shape, &dims_count),
        "getting output details by index");
    EXPECT_EQ(*output_name, *names[idx]);
    EXPECT_EQ(output_datatype, TRITONSERVER_TYPE_FP32);
    EXPECT_EQ(*output_shape, int64_t(16));
    EXPECT_EQ(dims_count, int64_t(1));

    FAIL_TEST_IF_ERR(
        TRITONBACKEND_InferenceResponseOutputByName(
            backend_response, names[idx], &output_datatype, &output_shape,
            &dims_count),
        "getting output details by name");
    EXPECT_EQ(output_datatype, TRITONSERVER_TYPE_FP32);
    EXPECT_EQ(*output_shape, int64_t(16));
    EXPECT_EQ(dims_count, int64_t(1));
  }
  TRITONSERVER_InferenceResponseDelete(response);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
