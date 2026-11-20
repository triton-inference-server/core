// Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <future>
#include <thread>

#include "gtest/gtest.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"


#define FAIL_TEST_IF_ERR(X)                                                   \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
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
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    // Notify that the server has released the request.
    std::promise<void>* p = reinterpret_cast<std::promise<void>*>(userp);
    p->set_value();
    delete p;
  }
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

class RequestCancellationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    // Create the server...
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
        server_options, "./models"));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(
        server_options, "/opt/tritonserver/backends"));
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetLogVerbose(server_options, 1));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
        server_options, "/opt/tritonserver/repoagents"));
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true));

    FAIL_TEST_IF_ERR(TRITONSERVER_ServerNew(&server_, server_options));
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options));
  }

  static void TearDownTestSuite()
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_));
  }

  void SetUp() override
  {
    ASSERT_TRUE(server_ != nullptr) << "Server has not created";
    // Wait until the server is both live and ready.
    size_t health_iters = 0;
    while (true) {
      bool live, ready;
      FAIL_TEST_IF_ERR(TRITONSERVER_ServerIsLive(server_, &live));
      FAIL_TEST_IF_ERR(TRITONSERVER_ServerIsReady(server_, &ready));
      if (live && ready) {
        break;
      }

      if (++health_iters >= 10) {
        FAIL() << "failed to find healthy inference server";
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Create allocator with common callback
    FAIL_TEST_IF_ERR(TRITONSERVER_ResponseAllocatorNew(
        &allocator_, ResponseAlloc, ResponseRelease, nullptr /* start_fn */));

    FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestNew(
        &irequest_, server_, "model", -1 /* model_version */));

    std::vector<int64_t> input0_shape({1, 1000});
    FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        irequest_, "INPUT0", TRITONSERVER_TYPE_INT32, input0_shape.data(),
        input0_shape.size()));

    const auto input0_byte_size = sizeof(input0_data_[0]) * input0_data_.size();
    FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
        irequest_, "INPUT0", input0_data_.data(), input0_byte_size,
        TRITONSERVER_MEMORY_CPU, 0));
  }

  void TearDown() override
  {
    ASSERT_NO_FATAL_FAILURE(DeleteRequest());
    FAIL_TEST_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator_));
  }

  // Submits 'irequest_' with a fresh release callback. The server owns the
  // request until the callback runs, unless the submission fails.
  TRITONSERVER_Error* InferAsync()
  {
    auto p = new std::promise<void>();
    std::future<void> release_future = p->get_future();
    TRITONSERVER_Error* err = TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest_, InferRequestComplete, reinterpret_cast<void*>(p));
    if (err == nullptr) {
      err = TRITONSERVER_ServerInferAsync(
          server_, irequest_, nullptr /* trace */);
    }
    if (err == nullptr) {
      release_future_ = std::move(release_future);
    } else {
      delete p;
    }
    return err;
  }

  // Returns true once the server has released 'irequest_', after which the
  // request may be reused or deleted.
  bool WaitForRequestRelease()
  {
    return release_future_.valid() &&
           (release_future_.wait_for(std::chrono::seconds(30)) ==
            std::future_status::ready);
  }

  // Deletes 'irequest_', first waiting for the server to release it if it was
  // submitted.
  void DeleteRequest()
  {
    if (release_future_.valid()) {
      ASSERT_TRUE(WaitForRequestRelease())
          << "request was not released by the server";
      release_future_ = std::future<void>();
    }
    if (irequest_ != nullptr) {
      FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestDelete(irequest_));
      irequest_ = nullptr;
    }
  }

  static TRITONSERVER_Server* server_;
  TRITONSERVER_ResponseAllocator* allocator_ = nullptr;
  static std::vector<int32_t> input0_data_;
  TRITONSERVER_InferenceRequest* irequest_ = nullptr;
  std::future<void> release_future_;
};

TRITONSERVER_Server* RequestCancellationTest::server_ = nullptr;
std::vector<int32_t> RequestCancellationTest::input0_data_(1000, 1);

TEST_F(RequestCancellationTest, Cancellation)
{
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  std::future<TRITONSERVER_InferenceResponse*> future = p->get_future();

  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
      irequest_, allocator_, nullptr /* response_allocator_userp */,
      InferResponseComplete, reinterpret_cast<void*>(p)));

  TRITONBACKEND_Request* backend_request =
      reinterpret_cast<TRITONBACKEND_Request*>(irequest_);

  FAIL_TEST_IF_ERR(InferAsync());
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestCancel(irequest_));

  TRITONBACKEND_ResponseFactory* response_factory;
  FAIL_TEST_IF_ERR(
      TRITONBACKEND_ResponseFactoryNew(&response_factory, backend_request));

  bool is_cancelled = false;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestIsCancelled(irequest_, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  is_cancelled = false;
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryIsCancelled(
      response_factory, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  TRITONSERVER_InferenceResponse* response = future.get();
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceResponseDelete(response));
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryDelete(response_factory));

  // The final response may be delivered before the request is released, so
  // wait for the release before reusing the request.
  ASSERT_TRUE(WaitForRequestRelease())
      << "request was not released by the server";

  p = new std::promise<TRITONSERVER_InferenceResponse*>();
  future = p->get_future();

  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
      irequest_, allocator_, nullptr /* response_allocator_userp */,
      InferResponseComplete, reinterpret_cast<void*>(p)));

  // Sending another request and the request should not be cancelled.
  FAIL_TEST_IF_ERR(InferAsync());
  FAIL_TEST_IF_ERR(
      TRITONBACKEND_ResponseFactoryNew(&response_factory, backend_request));

  is_cancelled = true;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestIsCancelled(irequest_, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  is_cancelled = true;
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryIsCancelled(
      response_factory, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  response = future.get();
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceResponseDelete(response));
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryDelete(response_factory));
}

TEST_F(RequestCancellationTest, CancellationAfterRelease)
{
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  std::future<TRITONSERVER_InferenceResponse*> future = p->get_future();

  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
      irequest_, allocator_, nullptr /* response_allocator_userp */,
      InferResponseComplete, reinterpret_cast<void*>(p)));

  FAIL_TEST_IF_ERR(InferAsync());

  TRITONBACKEND_Request* backend_request =
      reinterpret_cast<TRITONBACKEND_Request*>(irequest_);
  TRITONBACKEND_ResponseFactory* response_factory;
  FAIL_TEST_IF_ERR(
      TRITONBACKEND_ResponseFactoryNew(&response_factory, backend_request));

  // The server owns the request until it is released, so wait for the
  // release rather than releasing the request from the test.
  ASSERT_TRUE(WaitForRequestRelease())
      << "request was not released by the server";

  bool is_cancelled = true;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestIsCancelled(irequest_, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  is_cancelled = true;
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryIsCancelled(
      response_factory, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  is_cancelled = false;
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestCancel(irequest_));

  is_cancelled = false;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestIsCancelled(irequest_, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  TRITONSERVER_InferenceResponse* response = future.get();
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceResponseDelete(response));

  is_cancelled = false;
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryIsCancelled(
      response_factory, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryDelete(response_factory));
}

int
main(int argc, char** argv)
{
#ifdef TRITON_ENABLE_LOGGING
  LOG_SET_VERBOSE(2);
#endif  // TRITON_ENABLE_LOGGING

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
