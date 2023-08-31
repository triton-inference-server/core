// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  }


  static TRITONSERVER_Server* server_;
};

TRITONSERVER_Server* RequestCancellationTest::server_ = nullptr;


TEST_F(RequestCancellationTest, Cancellation)
{
  TRITONSERVER_InferenceRequest* request;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestNew(&request, server_, "model", 1));
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
      request, nullptr, nullptr, nullptr, nullptr));
  TRITONBACKEND_Request* backend_request =
      reinterpret_cast<TRITONBACKEND_Request*>(request);
  TRITONBACKEND_ResponseFactory* response_factory;
  FAIL_TEST_IF_ERR(
      TRITONBACKEND_ResponseFactoryNew(&response_factory, backend_request));

  bool is_cancelled = true;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestIsCancelled(request, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  is_cancelled = true;
  FAIL_TEST_IF_ERR(
      TRITONBACKEND_RequestIsCancelled(backend_request, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  is_cancelled = true;
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryIsCancelled(
      response_factory, &is_cancelled));
  ASSERT_FALSE(is_cancelled);

  is_cancelled = false;
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestCancel(request));

  FAIL_TEST_IF_ERR(
      TRITONBACKEND_RequestIsCancelled(backend_request, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  is_cancelled = false;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestIsCancelled(request, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  is_cancelled = false;
  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryIsCancelled(
      response_factory, &is_cancelled));
  ASSERT_TRUE(is_cancelled);

  FAIL_TEST_IF_ERR(TRITONBACKEND_ResponseFactoryDelete(response_factory));
  FAIL_TEST_IF_ERR(TRITONSERVER_InferenceRequestDelete(request));
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
