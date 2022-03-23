// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define TRITON_ENABLE_METRICS 1
#ifdef TRITON_ENABLE_METRICS

#include <iostream>
#include <thread>
#include "gtest/gtest.h"
#include "metric_family.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

namespace {

// TODO: Remove
#define FAIL_TEST_IF_ERR(X, MSG)                                              \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
        << "error: " << (MSG) << ": "                                         \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

// Test Fixture
class MetricsApiTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    // Create the server...
    /*TRITONSERVER_ServerOptions* server_options = nullptr;
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
        "deleting server options");*/
  }

  static void TearDownTestSuite()
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_), "deleting server");
  }

  void SetUp() override
  {
    //std::cout << "Calling setup test suite" << std::endl;
    //SetUpTestSuite();
    //std::cout << "Finished calling setup test suite" << std::endl;
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
  }

  static TRITONSERVER_Server* server_;
};

TRITONSERVER_Server* MetricsApiTest::server_ = nullptr;

// Test end-to-end flow of Generic Metrics API
TEST_F(MetricsApiTest, TestEndToEnd)
{
  // Create counter family
  TRITONSERVER_MetricFamily* cfamily;
  TRITONSERVER_MetricKind ckind = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* cname = "backend_counter_example";
  const char* cdescription = "this is an example counter metric added by a backend.";
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricFamilyNew(&cfamily, ckind, cname, cdescription), "Creating new metric family");

  // Create counter metric
  TRITONSERVER_Metric* cmetric;
  TRITONSERVER_Parameter** labels = nullptr; 
  const int num_labels = 0;
  // TODO: Use real labels
  //auto label = TRITONSERVER_ParameterNew("key", TRITONSERVER_PARAMETER_STRING, "value");
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricNew(&cmetric, cfamily, labels, num_labels), "Creating new metric");

  // Value of counter should be zero initially
  double value = -1;
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricValue(cmetric, &value), "query metric value 1");
  ASSERT_EQ(value, 0.0);

  FAIL_TEST_IF_ERR(TRITONSERVER_MetricIncrement(cmetric, 100), "increment metric");
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricValue(cmetric, &value), "query metric value 2");
  ASSERT_EQ(value, 100.0);

  TRITONSERVER_MetricKind kind;
  FAIL_TEST_IF_ERR(TRITONSERVER_GetMetricKind(cmetric, &kind), "query metric kind");
  ASSERT_EQ(kind, ckind);

  // Cleanup
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricDelete(cmetric), "delete metric");
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricFamilyDelete(cfamily), "delete metric family");
}

}  // namespace

int
main(int argc, char** argv)
{
#ifdef TRITON_ENABLE_LOGGING
  LOG_SET_VERBOSE(1);
#endif  // TRITON_ENABLE_LOGGING

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif  // TRITON_ENABLE_METRICS
