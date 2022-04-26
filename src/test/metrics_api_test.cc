// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_METRICS

#include <iostream>
#include <thread>
#include "gtest/gtest.h"
#include "triton/common/logging.h"
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

// Test Fixture
class MetricsApiTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    // Create server object to pass when retrieving metrics.
    // NOTE: It is currently not required to pass a valid server object to
    //       TRITONSERVER_ServerMetrics, but is more future-proof to include.
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsNew(&server_options),
        "creating server options");
    // This test doesn't require the use of any models, so we use "." as repo
    // and set ModelControlMode to EXPLICIT to avoid attempting to load models
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, "."),
        "setting model repository path");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelControlMode(
            server_options, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
        "setting model control mode");
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

  void SetUp() override {}
  void TearDown() override {}

  double increment_ = 10;
  double set_value_ = 42;
  double value_ = -1;
  double prev_value_ = -1;
  static TRITONSERVER_Server* server_;
};

TRITONSERVER_Server* MetricsApiTest::server_ = nullptr;

// Test end-to-end flow of Generic Metrics API for Counter metric
TEST_F(MetricsApiTest, TestCounterEndToEnd)
{
  // Create metric family
  TRITONSERVER_MetricFamily* family;
  TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* name = "api_counter_example";
  const char* description = "this is an example counter metric added via API.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family, kind, name, description),
      "Creating new metric family");

  // Create metric
  TRITONSERVER_Metric* metric;
  std::vector<const TRITONSERVER_Parameter*> labels;
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "example1", TRITONSERVER_PARAMETER_STRING, "counter_label1"));
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "example2", TRITONSERVER_PARAMETER_STRING, "counter_label2"));
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricNew(&metric, family, labels.data(), labels.size()),
      "Creating new metric");
  for (const auto label : labels) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }

  // Value should be zero initially
  value_ = -1;
  prev_value_ = value_;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric, &value_), "query metric initial value");
  ASSERT_EQ(value_, 0.0);

  // Increment Positively
  prev_value_ = value_;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricIncrement(metric, increment_),
      "increase metric value");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric, &value_),
      "query metric value after increment");
  ASSERT_EQ(value_, prev_value_ + increment_);

  // Verify negative increment fails on counter metric
  auto err = TRITONSERVER_MetricIncrement(metric, -1.0 * increment_);
  ASSERT_NE(err, nullptr);

  // Verify set fails on counter metric
  err = TRITONSERVER_MetricSet(metric, set_value_);
  ASSERT_NE(err, nullptr);

  // GetMetricKind
  TRITONSERVER_MetricKind kind_tmp;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_GetMetricKind(metric, &kind_tmp), "query metric kind");
  ASSERT_EQ(kind_tmp, kind);

  // Check metrics via C API
  ASSERT_NE(server_, nullptr);
  TRITONSERVER_Metrics* metrics = nullptr;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerMetrics(server_, &metrics), "fetch metrics");
  const char* base;
  size_t byte_size;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricsFormatted(
          metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size),
      "format metrics string");
  auto metrics_str = std::string(base, byte_size);

  // Assert custom metric is reported and found in output
  auto found = metrics_str.find(std::string("# HELP ") + name);
  std::cout << metrics_str << std::endl;
  ASSERT_NE(found, std::string::npos);
  std::cout << "======" << std::endl;
  std::cout << metrics_str.substr(found, std::string::npos) << std::endl;

  // Cleanup
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricDelete(metric), "delete metric");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyDelete(family), "delete metric family");
}

// Test end-to-end flow of Generic Metrics API for Gauge metric
TEST_F(MetricsApiTest, TestGaugeEndToEnd)
{
  // Create metric family
  TRITONSERVER_MetricFamily* family;
  TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_GAUGE;
  const char* name = "api_gauge_example";
  const char* description = "this is an example gauge metric added via API.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family, kind, name, description),
      "Creating new metric family");

  // Create metric
  TRITONSERVER_Metric* metric;
  std::vector<const TRITONSERVER_Parameter*> labels;
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "example1", TRITONSERVER_PARAMETER_STRING, "gauge_label1"));
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "example2", TRITONSERVER_PARAMETER_STRING, "gauge_label2"));
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricNew(&metric, family, labels.data(), labels.size()),
      "Creating new metric");
  for (const auto label : labels) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }

  // Value should be zero initially
  value_ = -1;
  prev_value_ = value_;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric, &value_), "query metric initial value");
  ASSERT_EQ(value_, 0.0);

  // Increment positively
  prev_value_ = value_;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricIncrement(metric, increment_),
      "increase metric value");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric, &value_),
      "query metric value after positive increment");
  ASSERT_EQ(value_, prev_value_ + increment_);

  // Increment negatively
  prev_value_ = value_;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricIncrement(metric, -1.0 * increment_),
      "decrease metric value");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric, &value_),
      "query metric value after negative increment");
  ASSERT_EQ(value_, prev_value_ + (-1.0 * increment_));

  // Set
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricSet(metric, set_value_), "set metric");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric, &value_),
      "query metric value after set");
  ASSERT_EQ(value_, set_value_);

  TRITONSERVER_MetricKind kind_tmp;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_GetMetricKind(metric, &kind_tmp), "query metric kind");
  ASSERT_EQ(kind_tmp, kind);

  // Check metrics via C API
  ASSERT_NE(server_, nullptr);
  TRITONSERVER_Metrics* metrics = nullptr;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerMetrics(server_, &metrics), "fetch metrics");
  const char* base;
  size_t byte_size;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricsFormatted(
          metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size),
      "format metrics string");
  auto metrics_str = std::string(base, byte_size);

  // Assert custom metric is reported and found in output
  auto found = metrics_str.find(std::string("# HELP ") + name);
  std::cout << metrics_str << std::endl;
  ASSERT_NE(found, std::string::npos);
  std::cout << "======" << std::endl;
  std::cout << metrics_str.substr(found, std::string::npos) << std::endl;

  // Cleanup
  FAIL_TEST_IF_ERR(TRITONSERVER_MetricDelete(metric), "delete metric");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyDelete(family), "delete metric family");
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
