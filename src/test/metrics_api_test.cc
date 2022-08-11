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

// Helpers
void PrintMetrics(TRITONSERVER_Server* server, std::string substr) {
  // Check metrics via C API
  ASSERT_NE(server, nullptr);
  TRITONSERVER_Metrics* metrics = nullptr;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerMetrics(server, &metrics), "fetch metrics");
  const char* base;
  size_t byte_size;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricsFormatted(
          metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size),
      "format metrics string");
  auto metrics_str = std::string(base, byte_size);

  // Only print substr if provided and found in output
  if (!substr.empty()) {
      auto found = metrics_str.find(std::string(substr));
      ASSERT_NE(found, std::string::npos);
      std::cout << "======" << std::endl;
      std::cout << metrics_str.substr(found, std::string::npos) << std::endl;
  } else {
      std::cout << metrics_str << std::endl;
  }
}

// Test Fixture
class MetricsApiTest : public ::testing::Test {
 protected:
  // Run only once before entire set of tests
  static void SetUpTestSuite() {}
  // Run only once after entire set of tests
  static void TearDownTestSuite() {}

  // Run before each test
  void SetUp() override
  {
    // Create server object to pass when retrieving metrics.
    // NOTE: It is currently not required to pass a valid server object to
    //       TRITONSERVER_ServerMetrics, but is more future-proof to include.
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsNew(&server_options),
        "creating server options");
    // Mute info output for the sake of this test, less output
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetLogInfo(server_options, false),
        "disabling log INFO for brevity");
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

  // TODO: See if prometheus registry is actually getting cleared or not across
  // server deletions.
  // Run after each test
  void TearDown() override
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_), "deleting server");
    server_ = nullptr;
  }

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

// Test duplicate metric family names with different descriptions
TEST_F(MetricsApiTest, TestDupeMetricFamilyDiffDescription)
{
  // Create metric family
  TRITONSERVER_MetricFamily* family1;
  TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* name = "api_counter_example";
  const char* description1 = "this is an example counter metric added via API.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family1, kind, name, description1),
      "Creating new metric family1");

  // Create duplicate metric family
  TRITONSERVER_MetricFamily* family2;
  const char* description2 = "this is a different description.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family2, kind, name, description2),
      "Creating new metric family2");

  std::cout << "family1 address: 0x" << family1 << std::endl;
  std::cout << "family2 address: 0x" << family2 << std::endl;
  // TODO
  ASSERT_NE(family1, family2);
  PrintMetrics(server_, "");
}

// Test duplicate metric family names with different descriptions
TEST_F(MetricsApiTest, TestDupeMetricFamilyDiffKind)
{
  // Create metric family
  TRITONSERVER_MetricFamily* family1;
  TRITONSERVER_MetricKind kind1 = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* name = "api_example";
  const char* description = "this is an example counter metric added via API.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family1, kind1, name, description),
      "Creating new metric family1");

  // Create duplicate metric family
  TRITONSERVER_MetricFamily* family2;
  TRITONSERVER_MetricKind kind2 = TRITONSERVER_METRIC_KIND_GAUGE;
  // TODO: THIS SHOULD FAIL
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family2, kind2, name, description),
      "Creating new metric family2");

  std::cout << "family1 address: 0x" << family1 << std::endl;
  std::cout << "family2 address: 0x" << family2 << std::endl;
  // TODO
  ASSERT_NE(family1, family2);
  PrintMetrics(server_, "");
}

// Test duplicate metric family objects
TEST_F(MetricsApiTest, TestDupeMetricFamily)
{
  // Create metric family
  TRITONSERVER_MetricFamily* family1;
  TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* name = "api_counter_example";
  const char* description = "this is an example counter metric added via API.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family1, kind, name, description),
      "Creating new metric family1");

  // Create duplicate metric family
  TRITONSERVER_MetricFamily* family2;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family2, kind, name, description),
      "Creating new metric family2");

  std::cout << "family1 address: 0x" << family1 << std::endl;
  std::cout << "family2 address: 0x" << family2 << std::endl;
  // TODO
  ASSERT_NE(family1, family2);
  PrintMetrics(server_, "");
}

// Test duplicate metric objects
TEST_F(MetricsApiTest, TestDupeMetricLabels)
{
  // Create metric family
  TRITONSERVER_MetricFamily* family;
  TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* name = "api_counter_example";
  const char* description = "this is an example counter metric added via API.";
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricFamilyNew(&family, kind, name, description),
      "Creating new metric family1");

  // Create metric
  TRITONSERVER_Metric* metric1;
  std::vector<const TRITONSERVER_Parameter*> labels;
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "example1", TRITONSERVER_PARAMETER_STRING, "counter_label1"));
  labels.emplace_back(TRITONSERVER_ParameterNew(
      "example2", TRITONSERVER_PARAMETER_STRING, "counter_label2"));
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricNew(&metric1, family, labels.data(), labels.size()),
      "Creating new metric");

  // Create duplicate metric
  TRITONSERVER_Metric* metric2;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricNew(&metric2, family, labels.data(), labels.size()),
      "Creating new metric");

  // Cleanup
  for (const auto label : labels) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
  }

  std::cout << "metric1 address: 0x" << metric1 << std::endl;
  std::cout << "metric2 address: 0x" << metric2 << std::endl;
  // Verify dupe metrics are different pointers
  ASSERT_NE(metric1, metric2);

  // Verify dupe metrics reference same underlying metric
  double value1 = -1;
  double value2 = -1;
  double inc = 7.5;

  // Verify initial values of zero
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric1, &value1),
      "query metric value after increment");
  ASSERT_EQ(value1, 0);
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric2, &value2),
      "query metric value after increment");
  ASSERT_EQ(value2, 0);

  // Increment metric 1, check metric 2 == metric 1
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricIncrement(metric1, inc), "increase metric value");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric1, &value1),
      "query metric value after increment");
  // Verify increment worked
  ASSERT_EQ(value1, inc);

  FAIL_TEST_IF_ERR(
      TRITONSERVER_MetricValue(metric2, &value2),
      "query metric value after increment");
  // Verify metric values are equal
  ASSERT_EQ(value1, value2);
  std::cout << "metric1 value: " << value1 << " == metric2 value: " << value2
            << std::endl;
}

// Test duplicate metric objects without labels
TEST_F(MetricsApiTest, TestDupeMetricNoLabels)
{
  // TODO: same as above with empty labels
  ASSERT_EQ(true, false);
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
