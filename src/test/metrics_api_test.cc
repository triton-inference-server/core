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

#ifdef TRITON_ENABLE_METRICS

#include "gtest/gtest.h"
#include "metric_family.h"
#include "triton/core/tritonserver.h"

namespace {

// Test Fixture
class MetricsApiTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test end-to-end flow of Generic Metrics API
TEST_F(MetricsApiTest, TestEndToEnd)
{
  // Test other API call
  auto param =  TRITONSERVER_ParameterNew("key", TRITONSERVER_PARAMETER_STRING, "value");

  /*
  // Create counter family
  TRITONSERVER_MetricFamily* cfamily;
  TRITONSERVER_MetricKind ckind = TRITONSERVER_METRIC_KIND_COUNTER;
  const char* cname = "backend_counter_example";
  const char* cdescription = "this is an example counter metric added by a backend.";
  TRITONSERVER_MetricFamilyNew(&cfamily, ckind, cname, cdescription);

  // Create counter metric
  TRITONSERVER_Metric* cmetric;
  TRITONSERVER_Parameter** labels = nullptr; 
  const int num_labels = 0;
  // TODO: Use real labels
  TRITONSERVER_MetricNew(&cmetric, cfamily, labels, num_labels);

  // Value of counter should be zero initially
  double value = -1;
  TRITONSERVER_MetricValue(cmetric, &value);
  ASSERT_EQ(value, 0.0);

  TRITONSERVER_MetricIncrement(cmetric, 100);
  ASSERT_EQ(value, 100.0);

  TRITONSERVER_MetricKind kind;
  TRITONSERVER_GetMetricKind(cmetric, &kind);
  ASSERT_EQ(kind, ckind);

  // Cleanup
  TRITONSERVER_MetricDelete(cmetric);
  TRITONSERVER_MetricFamilyDelete(cfamily);
  */
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif  // TRITON_ENABLE_METRICS
