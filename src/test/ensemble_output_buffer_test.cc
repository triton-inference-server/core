// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
// Covers CheckAndSetEnsembleOutput's output buffer sizing
// (ensemble_scheduler.cc).
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
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

// Lets a test assert on the allocation request itself, not just the
// copied bytes.
size_t g_last_alloc_byte_size = 0;

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
  g_last_alloc_byte_size = byte_size;

  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
  } else {
    void* allocated_ptr = malloc(byte_size);
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
  free(buffer);
  delete reinterpret_cast<std::string*>(buffer_userp);
  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  FAIL_TEST_IF_ERR(
      TRITONSERVER_InferenceRequestDelete(request),
      "deleting inference request");
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    auto p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
  }
}

// Non-repeating fill so a truncated or misaligned copy shows up as a
// mismatch.
std::vector<char>
FillPattern(size_t total_bytes)
{
  std::vector<char> data(total_bytes);
  for (size_t i = 0; i < total_bytes; ++i) {
    data[i] = static_cast<char>((i * 37 + 11) % 256);
  }
  return data;
}

class EnsembleOutputBufferTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
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
  }

  // "fire" declares "T" as both input and output with no step touching
  // it, so the output is filled straight from the request (not a
  // backend) -- keeping however many buffers 'segments' split it into.
  void RunSplitInput(
      const std::vector<char>& input_data, const std::vector<size_t>& segments,
      TRITONSERVER_InferenceResponse** response)
  {
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestNew(
            &irequest, server_, "fire", -1 /* model_version */),
        "creating inference request");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestSetReleaseCallback(
            irequest, InferRequestComplete, nullptr),
        "setting request release callback");

    std::vector<int64_t> shape{static_cast<int64_t>(input_data.size())};
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAddInput(
            irequest, "T", TRITONSERVER_TYPE_UINT8, shape.data(), shape.size()),
        "setting input for the request");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, "T"),
        "requesting output");

    size_t offset = 0;
    for (size_t segment_size : segments) {
      FAIL_TEST_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, "T", input_data.data() + offset, segment_size,
              TRITONSERVER_MEMORY_CPU, 0),
          "assigning INPUT data");
      offset += segment_size;
    }

    std::promise<TRITONSERVER_InferenceResponse*> p;
    std::future<TRITONSERVER_InferenceResponse*> future = p.get_future();
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator_, nullptr, InferResponseComplete,
            reinterpret_cast<void*>(&p)),
        "setting response callback");

    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerInferAsync(server_, irequest, nullptr),
        "running inference");

    *response = future.get();
  }

  // Every out-parameter here is written unconditionally, so none can be
  // nullptr.
  void GetOutput0(
      TRITONSERVER_InferenceResponse* response, const void** base,
      size_t* byte_size)
  {
    const char* name;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceResponseOutput(
            response, 0, &name, &datatype, &shape, &dim_count, base, byte_size,
            &memory_type, &memory_type_id, &userp),
        "getting output data");
  }

  void VerifySegmentedOutput(const std::vector<size_t>& segments)
  {
    std::string segments_label;
    for (size_t seg : segments) {
      segments_label += std::to_string(seg) + " ";
    }
    SCOPED_TRACE("segments=" + segments_label);

    size_t total = 0;
    for (size_t seg : segments) {
      total += seg;
    }
    const std::vector<char> input_data = FillPattern(total);

    g_last_alloc_byte_size = 0;
    TRITONSERVER_InferenceResponse* response = nullptr;
    RunSplitInput(input_data, segments, &response);
    ASSERT_NE(response, nullptr) << "no response received";
    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceResponseError(response), "response status");

    // The bug: allocation sized from one segment, not the total.
    ASSERT_EQ(g_last_alloc_byte_size, total);

    const void* base = nullptr;
    size_t byte_size = 0;
    GetOutput0(response, &base, &byte_size);
    ASSERT_EQ(byte_size, total);
    ASSERT_EQ(0, std::memcmp(base, input_data.data(), total));

    FAIL_TEST_IF_ERR(
        TRITONSERVER_InferenceResponseDelete(response),
        "deleting inference response");
  }

  static TRITONSERVER_Server* server_;
  TRITONSERVER_ResponseAllocator* allocator_ = nullptr;
};

TRITONSERVER_Server* EnsembleOutputBufferTest::server_ = nullptr;

// The exact shape from the report: 976 + 72 bytes.
TEST_F(EnsembleOutputBufferTest, ReportedShape)
{
  VerifySegmentedOutput({976, 72});
}

// A different pair of sizes, so the fix isn't just tuned to one case.
TEST_F(EnsembleOutputBufferTest, DifferentTwoSegmentShape)
{
  VerifySegmentedOutput({300, 1700});
}

// More than two segments.
TEST_F(EnsembleOutputBufferTest, ThreeSegmentShape)
{
  VerifySegmentedOutput({333, 333, 334});
}

// Single segment: the no-op case the fix must leave unchanged.
TEST_F(EnsembleOutputBufferTest, SingleSegmentUnaffected)
{
  VerifySegmentedOutput({500});
}

}  // namespace
