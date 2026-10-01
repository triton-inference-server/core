// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "dynamic_batch_scheduler.h"
#include "gtest/gtest.h"
#include "server.h"

extern "C" bool SchedulerTestWaitForBatchIncludes(
    size_t count, uint32_t timeout_ms);

namespace {

using namespace triton::core;
using namespace std::chrono_literals;

void
Require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void
Require(const Status& status)
{
  Require(status.IsOk(), status.Message());
}

void
RunRegression(const std::string& mode)
{
  const std::filesystem::path models(SCHEDULER_TEST_MODEL_REPOSITORY);
  InferenceServer server;
  server.SetModelRepositoryPaths({(models / "empty").string()});
  server.SetModelControlMode(ModelControlMode::MODE_EXPLICIT);
  server.SetPinnedMemoryPoolByteSize(0);
  server.SetRateLimiterMode(RateLimitMode::RL_OFF);
  server.SetEnablePeerAccess(false);
  server.SetResponseCacheEnabled(false);
  server.SetRepoAgentDir((models / "empty").string());
  server.SetCacheDir((models / "empty").string());
  const triton::common::BackendCmdlineConfigMap backend_config{
      {"",
       {{"backend-directory", models.string()},
        {"auto-complete-config", "false"},
        {"min-compute-capability", "0"}}}};
  server.SetBackendCmdlineConfig(backend_config);
  Require(server.Init());

  inference::ModelConfig config;
  config.set_name("scheduler_test");
  config.set_backend("scheduler_test");
  config.set_max_batch_size(8);
  config.mutable_version_policy()->mutable_latest()->set_num_versions(1);
  auto* input = config.add_input();
  input->set_name("INPUT");
  input->set_data_type(inference::TYPE_FP32);
  input->add_dims(1);
  auto* group = config.add_instance_group();
  group->set_kind(inference::ModelInstanceGroup::KIND_CPU);
  group->set_count(1);
  group->set_passive(true);
  if (mode == "accounting") {
    (*config.mutable_parameters())["TRITON_BATCH_STRATEGY_PATH"]
        .set_string_value(
            (models / "scheduler_test/libtriton_scheduler_test.so").string());
  }
  std::unique_ptr<TritonModel> model;
  Require(TritonModel::Create(
      &server, (models / "scheduler_test").string(), backend_config, {},
      ModelIdentifier("", "scheduler_test"), 1, config, true, &model));
  auto create = [&](const std::string& name) {
    std::shared_ptr<TritonModelInstance> instance;
    Require(TritonModelInstance::CreateInstance(
        model.get(), name, TritonModelInstance::Signature(*group, 0),
        TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {}, true, "", {}, {},
        &instance));
    return instance;
  };
  auto primary = create("primary");
  auto limiter = server.GetRateLimiter();
  Require(limiter->RegisterModelInstance(primary.get(), {}));

  if (mode == "accounting") {
    const std::array<float, 8> data{};
    inference::ModelDynamicBatching batching;
    batching.add_preferred_batch_size(8);
    batching.set_max_queue_delay_microseconds(5 * 1000 * 1000);
    batching.mutable_default_queue_policy()->set_max_queue_size(2);
    std::unique_ptr<Scheduler> scheduler;
    Require(DynamicBatchScheduler::Create(
        model.get(), nullptr, 0, true, 8, {}, batching, &scheduler));
    auto request = [&](int64_t batch_size) {
      auto result = std::make_unique<InferenceRequest>(model.get(), 1);
      const std::vector<int64_t> shape{batch_size, 1};
      Require(result->AddOriginalInput("INPUT", inference::TYPE_FP32, shape));
      InferenceRequest::Input* request_input;
      Require(result->MutableOriginalInput("INPUT", &request_input));
      Require(request_input->AppendData(
          data.data(), batch_size * sizeof(float), TRITONSERVER_MEMORY_CPU, 0));
      Require(result->SetResponseCallback(nullptr, nullptr, nullptr, nullptr));
      Require(result->PrepareForInference());
      return result;
    };
    auto consume = [&]() {
      return std::async(std::launch::async, [&]() {
        std::deque<TritonModelInstance*> instances{primary.get()};
        std::shared_ptr<Payload> payload;
        limiter->DequeuePayload(instances, &payload);
        return payload;
      });
    };

    auto accepted = request(8);
    Require(scheduler->Enqueue(accepted));
    Require(
        accepted == nullptr, "Successful admission must transfer ownership");
    auto pending = request(1);
    Require(scheduler->Enqueue(pending));
    for (int i = 0; i < 7; ++i) {
      auto rejected = request(1);
      const auto status = scheduler->Enqueue(rejected);
      Require(!status.IsOk(), "A full request queue must reject admission");
      Require(rejected != nullptr, "Rejected admission must retain ownership");
    }
    EXPECT_EQ(scheduler->InflightInferenceCount(), 2);

    auto consumer = consume();
    Require(
        consumer.wait_for(10s) == std::future_status::ready,
        "The admitted batch must drain");
    auto payload = consumer.get();
    Require(
        payload != nullptr && payload->BatchSize() == 8,
        "The first payload must contain the admitted batch");
    limiter->PayloadRelease(payload);

    auto final_consumer = consume();
    Require(
        SchedulerTestWaitForBatchIncludes(2, 5000),
        "The batcher must examine the pending request");
    auto later = request(1);
    Require(scheduler->Enqueue(later));
    EXPECT_EQ(scheduler->InflightInferenceCount(), 2);

    // Below preferred size eight, admission should leave the batcher waiting.
    // Restoring rejected-request credits makes this enqueue wake it instead.
    EXPECT_FALSE(SchedulerTestWaitForBatchIncludes(3, 200))
        << "Rejected admissions caused a wake-up below the preferred batch "
           "size";

    // Drain both remaining requests before joining the scheduler thread.
    Require(
        final_consumer.wait_for(10s) == std::future_status::ready,
        "The final requests must drain before scheduler teardown");
    payload = final_consumer.get();
    EXPECT_EQ(payload->BatchSize(), 2);
    EXPECT_EQ(payload->RequestCount(), 2);
    limiter->PayloadRelease(payload);
  } else if ((mode == "instance-removal") || (mode == "model-removal")) {
    std::promise<void> started;
    auto waiter = std::async(std::launch::async, [&]() {
      started.set_value();
      return limiter->PayloadSlotAvailable(
          model.get(), mode == "instance-removal" ? primary.get() : nullptr,
          false);
    });
    started.get_future().wait();
    Require(
        waiter.wait_for(50ms) == std::future_status::timeout,
        "Slot check must block before removal");
    if (mode == "instance-removal") {
      limiter->UnregisterModelInstance(primary.get());
    } else {
      limiter->UnregisterModel(model.get());
    }
    Require(
        waiter.wait_for(5s) == std::future_status::ready,
        "Removal must release the blocked slot check");
    EXPECT_FALSE(waiter.get());
    EXPECT_FALSE(
        limiter->PayloadSlotAvailable(model.get(), primary.get(), false, true));
    EXPECT_FALSE(
        limiter->PayloadSlotAvailable(model.get(), primary.get(), false));
  } else {
    auto extra = create("extra");
    auto extra_two = create("extra_two");
    std::atomic<bool> done{false};
    std::promise<void> started;
    std::thread reader([&]() {
      bool first = true;
      while (!done.load()) {
        if (mode == "capacity") {
          limiter->PayloadSlotAvailable(model.get(), nullptr, true, true);
        } else {
          limiter->PayloadSlotAvailable(
              model.get(), primary.get(), false, true);
        }
        if (first) {
          started.set_value();
          first = false;
        }
      }
    });
    started.get_future().wait();
    for (int i = 0; i < 2000; ++i) {
      Require(limiter->RegisterModelInstance(extra.get(), {}));
      Require(limiter->RegisterModelInstance(extra_two.get(), {}));
      limiter->UnregisterModelInstance(extra.get());
      limiter->UnregisterModelInstance(extra_two.get());
    }
    done.store(true);
    reader.join();
    EXPECT_TRUE(
        limiter->PayloadSlotAvailable(model.get(), nullptr, true, true));
    EXPECT_FALSE(
        limiter->PayloadSlotAvailable(model.get(), extra.get(), false, true));
  }
}

TEST(SchedulerRegression, AccountingAfterRejections)
{
  RunRegression("accounting");
}

TEST(SchedulerRegression, BlockingWaitDuringInstanceRemoval)
{
  RunRegression("instance-removal");
}

TEST(SchedulerRegression, BlockingWaitDuringModelRemoval)
{
  RunRegression("model-removal");
}

TEST(SchedulerRegression, CapacityDuringInstanceChanges)
{
  RunRegression("capacity");
}

TEST(SchedulerRegression, LookupDuringInstanceChanges)
{
  RunRegression("lookup");
}

}  // namespace
