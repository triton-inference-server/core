// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#ifdef TRITON_ENABLE_ENSEMBLE

#include <memory>

#include "metric_model_reporter.h"
#include "model.h"
#include "model_config.pb.h"
#include "model_config_utils.h"
#include "scheduler.h"
#include "scheduler_utils.h"
#include "status.h"
#include "triton/common/thread_pool.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace core {

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // TRITON_ENABLE_GPU

class InferenceServer;

struct EnsembleInfo {
  struct StepInfo {
    StepInfo(const ModelIdentifier& model_id, const int64_t model_version)
        : model_id_(model_id), model_version_(model_version)
    {
    }

    ModelIdentifier model_id_;
    int64_t model_version_;
    std::unordered_map<std::string, std::string> input_to_tensor_;
    std::unordered_map<std::string, std::string> output_to_tensor_;
  };

  std::string ensemble_name_;

  bool is_decoupled_;

  bool is_cache_enabled_;

  // the ensemble output (re)shape expected by the ensemble
  std::unordered_map<std::string, triton::common::DimsList>
      ensemble_output_shape_;

  // Inputs that is marked optional for the ensemble
  std::set<std::string> optional_inputs_;

  std::vector<StepInfo> steps_;

  // Only include a step if the ensemble tensor is used as input in that step
  std::unordered_map<std::string, std::set<size_t>> tensor_to_step_;

  // backward path, ensemble tensor to the step that provides its data
  std::unordered_map<std::string, size_t> tensor_to_prev_step_;
};

// Scheduler that implements ensemble scheduling.
class EnsembleScheduler : public Scheduler {
 public:
  // Create a scheduler to process ensemble requests and
  // to dispatch requests to models in ensemble internally.
  static Status Create(
      InferenceStatsAggregator* const stats_aggregator,
      InferenceServer* const server, const ModelIdentifier& model_id,
      const inference::ModelConfig& config,
      std::unique_ptr<Scheduler>* scheduler);

  ~EnsembleScheduler();

  // \see Scheduler::Enqueue()
  Status Enqueue(std::unique_ptr<InferenceRequest>& request) override;


  // \see Scheduler::InflightInferenceCount()
  size_t InflightInferenceCount() override { return inflight_count_; }

  // \see Scheduler::Stop()
  void Stop() override {}

  triton::common::ThreadPool* CallbackPool() const { return callback_pool_; }

 private:
  EnsembleScheduler(
      InferenceStatsAggregator* const stats_aggregator,
      InferenceServer* const server, const ModelIdentifier& model_id,
      const inference::ModelConfig& config);

  void CacheLookUp(
      std::unique_ptr<InferenceRequest>& request,
      std::unique_ptr<InferenceResponse>& cached_response);

  std::shared_ptr<MetricModelReporter> metric_reporter_;
  InferenceStatsAggregator* const stats_aggregator_;
  InferenceServer* const is_;

  // Ensemble information that is built from model config
  std::unique_ptr<EnsembleInfo> info_;

  // The stream used for data transfer.
  cudaStream_t stream_;

  std::atomic<size_t> inflight_count_;

  // Fixed-size thread pool to run callbacks at end of each ensemble step.
  // Managed by the server.
  triton::common::ThreadPool* callback_pool_;
};

}}  // namespace triton::core

#endif  // TRITON_ENABLE_ENSEMBLE
