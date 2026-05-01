// SPDX-FileCopyrightText: Copyright (c) 2018-2020 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>

#include "infer_request.h"
#include "status.h"

namespace triton { namespace core {

// Scheduler interface.
class Scheduler {
 public:
  virtual ~Scheduler() {}

  // The prototype for the initialization function that will be called
  // by the "standard" schedulers created based on a model's
  // scheduling_choice settings. The init function is called once by
  // the runner that will later execute requests for 'runner_idx'. A
  // non-OK error status indicates an initialization error that
  // prevents scheduler from using the runner.
  using StandardInitFunc = std::function<Status(uint32_t runner_idx)>;

  // The prototype for the warmup function that will be called by the
  // "standard" schedulers created based on a model's
  // scheduling_choice settings. The warmup function is called once by
  // the runner that will later execute requests for 'runner_idx'. A
  // non-OK error status indicates an error that prevents scheduler
  // from sending warmup requests to the runner.
  using StandardWarmupFunc = std::function<Status(uint32_t runner_idx)>;

  // The prototype for the run function that will be called by the
  // "standard" schedulers created based on a model's
  // scheduling_choice settings. The run function must accept a
  // 'runner_idx' indicating which runner should execute the
  // 'requests'. Ownership of the 'requests' is transferred to the
  // runner which is responsible for generating responses and
  // releasing the requests.
  using StandardRunFunc = std::function<void(
      uint32_t runner_idx,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests)>;

  // Enqueue a request with the scheduler. If Status::Success is returned
  // then the backend has taken ownership of the request object and so
  // 'request' will be nullptr. If non-success is returned then the
  // caller still retains ownership of 'request'.
  virtual Status Enqueue(std::unique_ptr<InferenceRequest>& request) = 0;

  // Return the number of in-flight inferences tracked by the scheduler.
  virtual size_t InflightInferenceCount() = 0;

  // Instruct the scheduler to stop processing future requests unless they are
  // considered as in-flight.
  virtual void Stop() = 0;
};

}}  // namespace triton::core
