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

#include <chrono>
#include <condition_variable>
#include <mutex>

#include "triton/core/tritonbackend.h"

namespace {
std::mutex batch_mu;
std::condition_variable batch_cv;
size_t batch_includes = 0;
}  // namespace

extern "C" bool
SchedulerTestWaitForBatchIncludes(size_t count, uint32_t timeout_ms)
{
  std::unique_lock<std::mutex> lock(batch_mu);
  return batch_cv.wait_for(
      lock, std::chrono::milliseconds(timeout_ms),
      [count]() { return batch_includes >= count; });
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherInitialize(
    TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model*)
{
  std::lock_guard<std::mutex> lock(batch_mu);
  batch_includes = 0;
  *batcher = nullptr;
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherFinalize(TRITONBACKEND_Batcher*)
{
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelBatchInitialize(const TRITONBACKEND_Batcher*, void** userp)
{
  *userp = nullptr;
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void*)
{
  return nullptr;
}

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelBatchIncludeRequest(
    TRITONBACKEND_Request*, void*, bool* should_include)
{
  {
    std::lock_guard<std::mutex> lock(batch_mu);
    ++batch_includes;
  }
  batch_cv.notify_all();
  *should_include = true;
  return nullptr;
}

// Passive instances let the tests drive the rate limiter without execution.
extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance*, TRITONBACKEND_Request**, const uint32_t)
{
  return nullptr;
}
