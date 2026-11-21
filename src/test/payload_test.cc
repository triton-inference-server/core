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

#include "payload.h"

#include <memory>

#include "gtest/gtest.h"

namespace triton { namespace core { namespace {

// The completion promise backing Payload::Wait() is only allocated for
// control payloads (INIT/WARM_UP/EXIT). INFER_RUN payloads — the hot path —
// carry no promise, and Wait() on them must fail fast instead of blocking
// forever on a never-fulfilled future.
TEST(PayloadTest, WaitOnInferRunPayloadFailsFast)
{
  Payload payload;
  payload.Reset(Payload::Operation::INFER_RUN, nullptr /* instance */);

  Status status = payload.Wait();
  EXPECT_FALSE(status.IsOk())
      << "Wait() on an INFER_RUN payload must not report success";
}

// Control payloads keep the full promise cycle: Execute() publishes the
// status and Wait() retrieves it. EXIT is the one operation whose Execute()
// does not touch the model instance, so the cycle is testable standalone.
TEST(PayloadTest, ExitPayloadCompletesWait)
{
  Payload payload;
  payload.Reset(Payload::Operation::EXIT, nullptr /* instance */);

  bool should_exit = false;
  payload.Execute(&should_exit);
  EXPECT_TRUE(should_exit);

  Status status = payload.Wait();
  EXPECT_TRUE(status.IsOk());
}

// Payloads are recycled through the rate limiter's bucket via Reset(); the
// lazy promise must appear and disappear correctly across op-type changes on
// the same object.
TEST(PayloadTest, ResetCyclesTogglePromise)
{
  Payload payload;

  payload.Reset(Payload::Operation::INFER_RUN, nullptr);
  EXPECT_FALSE(payload.Wait().IsOk());

  payload.Reset(Payload::Operation::EXIT, nullptr);
  bool should_exit = false;
  payload.Execute(&should_exit);
  EXPECT_TRUE(payload.Wait().IsOk());

  payload.Reset(Payload::Operation::INFER_RUN, nullptr);
  EXPECT_FALSE(payload.Wait().IsOk());
}

}}}  // namespace triton::core
