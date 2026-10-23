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
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "triton/core/tritonserver.h"

using namespace std::chrono_literals;
constexpr size_t kCheckpointHeaderSize = 88;
void
Check(TRITONSERVER_Error* e)
{
  if (e) {
    std::string s = TRITONSERVER_ErrorMessage(e);
    TRITONSERVER_ErrorDelete(e);
    throw std::runtime_error(s);
  }
}
void
Require(bool yes, const std::string& s)
{
  if (!yes)
    throw std::runtime_error(s);
}
struct Result {
  std::string error;
  std::vector<char> bytes;
};
struct Work {
  std::vector<char> input;
  std::promise<Result> response;
  std::promise<void> released;
  bool retain = false;
  TRITONSERVER_InferenceRequest* retained = nullptr;
  ~Work()
  {
    if (retained)
      TRITONSERVER_InferenceRequestDelete(retained);
  }
};
struct Pending {
  std::shared_ptr<Work> work;
  std::future<Result> response;
  std::future<void> released;
  Result Get()
  {
    Require(
        response.wait_for(15s) == std::future_status::ready,
        "response timeout");
    auto r = response.get();
    Require(
        released.wait_for(15s) == std::future_status::ready, "release timeout");
    released.get();
    return r;
  }
};
TRITONSERVER_Error*
Allocate(
    TRITONSERVER_ResponseAllocator*, const char*, size_t n,
    TRITONSERVER_MemoryType, int64_t, void*, void** p, void** user,
    TRITONSERVER_MemoryType* type, int64_t* device)
{
  *p = n ? malloc(n) : nullptr;
  *user = nullptr;
  *type = TRITONSERVER_MEMORY_CPU;
  *device = 0;
  return n && !*p ? TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "malloc")
                  : nullptr;
}
TRITONSERVER_Error*
Free(
    TRITONSERVER_ResponseAllocator*, void* p, void*, size_t,
    TRITONSERVER_MemoryType, int64_t)
{
  free(p);
  return nullptr;
}
void
Released(TRITONSERVER_InferenceRequest* request, uint32_t, void* user)
{
  auto* w = static_cast<Work*>(user);
  if (w->retain)
    w->retained = request;
  else
    TRITONSERVER_InferenceRequestDelete(request);
  w->released.set_value();
}
void
Responded(TRITONSERVER_InferenceResponse* response, uint32_t, void* user)
{
  auto* w = static_cast<Work*>(user);
  Result r;
  if (!response)
    r.error = "null response";
  else {
    auto* e = TRITONSERVER_InferenceResponseError(response);
    if (e)
      r.error = TRITONSERVER_ErrorMessage(e);
    else {
      const char* name;
      TRITONSERVER_DataType type;
      const int64_t* shape;
      uint64_t dims;
      const void* base;
      size_t n;
      TRITONSERVER_MemoryType mem;
      int64_t device;
      void* u;
      e = TRITONSERVER_InferenceResponseOutput(
          response, 0, &name, &type, &shape, &dims, &base, &n, &mem, &device,
          &u);
      if (e) {
        r.error = TRITONSERVER_ErrorMessage(e);
        TRITONSERVER_ErrorDelete(e);
      } else if (mem != TRITONSERVER_MEMORY_CPU)
        r.error = "non-CPU output";
      else
        r.bytes.assign(
            static_cast<const char*>(base), static_cast<const char*>(base) + n);
    }
    TRITONSERVER_InferenceResponseDelete(response);
  }
  w->response.set_value(std::move(r));
}
struct Stats {
  uint64_t slots = 0, bytes = 0;
  std::string error;
};
struct Server {
  TRITONSERVER_Server* server = nullptr;
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  Server(const std::string& repo, const std::string& backend)
  {
    TRITONSERVER_ServerOptions* opts;
    Check(TRITONSERVER_ServerOptionsNew(&opts));
    Check(TRITONSERVER_ServerOptionsSetModelRepositoryPath(opts, repo.c_str()));
    Check(TRITONSERVER_ServerOptionsSetBackendDirectory(opts, backend.c_str()));
    Check(TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(opts, 0));
    Check(TRITONSERVER_ServerOptionsSetLogInfo(opts, false));
    Check(TRITONSERVER_ServerOptionsSetLogWarn(opts, false));
    Check(TRITONSERVER_ServerOptionsSetExitTimeout(opts, 5));
    Check(TRITONSERVER_ServerNew(&server, opts));
    Check(TRITONSERVER_ServerOptionsDelete(opts));
    Check(
        TRITONSERVER_ResponseAllocatorNew(&allocator, Allocate, Free, nullptr));
  }
  ~Server()
  {
    if (server) {
      TRITONSERVER_ServerStop(server);
      TRITONSERVER_ServerDelete(server);
    }
    if (allocator)
      TRITONSERVER_ResponseAllocatorDelete(allocator);
  }
  Pending Submit(
      const std::string& model, uint64_t id, const std::vector<char>& input,
      bool start = false, bool end = false, bool fp = false,
      bool retain = false)
  {
    auto w = std::make_shared<Work>();
    w->input = input;
    w->retain = retain;
    Pending p{w, w->response.get_future(), w->released.get_future()};
    TRITONSERVER_InferenceRequest* req;
    Check(TRITONSERVER_InferenceRequestNew(&req, server, model.c_str(), 1));
    Check(TRITONSERVER_InferenceRequestSetCorrelationId(req, id));
    Check(TRITONSERVER_InferenceRequestSetFlags(
        req, (start ? TRITONSERVER_REQUEST_FLAG_SEQUENCE_START : 0) |
                 (end ? TRITONSERVER_REQUEST_FLAG_SEQUENCE_END : 0)));
    int64_t shape[2] = {1, static_cast<int64_t>(input.size() / 4)};
    Check(TRITONSERVER_InferenceRequestAddInput(
        req, "X", fp ? TRITONSERVER_TYPE_FP32 : TRITONSERVER_TYPE_INT32, shape,
        2));
    Check(TRITONSERVER_InferenceRequestAppendInputData(
        req, "X", w->input.data(), input.size(), TRITONSERVER_MEMORY_CPU, 0));
    Check(TRITONSERVER_InferenceRequestAddRequestedOutput(req, "Y"));
    Check(TRITONSERVER_InferenceRequestSetReleaseCallback(
        req, Released, w.get()));
    Check(TRITONSERVER_InferenceRequestSetResponseCallback(
        req, allocator, nullptr, Responded, w.get()));
    auto* e = TRITONSERVER_ServerInferAsync(server, req, nullptr);
    if (e) {
      Result r{TRITONSERVER_ErrorMessage(e), {}};
      TRITONSERVER_ErrorDelete(e);
      TRITONSERVER_InferenceRequestDelete(req);
      w->response.set_value(r);
      w->released.set_value();
    }
    return p;
  }
  Stats Control(
      const std::string& model, int op, uint64_t id = 0,
      const std::string& path = "")
  {
    Stats s;
    auto* e = TRITONSERVER_ServerSequenceStateCheckpoint(
        server, model.c_str(), 1,
        static_cast<TRITONSERVER_SequenceStateCheckpointOperation>(op), id,
        path.c_str(), &s.slots, &s.bytes);
    if (e) {
      s.error = TRITONSERVER_ErrorMessage(e);
      TRITONSERVER_ErrorDelete(e);
    }
    return s;
  }
};
template <class T>
std::vector<char>
Bytes(const std::vector<T>& x)
{
  const auto* p = reinterpret_cast<const char*>(x.data());
  return {p, p + x.size() * sizeof(T)};
}
int32_t
Integer(Result r)
{
  Require(r.error.empty(), r.error);
  Require(r.bytes.size() == 4, "integer output size");
  int32_t x;
  memcpy(&x, r.bytes.data(), 4);
  return x;
}
int32_t
Step(
    Server& s, const std::string& m, uint64_t id, int x, bool start = false,
    bool end = false)
{
  return Integer(s.Submit(m, id, Bytes<int32_t>({x}), start, end).Get());
}
void
Good(const Stats& s)
{
  Require(s.error.empty(), s.error);
}
void
Pass(const std::string& name)
{
  std::cout << "{\"test\":\"" << name << "\",\"status\":\"pass\"}" << std::endl;
}

void
SlotDemo(Server& s, const std::filesystem::path& dir)
{
  const std::string m = "acc";
  Require(Step(s, m, 1, 7, true) == 7, "start A");
  Require(Step(s, m, 1, 4) == 11, "A second");
  Require(Step(s, m, 2, 100, true) == 100, "start B");
  auto c = s.Submit(m, 3, Bytes<int32_t>({200}), true);
  Require(
      c.response.wait_for(150ms) == std::future_status::timeout,
      "C must wait behind two occupied slots");
  auto before = s.Control(m, 3);
  Good(before);
  Require(before.slots == 2 && before.bytes == 16, "two double-buffer states");
  auto saved = s.Control(m, 0, 1, (dir / "A.state").string());
  Good(saved);
  Require(
      saved.slots == 2 && saved.bytes == 8,
      "A buffers freed before C initialization");
  Require(
      !s.Control(m, 0, 1, (dir / "duplicate.state").string()).error.empty(),
      "duplicate suspend must fail");
  Require(Integer(c.Get()) == 200, "C released");
  Require(
      !s.Control(m, 1, 1, (dir / "A.state").string()).error.empty(),
      "resume full must fail");
  Step(s, m, 3, 0, false, true);
  auto resumed = s.Control(m, 1, 1, (dir / "A.state").string());
  Good(resumed);
  Require(
      !s.Control(m, 1, 1, (dir / "A.state").string()).error.empty(),
      "duplicate resume must fail");
  auto bad = s.Submit(m, 1, Bytes<int32_t>({0}), true).Get();
  Require(!bad.error.empty(), "START after restore must reject");
  Require(Step(s, m, 1, 5) == 16, "resume lost state");
  Require(Step(s, m, 2, 1) == 101, "B isolated");
  Step(s, m, 1, 0, false, true);
  Step(s, m, 2, 0, false, true);
  auto empty = s.Control(m, 3);
  Good(empty);
  Require(
      empty.slots == 0 && empty.bytes == 0, "END must clear all state buffers");
  Good(s.Control(m, 2, 1, (dir / "A.state").string()));
  Require(!std::filesystem::exists(dir / "A.state"), "discard storage");
  Require(Step(s, m, 1, 3, true, true) == 3, "sequence ID reuse after discard");
  Pass("slot_release_resume_isolation_and_discard");
}

void
Cycles(Server& s, const std::filesystem::path& dir, const std::string& m)
{
  // Execute the same sequence uninterrupted through the same real backend.
  Step(s, m, 11, 1, true);
  std::vector<int32_t> reference;
  for (int i = 0; i < 20; ++i) reference.push_back(Step(s, m, 11, i));
  const auto final_reference = Step(s, m, 11, 3, false, true);
  int expected = Step(s, m, 10, 1, true);
  for (int i = 0; i < 20; ++i) {
    expected += i;
    const auto actual = Step(s, m, 10, i);
    Require(
        actual == expected && actual == reference[i], "cycle vs uninterrupted");
    auto path = (dir / (m + std::to_string(i) + ".state")).string();
    auto saved = s.Control(m, 0, 10, path);
    Good(saved);
    Require(
        saved.slots == 0 && saved.bytes == 0, "suspend frees slot and buffers");
    Require(
        std::filesystem::file_size(path) ==
            kCheckpointHeaderSize + sizeof(int32_t),
        "accumulator checkpoint size");
    auto restored = s.Control(m, 1, 10, path);
    Good(restored);
    Good(s.Control(m, 2, 10, path));
  }
  Require(
      Step(s, m, 10, 3, false, true) == final_reference,
      "repeated exact comparison");
  Pass(m + "_20_cycles");
}

void
Failures(Server& s, const std::filesystem::path& dir)
{
  {
    auto retained =
        s.Submit("acc", 19, Bytes<int32_t>({10}), true, false, false, true);
    Require(Integer(retained.Get()) == 10, "retained request inference");
    Require(
        !s.Control("acc", 0, 19, (dir / "retained.state").string())
             .error.empty(),
        "suspend must reject externally retained state");
    Require(
        Step(s, "acc", 19, 1) == 11,
        "retained request rejection damaged state");
  }
  Good(s.Control("acc", 0, 19, (dir / "retained.state").string()));
  Good(s.Control("acc", 1, 19, (dir / "retained.state").string()));
  Require(
      Step(s, "acc", 19, 1, false, true) == 12,
      "retry after dropping retained request");
  Good(s.Control("acc", 2, 19, (dir / "retained.state").string()));
  Pass("retained_request_ownership_rejection_and_retry");
  Step(s, "acc", 20, 42, true);
  Require(
      !s.Control("acc", 0, 20, (dir / "missing/x.state").string())
           .error.empty(),
      "failed persistence expected");
  Require(Step(s, "acc", 20, 1) == 43, "failed persistence lost active state");
  // Fail after the header and two state bytes.
  struct rlimit old_limit {
  }, small_limit{};
  Require(getrlimit(RLIMIT_FSIZE, &old_limit) == 0, "getrlimit");
  small_limit = old_limit;
  small_limit.rlim_cur = kCheckpointHeaderSize + 2;
  auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
  Require(setrlimit(RLIMIT_FSIZE, &small_limit) == 0, "setrlimit");
  auto write_failure =
      s.Control("acc", 0, 20, (dir / "partial.state").string());
  Require(setrlimit(RLIMIT_FSIZE, &old_limit) == 0, "restore file limit");
  std::signal(SIGXFSZ, old_signal);
  Require(
      !write_failure.error.empty() &&
          !std::filesystem::exists(dir / "partial.state"),
      "partial write must fail without publishing");
  Require(Step(s, "acc", 20, 0) == 43, "partial write lost live state");
  auto path = dir / "failure.state";
  Good(s.Control("acc", 0, 20, path.string()));
  std::ifstream in(path, std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(in)), {});
  for (const std::string kind : {"truncated", "corrupted", "oversized"}) {
    auto bad = dir / (kind + ".state");
    std::string copy = data;
    if (kind == "truncated")
      copy.resize(30);
    else if (kind == "corrupted")
      copy.back() ^= 1;
    else
      copy.push_back(0);
    std::ofstream(bad, std::ios::binary).write(copy.data(), copy.size());
    Require(
        !s.Control("acc", 1, 20, bad.string()).error.empty(),
        "invalid checkpoint accepted");
    auto stats = s.Control("acc", 3);
    Good(stats);
    Require(
        stats.slots == 0 && stats.bytes == 0,
        "invalid resume allocated live state");
  }
  Require(
      !s.Control("acc_mismatch", 1, 20, path.string()).error.empty(),
      "model mismatch accepted");
  Require(
      !s.Control("acc", 1, 21, path.string()).error.empty(),
      "cross-session checkpoint accepted");
  Good(s.Control("acc", 1, 20, path.string()));
  Require(Step(s, "acc", 20, 7) == 50, "restore after invalid checkpoints");
  Require(
      !s.Control("acc", 0, 20, path.string()).error.empty(),
      "overwrite existing checkpoint accepted");
  Require(
      Step(s, "acc", 20, 1, false, true) == 51, "failed overwrite lost state");
  Good(s.Control("acc", 2, 20, path.string()));
  Pass("persistence_failure_corruption_truncation_size_model_ID_overwrite");
}

void
Race(Server& s, const std::filesystem::path& dir)
{
  Step(s, "acc", 30, 0, true);
  std::atomic<int> submitted{0};
  std::vector<Pending> pending;
  std::thread producer([&] {
    for (int i = 0; i < 500; ++i) {
      pending.push_back(s.Submit("acc", 30, Bytes<int32_t>({1})));
      submitted.fetch_add(1);
    }
  });
  while (submitted.load() < 20) std::this_thread::yield();
  auto path = (dir / "race.state").string();
  auto saved = s.Control("acc", 0, 30, path);
  producer.join();
  Good(saved);
  int accepted = 0, rejected = 0;
  for (auto& p : pending) {
    auto r = p.Get();
    if (r.error.empty())
      ++accepted;
    else
      ++rejected;
  }
  Require(
      accepted >= 20 && rejected > 0,
      "race must exercise accepted and rejected requests");
  auto suspended = s.Submit("acc", 30, Bytes<int32_t>({1})).Get();
  Require(!suspended.error.empty(), "suspended continuation accepted");
  Require(
      !s.Submit("acc", 30, Bytes<int32_t>({1}), true).Get().error.empty(),
      "suspended ID reuse accepted");
  Good(s.Control("acc", 1, 30, path));
  Require(
      Step(s, "acc", 30, 0, false, true) == accepted,
      "snapshot must include every admitted request exactly once");
  std::cout << "{\"event\":\"race_boundary\",\"accepted\":" << accepted
            << ",\"rejected\":" << rejected << "}" << std::endl;
  Good(s.Control("acc", 2, 30, path));
  Pass("concurrent_admission_drain_boundary");
}

void
Neural(Server& s, const std::filesystem::path& dir)
{
  double max_error = 0;
  for (int i = 0; i < 80; ++i) {
    std::vector<float> x = {
        float(std::sin(i * .3)), float(std::cos(i * .2)), float(i * .001),
        float(-.2)};
    auto input = Bytes(x);
    auto a = s.Submit("rnn", 100, input, i == 0, i == 79, true).Get();
    auto b = s.Submit("rnn", 101, input, i == 0, i == 79, true).Get();
    Require(
        a.error.empty() && b.error.empty(),
        "RNN inference error: " + a.error + b.error);
    Require(a.bytes.size() == 16 && b.bytes.size() == 16, "RNN shape");
    for (int j = 0; j < 4; ++j) {
      float av, bv;
      memcpy(&av, a.bytes.data() + 4 * j, 4);
      memcpy(&bv, b.bytes.data() + 4 * j, 4);
      double e = std::abs(av - bv);
      max_error = std::max(max_error, e);
      Require(e <= 1e-6 + 1e-6 * std::abs(av), "RNN equivalence");
    }
    if (i % 7 == 3 && i < 79) {
      auto path = (dir / ("rnn" + std::to_string(i) + ".state")).string();
      auto saved = s.Control("rnn", 0, 101, path);
      Good(saved);
      Require(
          saved.slots == 1 && saved.bytes == 32, "RNN independent allocation");
      Require(
          std::filesystem::file_size(path) ==
              kCheckpointHeaderSize + (4 * sizeof(float)),
          "RNN checkpoint size");
      Good(s.Control("rnn", 1, 101, path));
      Good(s.Control("rnn", 2, 101, path));
    }
  }
  std::cout
      << "{\"event\":\"neural_equivalence\",\"steps\":80,\"max_abs_error\":"
      << max_error << ",\"atol\":1e-6,\"rtol\":1e-6}" << std::endl;
  Pass("recurrent_neural_uninterrupted_vs_restored");
}

int
main(int argc, char** argv)
{
  try {
    Require(
        argc == 5,
        "usage: integration repository backend-directory work-directory mode");
    std::filesystem::path dir = argv[3];
    std::filesystem::create_directories(dir);
    const std::string mode = argv[4];
    Server s(argv[1], argv[2]);
    if (mode == "restart-save") {
      Step(s, "acc", 900, 123, true);
      Good(s.Control("acc", 0, 900, (dir / "restart.state").string()));
      Pass("restart_checkpoint_written");
    } else if (mode == "restart-load") {
      Good(s.Control("acc", 1, 900, (dir / "restart.state").string()));
      Require(Step(s, "acc", 900, 9, false, true) == 132, "restart restore");
      Good(s.Control("acc", 2, 900, (dir / "restart.state").string()));
      Pass("new_process_restart_restore");
    } else {
      SlotDemo(s, dir);
      Cycles(s, dir, "acc");
      Cycles(s, dir, "acc_single");
      Failures(s, dir);
      Race(s, dir);
      Neural(s, dir);
    }
    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << std::endl;
    return 1;
  }
}
