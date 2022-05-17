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
#pragma once

#ifdef TRITON_ENABLE_METRICS

#include <mutex>
#include <unordered_map>

#include "infer_parameter.h"
#include "prometheus/registry.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

//
// Implementation for TRITONSERVER_MetricFamily.
//
class MetricFamily {
 public:
  MetricFamily(
      TRITONSERVER_MetricKind kind, const char* name, const char* description);
  ~MetricFamily();

  void* Family() const { return family_; }
  TRITONSERVER_MetricKind Kind() const { return kind_; }

  void* Add(std::map<std::string, std::string> label_map);
  void Remove(void* metric);

 private:
  void* family_;
  std::mutex mtx_;
  // prometheus returns the existing metric pointer if the metric with the same
  // set of labels are requested, as a result, different Metric objects may
  // refer to the same prometheus metric. So we must track the reference count
  // of the metric and request prometheus to remove it only when all references
  // are released.
  std::unordered_map<void*, size_t> metric_ref_cnt_;
  TRITONSERVER_MetricKind kind_;
};

//
// Implementation for TRITONSERVER_Metric.
//
class Metric {
 public:
  Metric(
      TRITONSERVER_MetricFamily* family,
      std::vector<const InferenceParameter*> labels);
  ~Metric();

  MetricFamily* Family() const { return family_; }
  TRITONSERVER_MetricKind Kind() const { return kind_; }

  TRITONSERVER_Error* Value(double* value);
  TRITONSERVER_Error* Increment(double value);
  TRITONSERVER_Error* Set(double value);

 private:
  void* metric_;
  MetricFamily* family_;
  TRITONSERVER_MetricKind kind_;
};

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METRICS
