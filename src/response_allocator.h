// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tritonserver_apis.h"

namespace triton { namespace core {

//
// Implementation for TRITONSERVER_ResponseAllocator.
//
class ResponseAllocator {
 public:
  explicit ResponseAllocator(
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
      TRITONSERVER_ResponseAllocatorStartFn_t start_fn)
      : alloc_fn_(alloc_fn), buffer_attributes_fn_(nullptr), query_fn_(nullptr),
        release_fn_(release_fn), start_fn_(start_fn)
  {
  }

  void SetQueryFunction(TRITONSERVER_ResponseAllocatorQueryFn_t query_fn)
  {
    query_fn_ = query_fn;
  }

  void SetBufferAttributesFunction(
      TRITONSERVER_ResponseAllocatorBufferAttributesFn_t buffer_attributes_fn)
  {
    buffer_attributes_fn_ = buffer_attributes_fn;
  }

  TRITONSERVER_ResponseAllocatorAllocFn_t AllocFn() const { return alloc_fn_; }
  TRITONSERVER_ResponseAllocatorBufferAttributesFn_t BufferAttributesFn() const
  {
    return buffer_attributes_fn_;
  }
  TRITONSERVER_ResponseAllocatorQueryFn_t QueryFn() const { return query_fn_; }
  TRITONSERVER_ResponseAllocatorReleaseFn_t ReleaseFn() const
  {
    return release_fn_;
  }
  TRITONSERVER_ResponseAllocatorStartFn_t StartFn() const { return start_fn_; }

 private:
  TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  TRITONSERVER_ResponseAllocatorBufferAttributesFn_t buffer_attributes_fn_;
  TRITONSERVER_ResponseAllocatorQueryFn_t query_fn_;
  TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn_;
  TRITONSERVER_ResponseAllocatorStartFn_t start_fn_;
};

}}  // namespace triton::core
