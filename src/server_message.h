// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

#include "status.h"

#define TRITONJSON_STATUSTYPE triton::core::Status
#define TRITONJSON_STATUSRETURN(M) \
  return triton::core::Status(triton::core::Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS triton::core::Status::Success
#include "triton/common/triton_json.h"

namespace triton { namespace core {

//
// Implementation for TRITONSERVER_Message.
//
class TritonServerMessage {
 public:
  TritonServerMessage(const triton::common::TritonJson::Value& msg)
  {
    json_buffer_.Clear();
    msg.Write(&json_buffer_);
    base_ = json_buffer_.Base();
    byte_size_ = json_buffer_.Size();
    from_json_ = true;
  }

  TritonServerMessage(std::string&& msg)
  {
    str_buffer_ = std::move(msg);
    base_ = str_buffer_.data();
    byte_size_ = str_buffer_.size();
    from_json_ = false;
  }

  TritonServerMessage(const TritonServerMessage& rhs)
  {
    from_json_ = rhs.from_json_;
    if (from_json_) {
      json_buffer_ = rhs.json_buffer_;
      base_ = json_buffer_.Base();
      byte_size_ = json_buffer_.Size();
    } else {
      str_buffer_ = rhs.str_buffer_;
      base_ = str_buffer_.data();
      byte_size_ = str_buffer_.size();
    }
  }

  void Serialize(const char** base, size_t* byte_size) const
  {
    *base = base_;
    *byte_size = byte_size_;
  }

 private:
  bool from_json_;
  triton::common::TritonJson::WriteBuffer json_buffer_;
  std::string str_buffer_;

  const char* base_;
  size_t byte_size_;
};

}}  // namespace triton::core
