// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iostream>
#include <string>

#include "tritonserver_apis.h"

namespace triton { namespace core {

//
// An inference parameter.
//
class InferenceParameter {
 public:
  InferenceParameter(const char* name, const char* value)
      : name_(name), type_(TRITONSERVER_PARAMETER_STRING), value_string_(value)
  {
    byte_size_ = value_string_.size();
  }

  InferenceParameter(const char* name, const int64_t value)
      : name_(name), type_(TRITONSERVER_PARAMETER_INT), value_int64_(value),
        byte_size_(sizeof(int64_t))
  {
  }

  InferenceParameter(const char* name, const bool value)
      : name_(name), type_(TRITONSERVER_PARAMETER_BOOL), value_bool_(value),
        byte_size_(sizeof(bool))
  {
  }

  InferenceParameter(const char* name, const double value)
      : name_(name), type_(TRITONSERVER_PARAMETER_DOUBLE), value_double_(value),
        byte_size_(sizeof(double))
  {
  }

  InferenceParameter(const char* name, const void* ptr, const uint64_t size)
      : name_(name), type_(TRITONSERVER_PARAMETER_BYTES), value_bytes_(ptr),
        byte_size_(size)
  {
  }

  // The name of the parameter.
  const std::string& Name() const { return name_; }

  // Data type of the parameter.
  TRITONSERVER_ParameterType Type() const { return type_; }

  // Return a pointer to the parameter, or a pointer to the data content
  // if type_ is TRITONSERVER_PARAMETER_BYTES. This returned pointer must be
  // cast correctly based on 'type_'.
  //   TRITONSERVER_PARAMETER_STRING -> const char*
  //   TRITONSERVER_PARAMETER_INT -> int64_t*
  //   TRITONSERVER_PARAMETER_BOOL -> bool*
  //   TRITONSERVER_PARAMETER_DOUBLE -> double*
  //   TRITONSERVER_PARAMETER_BYTES -> const void*
  const void* ValuePointer() const;

  // Return the data byte size of the parameter.
  uint64_t ValueByteSize() const { return byte_size_; }

  // Return the parameter value string, the return value is valid only if
  // Type() returns TRITONSERVER_PARAMETER_STRING
  const std::string& ValueString() const { return value_string_; }

 private:
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceParameter& parameter);

  std::string name_;
  TRITONSERVER_ParameterType type_;

  std::string value_string_;
  int64_t value_int64_;
  bool value_bool_;
  double value_double_;
  const void* value_bytes_;
  uint64_t byte_size_;
};

std::ostream& operator<<(
    std::ostream& out, const InferenceParameter& parameter);

}}  // namespace triton::core
