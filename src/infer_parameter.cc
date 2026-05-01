// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "infer_parameter.h"

namespace triton { namespace core {


const void*
InferenceParameter::ValuePointer() const
{
  switch (type_) {
    case TRITONSERVER_PARAMETER_STRING:
      return reinterpret_cast<const void*>(value_string_.c_str());
    case TRITONSERVER_PARAMETER_INT:
      return reinterpret_cast<const void*>(&value_int64_);
    case TRITONSERVER_PARAMETER_BOOL:
      return reinterpret_cast<const void*>(&value_bool_);
    case TRITONSERVER_PARAMETER_DOUBLE:
      return reinterpret_cast<const void*>(&value_double_);
    case TRITONSERVER_PARAMETER_BYTES:
      return reinterpret_cast<const void*>(value_bytes_);
    default:
      break;
  }

  return nullptr;
}

std::ostream&
operator<<(std::ostream& out, const InferenceParameter& parameter)
{
  out << "[0x" << std::addressof(parameter) << "] "
      << "name: " << parameter.Name()
      << ", type: " << TRITONSERVER_ParameterTypeString(parameter.Type())
      << ", value: ";
  return out;
}

}}  // namespace triton::core
