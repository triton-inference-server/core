// SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "status.h"

namespace triton { namespace core {

const Status Status::Success(Status::Code::SUCCESS);

Status::Code
TritonCodeToStatusCode(TRITONSERVER_Error_Code code)
{
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return Status::Code::UNKNOWN;
    case TRITONSERVER_ERROR_INTERNAL:
      return Status::Code::INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return Status::Code::NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return Status::Code::INVALID_ARG;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return Status::Code::UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return Status::Code::UNSUPPORTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return Status::Code::ALREADY_EXISTS;
    case TRITONSERVER_ERROR_CANCELLED:
      return Status::Code::CANCELLED;
    default:
      break;
  }

  return Status::Code::UNKNOWN;
}

TRITONSERVER_Error_Code
StatusCodeToTritonCode(Status::Code status_code)
{
  switch (status_code) {
    case Status::Code::UNKNOWN:
      return TRITONSERVER_ERROR_UNKNOWN;
    case Status::Code::INTERNAL:
      return TRITONSERVER_ERROR_INTERNAL;
    case Status::Code::NOT_FOUND:
      return TRITONSERVER_ERROR_NOT_FOUND;
    case Status::Code::INVALID_ARG:
      return TRITONSERVER_ERROR_INVALID_ARG;
    case Status::Code::UNAVAILABLE:
      return TRITONSERVER_ERROR_UNAVAILABLE;
    case Status::Code::UNSUPPORTED:
      return TRITONSERVER_ERROR_UNSUPPORTED;
    case Status::Code::ALREADY_EXISTS:
      return TRITONSERVER_ERROR_ALREADY_EXISTS;
    case Status::Code::CANCELLED:
      return TRITONSERVER_ERROR_CANCELLED;
    default:
      break;
  }

  return TRITONSERVER_ERROR_UNKNOWN;
}

Status
CommonErrorToStatus(const triton::common::Error& error)
{
  return Status(error);
}

}}  // namespace triton::core
