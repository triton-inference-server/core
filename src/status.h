// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

#include "triton/common/error.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

class Status : public triton::common::Error {
 public:
  // Construct a status from a code with no message.
  explicit Status(Code code = Code::SUCCESS) : Error(code) {}

  // Construct a status from a code and message.
  explicit Status(Code code, const std::string& msg) : Error(code, msg) {}

  // Construct a status from a code and message.
  explicit Status(const Error& error) : Error(error) {}

  // Convenience "success" value. Can be used as Error::Success to
  // indicate no error.
  static const Status Success;

  // Return the code for this status.
  Code StatusCode() const { return code_; }
};

// Return the Status::Code corresponding to a
// TRITONSERVER_Error_Code.
Status::Code TritonCodeToStatusCode(TRITONSERVER_Error_Code code);

// Return the TRITONSERVER_Error_Code corresponding to a
// Status::Code.
TRITONSERVER_Error_Code StatusCodeToTritonCode(Status::Code status_code);

// Converts the common Error to Status object
Status CommonErrorToStatus(const triton::common::Error& error);

// If status is non-OK, return the Status.
#define RETURN_IF_ERROR(S)        \
  do {                            \
    const Status& status__ = (S); \
    if (!status__.IsOk()) {       \
      return status__;            \
    }                             \
  } while (false)

// If TRITONSERVER error is non-OK, return the corresponding status.
#define RETURN_IF_TRITONSERVER_ERROR(E)                          \
  do {                                                           \
    TRITONSERVER_Error* err__ = (E);                             \
    if (err__ != nullptr) {                                      \
      Status status__ = Status(                                  \
          TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err__)), \
          TRITONSERVER_ErrorMessage(err__));                     \
      TRITONSERVER_ErrorDelete(err__);                           \
      return status__;                                           \
    }                                                            \
  } while (false)

// If status is non-OK, return the corresponding TRITONSERVER_Error.
#define RETURN_TRITONSERVER_ERROR_IF_ERROR(S)            \
  do {                                                   \
    const Status& status__ = (S);                        \
    if (!status__.IsOk()) {                              \
      return TRITONSERVER_ErrorNew(                      \
          StatusCodeToTritonCode(status__.StatusCode()), \
          status__.Message().c_str());                   \
    }                                                    \
  } while (false)

}}  // namespace triton::core
