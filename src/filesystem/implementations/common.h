// SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <re2/re2.h>

#include <cerrno>
#include <memory>
#include <set>

#include "../api.h"
#include "triton/common/logging.h"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>

// _CRT_INTERNAL_NONSTDC_NAMES 1 before including Microsoft provided C Runtime
// library to expose declarations without "_" prefix to match POSIX style.
#define _CRT_INTERNAL_NONSTDC_NAMES 1
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

#ifdef _WIN32
// <sys/stat.h> in Windows doesn't define S_ISDIR macro
#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif
#define F_OK 0
#endif

#define TRITONJSON_STATUSTYPE triton::core::Status
#define TRITONJSON_STATUSRETURN(M) \
  return triton::core::Status(triton::core::Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS triton::core::Status::Success
#include "triton/common/triton_json.h"

namespace triton { namespace core {

// Default folder for temporary local cache
constexpr char kDefaultMountDirectory[] = "/tmp";

// FileSystem interface that all file system implementation should inherit from.
// To add new file system support, the implementation should be added and made
// visible to FileSystemManager in api.cc
class FileSystem {
 public:
  virtual Status FileExists(const std::string& path, bool* exists) = 0;
  virtual Status IsDirectory(const std::string& path, bool* is_dir) = 0;
  virtual Status FileModificationTime(
      const std::string& path, int64_t* mtime_ns) = 0;
  virtual Status GetDirectoryContents(
      const std::string& path, std::set<std::string>* contents) = 0;
  virtual Status GetDirectorySubdirs(
      const std::string& path, std::set<std::string>* subdirs) = 0;
  virtual Status GetDirectoryFiles(
      const std::string& path, std::set<std::string>* files) = 0;
  virtual Status ReadTextFile(
      const std::string& path, std::string* contents) = 0;
  virtual Status LocalizePath(
      const std::string& path, std::shared_ptr<LocalizedPath>* localized) = 0;
  virtual Status WriteTextFile(
      const std::string& path, const std::string& contents) = 0;
  virtual Status WriteBinaryFile(
      const std::string& path, const char* contents,
      const size_t content_len) = 0;
  virtual Status MakeDirectory(
      const std::string& dir, const bool recursive) = 0;
  virtual Status MakeTemporaryDirectory(
      std::string dir_path, std::string* temp_dir) = 0;
  virtual Status DeletePath(const std::string& path) = 0;
};

// Helper function to take care of lack of trailing slashes
std::string
AppendSlash(const std::string& name)
{
  if (name.empty() || (name.back() == '/')) {
    return name;
  }

  return (name + "/");
}

/// Helper function to get the value of the environment variable,
/// or default value if not set.
///
/// \param variable_name The name of the environment variable.
/// \param default_value The default value.
/// \return The environment variable or the default value if not set.
std::string
GetEnvironmentVariableOrDefault(
    const std::string& variable_name, const std::string& default_value)
{
  const char* value = getenv(variable_name.c_str());
  return value ? value : default_value;
}

}}  // namespace triton::core
