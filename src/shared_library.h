// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "constants.h"
#include "status.h"

namespace triton { namespace core {

// SharedLibrary
//
// Utility functions for shared libraries. Because some operations
// require serialization, this object cannot be directly constructed
// and must instead be accessed using Acquire().
class SharedLibrary {
 public:
  // Acquire a SharedLibrary object exclusively. Any other attempts to
  // concurrently acquire a SharedLibrary object will block.
  // object. Ownership is released by destroying the SharedLibrary
  // object.
  static Status Acquire(std::unique_ptr<SharedLibrary>* slib);

  ~SharedLibrary();

  // Configuration so that dependent libraries will be searched for in
  // 'path' during OpenLibraryHandle.
  Status AddLibraryDirectory(const std::string& path, void** directory_cookie);

  // Removes a library directory set by AddLibraryDirectory.
  Status RemoveLibraryDirectory(void* directory_cookie);

  // Open shared library and return generic handle.
  Status OpenLibraryHandle(const std::string& path, void** handle);

  // Close shared library.
  Status CloseLibraryHandle(void* handle);

  // Get a generic pointer for an entrypoint into a shared library.
  Status GetEntrypoint(
      void* handle, const std::string& name, const bool optional, void** befn);

  // Add an additional dependency directories to load search.
  Status SetAdditionalDependencyDirs(const std::string& additional_path);

#ifdef _WIN32
 private:
  Status SharedLibrary::AddAdditionalDependencyDirs();
  Status SharedLibrary::RemoveAdditionalDependencyDirs();

  std::vector<std::string> mAdditionalDependencyDirs;
  std::vector<void*> mAdditionalDirHandles;
#endif

 private:
  DISALLOW_COPY_AND_ASSIGN(SharedLibrary);
  explicit SharedLibrary() = default;
};

}}  // namespace triton::core
