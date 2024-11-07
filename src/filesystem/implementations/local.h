// Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>

#include "../../constants.h"
#include "common.h"

namespace triton { namespace core {

class LocalFileSystem : public FileSystem {
 public:
  Status FileExists(const std::string& path, bool* exists) override;
  Status IsDirectory(const std::string& path, bool* is_dir) override;
  Status FileModificationTime(
      const std::string& path, int64_t* mtime_ns) override;
  Status GetDirectoryContents(
      const std::string& path, std::set<std::string>* contents) override;
  Status GetDirectorySubdirs(
      const std::string& path, std::set<std::string>* subdirs) override;
  Status GetDirectoryFiles(
      const std::string& path, std::set<std::string>* files) override;
  Status ReadTextFile(const std::string& path, std::string* contents) override;
  Status LocalizePath(
      const std::string& path,
      std::shared_ptr<LocalizedPath>* localized) override;
  Status WriteTextFile(
      const std::string& path, const std::string& contents) override;
  Status WriteBinaryFile(
      const std::string& path, const char* contents,
      const size_t content_len) override;
  Status MakeDirectory(const std::string& dir, const bool recursive) override;
  Status MakeTemporaryDirectory(
      std::string dir_path, std::string* temp_dir) override;
  Status DeletePath(const std::string& path) override;

 private:
  inline std::string GetOSValidPath(const std::string& path);
};

//! Converts incoming utf-8 path to an OS valid path
//!
//! On Linux there is not much to do but make sure correct slashes are used
//! On Windows we need to take care of the long paths and handle them correctly
//! to avoid legacy issues with MAX_PATH
//!
//! More details:
//! https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
//!
inline std::string
LocalFileSystem::GetOSValidPath(const std::string& path)
{
  std::string l_path(path);
#ifdef _WIN32
  // On Windows long paths must be marked correctly otherwise, due to backwards
  // compatibility, all paths are limited to MAX_PATH length
  static constexpr const char* kWindowsLongPathPrefix = "\\\\?\\";
  if (l_path.size() >= MAX_PATH) {
    // Must be prefixed with "\\?\" to be considered long path
    if (l_path.substr(0, 4) != (kWindowsLongPathPrefix)) {
      // Long path but not "tagged" correctly
      l_path = (kWindowsLongPathPrefix) + l_path;
    }
  }
  std::replace(l_path.begin(), l_path.end(), '/', '\\');
#endif
  return l_path;
}

Status
LocalFileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = (access(GetOSValidPath(path).c_str(), F_OK) == 0);
  return Status::Success;
}

Status
LocalFileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;

  struct stat st;
  if (stat(GetOSValidPath(path).c_str(), &st) != 0) {
    return Status(Status::Code::INTERNAL, "failed to stat file " + path);
  }

  *is_dir = S_ISDIR(st.st_mode);
  return Status::Success;
}

Status
LocalFileSystem::FileModificationTime(
    const std::string& path, int64_t* mtime_ns)
{
  struct stat st;
  if (stat(GetOSValidPath(path).c_str(), &st) != 0) {
    return Status(Status::Code::INTERNAL, "failed to stat file " + path);
  }

#ifdef _WIN32
  // In Windows, st_mtime is in time_t
  *mtime_ns = std::max(st.st_mtime, st.st_ctime);
#else
  *mtime_ns =
      std::max(TIMESPEC_TO_NANOS(st.st_mtim), TIMESPEC_TO_NANOS(st.st_ctim));
#endif
  return Status::Success;
}

Status
LocalFileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
#ifdef _WIN32
  WIN32_FIND_DATA entry;
  // Append "*" to obtain all files under 'path'
  HANDLE dir = FindFirstFile(JoinPath({path, "*"}).c_str(), &entry);
  if (dir == INVALID_HANDLE_VALUE) {
    return Status(Status::Code::INTERNAL, "failed to open directory " + path);
  }
  if ((strcmp(entry.cFileName, ".") != 0) &&
      (strcmp(entry.cFileName, "..") != 0)) {
    contents->insert(entry.cFileName);
  }
  while (FindNextFile(dir, &entry)) {
    if ((strcmp(entry.cFileName, ".") != 0) &&
        (strcmp(entry.cFileName, "..") != 0)) {
      contents->insert(entry.cFileName);
    }
  }

  FindClose(dir);
#else
  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    return Status(Status::Code::INTERNAL, "failed to open directory " + path);
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string entryname = entry->d_name;
    if ((entryname != ".") && (entryname != "..")) {
      contents->insert(entryname);
    }
  }

  closedir(dir);
#endif
  return Status::Success;
}

Status
LocalFileSystem::GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, subdirs));

  // Erase non-directory entries...
  for (auto iter = subdirs->begin(); iter != subdirs->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (!is_dir) {
      iter = subdirs->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
LocalFileSystem::GetDirectoryFiles(
    const std::string& path, std::set<std::string>* files)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, files));

  // Erase directory entries...
  for (auto iter = files->begin(); iter != files->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (is_dir) {
      iter = files->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
LocalFileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  std::ifstream in(GetOSValidPath(path), std::ios::in | std::ios::binary);
  if (!in) {
    return Status(
        Status::Code::INTERNAL,
        "failed to open text file for read " + path + ": " + strerror(errno));
  }

  in.seekg(0, std::ios::end);
  contents->resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&(*contents)[0], contents->size());
  in.close();

  return Status::Success;
}

Status
LocalFileSystem::LocalizePath(
    const std::string& path, std::shared_ptr<LocalizedPath>* localized)
{
  // For local file system we don't actually need to download the
  // directory or file. We use it in place.
  localized->reset(new LocalizedPath(path));
  return Status::Success;
}

Status
LocalFileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    return Status(
        Status::Code::INTERNAL,
        "failed to open text file for write " + path + ": " + strerror(errno));
  }

  out.write(&contents[0], contents.size());
  out.close();

  return Status::Success;
}

Status
LocalFileSystem::WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len)
{
  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    return Status(
        Status::Code::INTERNAL, "failed to open binary file for write " + path +
                                    ": " + strerror(errno));
  }

  out.write(contents, content_len);

  return Status::Success;
}

Status
LocalFileSystem::MakeDirectory(const std::string& dir, const bool recursive)
{
#ifdef _WIN32
  if (mkdir(dir.c_str()) == -1)
#else
  if (mkdir(dir.c_str(), S_IRWXU) == -1)
#endif
  {
    // Only allow the error due to parent directory does not exist
    // if 'recursive' is requested
    if ((errno == ENOENT) && (!dir.empty()) && recursive) {
      RETURN_IF_ERROR(MakeDirectory(DirName(dir), recursive));
      // Retry the creation
#ifdef _WIN32
      if (mkdir(dir.c_str()) == -1)
#else
      if (mkdir(dir.c_str(), S_IRWXU) == -1)
#endif
      {
        return Status(
            Status::Code::INTERNAL, "Failed to create directory '" + dir +
                                        "', errno:" + strerror(errno));
      }
    } else {
      return Status(
          Status::Code::INTERNAL,
          "Failed to create directory '" + dir + "', errno:" + strerror(errno));
    }
  }

  return Status::Success;
}

Status
LocalFileSystem::MakeTemporaryDirectory(
    std::string dir_path, std::string* temp_dir)
{
#ifdef _WIN32
  char temp_path[MAX_PATH + 1];
  size_t temp_path_length = GetTempPath(MAX_PATH + 1, temp_path);
  if (temp_path_length == 0) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get local directory for temporary files");
  }
  // There is no single operation like 'mkdtemp' in Windows, thus generating
  // unique temporary directory is a process of getting temporary file name,
  // deleting the file (file creation is side effect fo getting name), creating
  // corresponding directory, so mutex is used to avoid possible race condition.
  // However, it doesn't prevent other process on creating temporary file and
  // thus the race condition may still happen. One possible solution is
  // to reserve a temporary directory for the process and generate temporary
  // model directories inside it.
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  // Construct a std::string as filled 'temp_path' is not C string,
  // and so that we can reuse 'temp_path' to hold the temp file name.
  std::string temp_path_str(temp_path, temp_path_length);
  if (GetTempFileName(temp_path_str.c_str(), "folder", 0, temp_path) == 0) {
    return Status(Status::Code::INTERNAL, "Failed to create local temp folder");
  }
  *temp_dir = temp_path;
  DeleteFile(temp_dir->c_str());
  if (CreateDirectory(temp_dir->c_str(), NULL) == 0) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + *temp_dir);
  }
#else
  if (dir_path.empty()) {
    dir_path = kDefaultMountDirectory;
  }
  std::string folder_template = JoinPath({dir_path, "folderXXXXXX"});
  char* res = mkdtemp(const_cast<char*>(folder_template.c_str()));
  if (res == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + folder_template +
            ", errno:" + strerror(errno));
  }
  *temp_dir = res;
#endif
  return Status::Success;
}

Status
LocalFileSystem::DeletePath(const std::string& path)
{
  bool is_dir = false;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (is_dir) {
    std::set<std::string> contents;
    RETURN_IF_ERROR(GetDirectoryContents(path, &contents));
    for (const auto& content : contents) {
      RETURN_IF_ERROR(DeletePath(JoinPath({path, content})));
    }
    rmdir(path.c_str());
  } else {
    remove(path.c_str());
  }
  return Status::Success;
}

}}  // namespace triton::core
