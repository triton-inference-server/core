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

#include "api.h"

#include "implementations/common.h"

// Different file system implementations
#include "implementations/local.h"
#ifdef TRITON_ENABLE_GCS
#include "implementations/gcs.h"
#endif  // TRITON_ENABLE_GCS
#ifdef TRITON_ENABLE_S3
#include "implementations/s3.h"
#endif  // TRITON_ENABLE_S3
#ifdef TRITON_ENABLE_AZURE_STORAGE
#include "implementations/as.h"
#endif  // TRITON_ENABLE_AZURE_STORAGE

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <stdio.h>
#include <stdlib.h>

#include <mutex>

namespace triton { namespace core {

LocalizedPath::~LocalizedPath()
{
  if (!local_path_.empty()) {
    bool is_dir = true;
    IsDirectory(local_path_, &is_dir);
    LOG_STATUS_ERROR(
        DeletePath(is_dir ? local_path_ : DirName(local_path_)),
        "failed to delete localized path");
  }
}

namespace {

class FileSystemManager {
 public:
  Status GetFileSystem(
      const std::string& path, std::shared_ptr<FileSystem>& file_system);
  Status GetFileSystem(
      FileSystemType type, std::shared_ptr<FileSystem>& file_system);
  FileSystemManager();

 private:
  template <class CacheType, class CredentialType, class FileSystemType>
  Status GetFileSystem(
      const std::string& path, CacheType& cache,
      std::shared_ptr<FileSystem>& file_system);
  template <class CacheType, class CredentialType, class FileSystemType>
  Status ReturnErrorOrReload(
      const Status& load_status, const Status& error_status,
      const std::string& path, CacheType& cache,
      std::shared_ptr<FileSystem>& file_system);
  Status LoadCredentials(bool flush_cache = false);
  template <class CacheType, class CredentialType, class FileSystemType>
  static void LoadCredential(
      triton::common::TritonJson::Value& creds_json, const char* fs_type,
      CacheType& cache);
  template <class CredentialType, class FileSystemType>
  static void SortCache(
      std::vector<std::tuple<
          std::string, CredentialType, std::shared_ptr<FileSystemType>>>&
          cache);
  template <class CredentialType, class FileSystemType>
  static Status GetLongestMatchingNameIndex(
      const std::vector<std::tuple<
          std::string, CredentialType, std::shared_ptr<FileSystemType>>>& cache,
      const std::string& path, size_t& idx);

  std::shared_ptr<LocalFileSystem> local_fs_;
  std::mutex mu_;   // protect concurrent access into variables
  bool is_cached_;  // if name and credential is cached, lazy load file system
  // cloud credential cache should be sorted in descending name length order
  // [(name_long, credential, file_system), (name, ...)]
#ifdef TRITON_ENABLE_GCS
  std::vector<
      std::tuple<std::string, GCSCredential, std::shared_ptr<GCSFileSystem>>>
      gs_cache_;
#endif  // TRITON_ENABLE_GCS
#ifdef TRITON_ENABLE_S3
  std::vector<
      std::tuple<std::string, S3Credential, std::shared_ptr<S3FileSystem>>>
      s3_cache_;
#endif  // TRITON_ENABLE_S3
#ifdef TRITON_ENABLE_AZURE_STORAGE
  std::vector<
      std::tuple<std::string, ASCredential, std::shared_ptr<ASFileSystem>>>
      as_cache_;
#endif  // TRITON_ENABLE_AZURE_STORAGE
};

FileSystemManager::FileSystemManager()
    : local_fs_(new LocalFileSystem()), is_cached_(false)
{
}

Status
FileSystemManager::GetFileSystem(
    const std::string& path, std::shared_ptr<FileSystem>& file_system)
{
  // Check if this is a GCS path (gs://$BUCKET_NAME)
  if (!path.empty() && !path.rfind("gs://", 0)) {
#ifndef TRITON_ENABLE_GCS
    return Status(
        Status::Code::INTERNAL,
        "gs:// file-system not supported. To enable, build with "
        "-DTRITON_ENABLE_GCS=ON.");
#else
    return GetFileSystem<
        std::vector<std::tuple<
            std::string, GCSCredential, std::shared_ptr<GCSFileSystem>>>,
        GCSCredential, GCSFileSystem>(path, gs_cache_, file_system);
#endif  // TRITON_ENABLE_GCS
  }

  // Check if this is an S3 path (s3://$BUCKET_NAME)
  if (!path.empty() && !path.rfind("s3://", 0)) {
#ifndef TRITON_ENABLE_S3
    return Status(
        Status::Code::INTERNAL,
        "s3:// file-system not supported. To enable, build with "
        "-DTRITON_ENABLE_S3=ON.");
#else
    return GetFileSystem<
        std::vector<std::tuple<
            std::string, S3Credential, std::shared_ptr<S3FileSystem>>>,
        S3Credential, S3FileSystem>(path, s3_cache_, file_system);
#endif  // TRITON_ENABLE_S3
  }

  // Check if this is an Azure Storage path
  if (!path.empty() && !path.rfind("as://", 0)) {
#ifndef TRITON_ENABLE_AZURE_STORAGE
    return Status(
        Status::Code::INTERNAL,
        "as:// file-system not supported. To enable, build with "
        "-DTRITON_ENABLE_AZURE_STORAGE=ON.");
#else
    return GetFileSystem<
        std::vector<std::tuple<
            std::string, ASCredential, std::shared_ptr<ASFileSystem>>>,
        ASCredential, ASFileSystem>(path, as_cache_, file_system);
#endif  // TRITON_ENABLE_AZURE_STORAGE
  }

  // Assume path is for local filesystem
  file_system = local_fs_;
  return Status::Success;
}

Status
FileSystemManager::GetFileSystem(
    FileSystemType type, std::shared_ptr<FileSystem>& file_system)
{
  // only LOCAL and GCS are not path-dependent and can be accessed by type
  switch (type) {
    case FileSystemType::LOCAL:
      return GetFileSystem("", file_system);
    case FileSystemType::GCS:
      return GetFileSystem("gs://", file_system);
    case FileSystemType::S3:
      return Status(
          Status::Code::UNSUPPORTED,
          "S3 filesystem cannot be accessed by type");
    case FileSystemType::AS:
      return Status(
          Status::Code::UNSUPPORTED,
          "AS filesystem cannot be accessed by type");
    default:
      return Status(Status::Code::UNSUPPORTED, "Unsupported filesystem type");
  }
}

template <class CacheType, class CredentialType, class FileSystemType>
Status
FileSystemManager::GetFileSystem(
    const std::string& path, CacheType& cache,
    std::shared_ptr<FileSystem>& file_system)
{
  const Status& cred_status = LoadCredentials();
  if (cred_status.IsOk() ||
      cred_status.StatusCode() == Status::Code::ALREADY_EXISTS) {
    // Find credential
    size_t idx;
    const Status& match_status = GetLongestMatchingNameIndex(cache, path, idx);
    if (!match_status.IsOk()) {
      return ReturnErrorOrReload<CacheType, CredentialType, FileSystemType>(
          cred_status, match_status, path, cache, file_system);
    }
    // Find or lazy load file system
    std::shared_ptr<FileSystemType> fs = std::get<2>(cache[idx]);
    if (fs == nullptr) {
      std::string cred_name = std::get<0>(cache[idx]);
      CredentialType cred = std::get<1>(cache[idx]);
      fs = std::make_shared<FileSystemType>(path, cred);
      cache[idx] = std::make_tuple(cred_name, cred, fs);
    }
    // Check client
    const Status& client_status = fs->CheckClient(path);
    if (!client_status.IsOk()) {
      return ReturnErrorOrReload<CacheType, CredentialType, FileSystemType>(
          cred_status, client_status, path, cache, file_system);
    }
    // Return client
    file_system = fs;
    return Status::Success;
  }
  return cred_status;
}

template <class CacheType, class CredentialType, class FileSystemType>
Status
FileSystemManager::ReturnErrorOrReload(
    const Status& load_status, const Status& error_status,
    const std::string& path, CacheType& cache,
    std::shared_ptr<FileSystem>& file_system)
{
  if (load_status.StatusCode() == Status::Code::ALREADY_EXISTS) {
    return error_status;
  }
  LoadCredentials(true);  // flush cache
  return GetFileSystem<CacheType, CredentialType, FileSystemType>(
      path, cache, file_system);
}

// return status meaning:
// - SUCCESS, "" -> loaded credential from file
// - ALREADY_EXISTS, "Cached" -> credential already loaded
Status
FileSystemManager::LoadCredentials(bool flush_cache)
{
  // prevent concurrent access into class variables
  std::lock_guard<std::mutex> lock(mu_);

  // check if credential is already cached
  if (is_cached_ && !flush_cache) {
    return Status(Status::Code::ALREADY_EXISTS, "Cached");
  }

  const char* file_path_c_str = std::getenv("TRITON_CLOUD_CREDENTIAL_PATH");
  if (file_path_c_str != nullptr) {
    // Load from credential file
    std::string file_path = std::string(file_path_c_str);
    LOG_VERBOSE(1) << "Reading cloud credential from " << file_path;

    triton::common::TritonJson::Value creds_json;
    std::string cred_file_content;
    RETURN_IF_ERROR(local_fs_->ReadTextFile(file_path, &cred_file_content));
    RETURN_IF_ERROR(creds_json.Parse(cred_file_content));

#ifdef TRITON_ENABLE_GCS
    // load GCS credentials
    LoadCredential<
        std::vector<std::tuple<
            std::string, GCSCredential, std::shared_ptr<GCSFileSystem>>>,
        GCSCredential, GCSFileSystem>(creds_json, "gs", gs_cache_);
#endif  // TRITON_ENABLE_GCS
#ifdef TRITON_ENABLE_S3
    // load S3 credentials
    LoadCredential<
        std::vector<std::tuple<
            std::string, S3Credential, std::shared_ptr<S3FileSystem>>>,
        S3Credential, S3FileSystem>(creds_json, "s3", s3_cache_);
#endif  // TRITON_ENABLE_S3
#ifdef TRITON_ENABLE_AZURE_STORAGE
    // load AS credentials
    LoadCredential<
        std::vector<std::tuple<
            std::string, ASCredential, std::shared_ptr<ASFileSystem>>>,
        ASCredential, ASFileSystem>(creds_json, "as", as_cache_);
#endif  // TRITON_ENABLE_AZURE_STORAGE
  } else {
    // Load from environment variables
    LOG_VERBOSE(1) << "TRITON_CLOUD_CREDENTIAL_PATH environment variable is "
                      "not set, reading from environment variables";

#ifdef TRITON_ENABLE_GCS
    // load GCS credentials
    gs_cache_.clear();
    gs_cache_.push_back(
        std::make_tuple("", GCSCredential(), std::shared_ptr<GCSFileSystem>()));
#endif  // TRITON_ENABLE_GCS

#ifdef TRITON_ENABLE_S3
    // load S3 credentials
    s3_cache_.clear();
    s3_cache_.push_back(
        std::make_tuple("", S3Credential(), std::shared_ptr<S3FileSystem>()));
#endif  // TRITON_ENABLE_S3

#ifdef TRITON_ENABLE_AZURE_STORAGE
    // load AS credentials
    as_cache_.clear();
    as_cache_.push_back(
        std::make_tuple("", ASCredential(), std::shared_ptr<ASFileSystem>()));
#endif  // TRITON_ENABLE_AZURE_STORAGE
  }

  is_cached_ = true;
  return Status::Success;
}

template <class CacheType, class CredentialType, class FileSystemType>
void
FileSystemManager::LoadCredential(
    triton::common::TritonJson::Value& creds_json, const char* fs_type,
    CacheType& cache)
{
  cache.clear();
  triton::common::TritonJson::Value creds_fs_json;
  if (creds_json.Find(fs_type, &creds_fs_json)) {
    std::vector<std::string> cred_names;
    creds_fs_json.Members(&cred_names);
    for (size_t i = 0; i < cred_names.size(); i++) {
      std::string cred_name = cred_names[i];
      triton::common::TritonJson::Value cred_json;
      creds_fs_json.Find(cred_name.c_str(), &cred_json);
      cache.push_back(std::make_tuple(
          cred_name, CredentialType(cred_json),
          std::shared_ptr<FileSystemType>()));
    }
    SortCache(cache);
  }
}

template <class CredentialType, class FileSystemType>
void
FileSystemManager::SortCache(
    std::vector<std::tuple<
        std::string, CredentialType, std::shared_ptr<FileSystemType>>>& cache)
{
  std::sort(
      cache.begin(), cache.end(),
      [](std::tuple<
             std::string, CredentialType, std::shared_ptr<FileSystemType>>
             a,
         std::tuple<
             std::string, CredentialType, std::shared_ptr<FileSystemType>>
             b) { return std::get<0>(a).size() >= std::get<0>(b).size(); });
}

template <class CredentialType, class FileSystemType>
Status
FileSystemManager::GetLongestMatchingNameIndex(
    const std::vector<std::tuple<
        std::string, CredentialType, std::shared_ptr<FileSystemType>>>& cache,
    const std::string& path, size_t& idx)
{
  for (size_t i = 0; i < cache.size(); i++) {
    if (!path.rfind(std::get<0>(cache[i]), 0)) {
      idx = i;
      LOG_VERBOSE(1) << "Using credential  " + std::get<0>(cache[i]) +
                            "  for path  " + path;
      return Status::Success;
    }
  }
  return Status(
      Status::Code::NOT_FOUND, "Cannot match credential for path  " + path);
}

static FileSystemManager fsm_;

}  // namespace

// FIXME: Windows support '/'? If so, the below doesn't need to change
bool
IsAbsolutePath(const std::string& path)
{
  return !path.empty() && (path[0] == '/');
}

std::string
JoinPath(std::initializer_list<std::string> segments)
{
  std::string joined;

  for (const auto& seg : segments) {
    if (joined.empty()) {
      joined = seg;
    } else if (IsAbsolutePath(seg)) {
      if (joined[joined.size() - 1] == '/') {
        joined.append(seg.substr(1));
      } else {
        joined.append(seg);
      }
    } else {  // !IsAbsolutePath(seg)
      if (joined[joined.size() - 1] != '/') {
        joined.append("/");
      }
      joined.append(seg);
    }
  }

  return joined;
}

std::string
BaseName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string();
  }

  const size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return path.substr(0, last + 1);
  }

  return path.substr(idx + 1, last - idx);
}

std::string
DirName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string("/");
  }

  const size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return std::string(".");
  }
  if (idx == 0) {
    return std::string("/");
  }

  return path.substr(0, idx);
}

Status
FileExists(const std::string& path, bool* exists)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->FileExists(path, exists);
}

Status
IsDirectory(const std::string& path, bool* is_dir)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->IsDirectory(path, is_dir);
}

Status
FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->FileModificationTime(path, mtime_ns);
}

Status
GetDirectoryContents(const std::string& path, std::set<std::string>* contents)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->GetDirectoryContents(path, contents);
}

Status
GetDirectorySubdirs(const std::string& path, std::set<std::string>* subdirs)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->GetDirectorySubdirs(path, subdirs);
}

Status
GetDirectoryFiles(
    const std::string& path, const bool skip_hidden_files,
    std::set<std::string>* files)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  std::set<std::string> all_files;
  RETURN_IF_ERROR(fs->GetDirectoryFiles(path, &all_files));
  // Remove the hidden files
  for (auto f : all_files) {
    if ((f[0] != '.') || (!skip_hidden_files)) {
      files->insert(f);
    }
  }
  return Status::Success;
}

Status
ReadTextFile(const std::string& path, std::string* contents)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->ReadTextFile(path, contents);
}

Status
ReadTextProto(const std::string& path, google::protobuf::Message* msg)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));

  std::string contents;
  RETURN_IF_ERROR(fs->ReadTextFile(path, &contents));

  if (!google::protobuf::TextFormat::ParseFromString(contents, msg)) {
    return Status(
        Status::Code::INTERNAL, "failed to read text proto from " + path);
  }

  return Status::Success;
}

Status
LocalizePath(const std::string& path, std::shared_ptr<LocalizedPath>* localized)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->LocalizePath(path, localized);
}

Status
WriteTextProto(const std::string& path, const google::protobuf::Message& msg)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));

  std::string prototxt;
  if (!google::protobuf::TextFormat::PrintToString(msg, &prototxt)) {
    return Status(
        Status::Code::INTERNAL, "failed to write text proto to " + path);
  }

  return fs->WriteTextFile(path, prototxt);
}

Status
WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->WriteBinaryFile(path, contents, content_len);
}

Status
ReadBinaryProto(const std::string& path, google::protobuf::MessageLite* msg)
{
  std::string msg_str;
  RETURN_IF_ERROR(ReadTextFile(path, &msg_str));

  google::protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8_t*>(msg_str.c_str()), msg_str.size());
  coded_stream.SetTotalBytesLimit(INT_MAX);
  if (!msg->ParseFromCodedStream(&coded_stream)) {
    return Status(
        Status::Code::INTERNAL, "Can't parse " + path + " as binary proto");
  }

  return Status::Success;
}

Status
MakeDirectory(const std::string& dir, const bool recursive)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(dir, fs));
  return fs->MakeDirectory(dir, recursive);
}

Status
MakeTemporaryDirectory(const FileSystemType type, std::string* temp_dir)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(type, fs));
  return fs->MakeTemporaryDirectory(kDefaultMountDirectory, temp_dir);
}

Status
MakeTemporaryDirectory(
    const FileSystemType type, std::string dir_path, std::string* temp_dir)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(type, fs));
  return fs->MakeTemporaryDirectory(dir_path, temp_dir);
}

Status
DeletePath(const std::string& path)
{
  std::shared_ptr<FileSystem> fs;
  RETURN_IF_ERROR(fsm_.GetFileSystem(path, fs));
  return fs->DeletePath(path);
}

Status
GetFileSystemType(const std::string& path, FileSystemType* type)
{
  if (path.empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "Can not infer filesystem type from empty path");
  }
#ifdef TRITON_ENABLE_GCS
  // Check if this is a GCS path (gs://$BUCKET_NAME)
  if (!path.rfind("gs://", 0)) {
    *type = FileSystemType::GCS;
    return Status::Success;
  }
#endif  // TRITON_ENABLE_GCS

#ifdef TRITON_ENABLE_S3
  // Check if this is an S3 path (s3://$BUCKET_NAME)
  if (!path.rfind("s3://", 0)) {
    *type = FileSystemType::S3;
    return Status::Success;
  }
#endif  // TRITON_ENABLE_S3

#ifdef TRITON_ENABLE_AZURE_STORAGE
  // Check if this is an Azure Storage path
  if (!path.rfind("as://", 0)) {
    *type = FileSystemType::AS;
    return Status::Success;
  }
#endif  // TRITON_ENABLE_AZURE_STORAGE

  // Assume path is for local filesystem
  *type = FileSystemType::LOCAL;
  return Status::Success;
}

const std::string&
FileSystemTypeString(const FileSystemType type)
{
  static const std::string local_str("LOCAL");
  static const std::string gcs_str("GCS");
  static const std::string s3_str("S3");
  static const std::string as_str("AS");
  static const std::string unknown_str("UNKNOWN");
  switch (type) {
    case FileSystemType::LOCAL:
      return local_str;
    case FileSystemType::GCS:
      return gcs_str;
    case FileSystemType::S3:
      return s3_str;
    case FileSystemType::AS:
      return as_str;
    default:
      return unknown_str;
  }
}

}}  // namespace triton::core
