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

#include <azure/storage/blobs.hpp>
#include <azure/storage/common/storage_credential.hpp>

#include "common.h"
// [WIP] below needed?
#undef LOG_INFO
#undef LOG_WARNING

namespace triton { namespace core {

namespace as = Azure::Storage;
namespace asb = Azure::Storage::Blobs;
const std::string AS_URL_PATTERN = "as://([^/]+)/([^/?]+)(?:/([^?]*))?(\\?.*)?";

struct ASCredential {
  std::string account_str_;
  std::string account_key_;

  ASCredential();  // from env var
  ASCredential(triton::common::TritonJson::Value& cred_json);
};

ASCredential::ASCredential()
{
  const auto to_str = [](const char* s) -> std::string {
    return (s != nullptr ? std::string(s) : "");
  };
  const char* account_str = std::getenv("AZURE_STORAGE_ACCOUNT");
  const char* account_key = std::getenv("AZURE_STORAGE_KEY");
  account_str_ = to_str(account_str);
  account_key_ = to_str(account_key);
}

ASCredential::ASCredential(triton::common::TritonJson::Value& cred_json)
{
  triton::common::TritonJson::Value account_str_json, account_key_json;
  if (cred_json.Find("account_str", &account_str_json))
    account_str_json.AsString(&account_str_);
  if (cred_json.Find("account_key", &account_key_json))
    account_key_json.AsString(&account_key_);
}

class ASFileSystem : public FileSystem {
 public:
  ASFileSystem(const std::string& path, const ASCredential& as_cred);
  Status CheckClient();
  // unify with S3 interface
  Status CheckClient(const std::string& path) { return CheckClient(); }

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
  Status ParsePath(
      const std::string& path, std::string* container, std::string* blob);

  // 'callback' will be invoked when directory content is received, it may
  // be invoked multiple times within the same ListDirectory() call if the
  // result is paged.
  Status ListDirectory(
      const std::string& path, const std::string& dir_path,
      std::function<Status(
          const std::vector<asb::Models::BlobItem>& blobs,
          const std::vector<std::string>& blob_prefixes)>
          callback);

  Status DownloadFolder(
      const std::string& container, const std::string& path,
      const std::string& dest);

  std::shared_ptr<asb::BlobServiceClient> client_;
  re2::RE2 as_regex_;
};

Status
ASFileSystem::ParsePath(
    const std::string& path, std::string* container, std::string* blob)
{
  std::string host_name, query;
  if (!RE2::FullMatch(path, as_regex_, &host_name, container, blob, &query)) {
    return Status(
        Status::Code::INTERNAL, "Invalid azure storage path: " + path);
  }
  return Status::Success;
}

ASFileSystem::ASFileSystem(const std::string& path, const ASCredential& as_cred)
    : as_regex_(AS_URL_PATTERN)
{
  std::string host_name, container, blob_path, query;
  if (RE2::FullMatch(
          path, as_regex_, &host_name, &container, &blob_path, &query)) {
    size_t pos = host_name.rfind(".blob.core.windows.net");
    std::string account_name;
    if (as_cred.account_str_.empty()) {
      if (pos != std::string::npos) {
        account_name = host_name.substr(0, pos);
      } else {
        account_name = host_name;
      }
    } else {
      account_name = as_cred.account_str_;
    }
    std::string service_url(
        "https://" + account_name + ".blob.core.windows.net");

    if (!as_cred.account_key_.empty()) {
      // Shared Key
      auto cred = std::make_shared<as::StorageSharedKeyCredential>(
          account_name, as_cred.account_key_);
      client_ = std::make_shared<asb::BlobServiceClient>(service_url, cred);
    } else {
      client_ = std::make_shared<asb::BlobServiceClient>(service_url);
    }
  }
}

Status
ASFileSystem::CheckClient()
{
  if (client_ == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Unable to create Azure filesystem client. Check account credentials.");
  }
  return Status::Success;
}


Status
ASFileSystem::FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  std::string container, blob;
  RETURN_IF_ERROR(ParsePath(path, &container, &blob));
  auto bc = client_->GetBlobContainerClient(container).GetBlobClient(blob);

  try {
    auto blobProperty = bc.GetProperties().Value;
    *mtime_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                    blobProperty.LastModified)
                    .time_since_epoch()
                    .count();
  }
  catch (as::StorageException& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Unable to get blob property for file at " + path + ":" + ex.what());
  }

  return Status::Success;
};

Status
ASFileSystem::ListDirectory(
    const std::string& container, const std::string& dir_path,
    std::function<Status(
        const std::vector<asb::Models::BlobItem>& blobs,
        const std::vector<std::string>& blob_prefixes)>
        callback)
{
  auto container_client = client_->GetBlobContainerClient(container);
  auto options = asb::ListBlobsOptions();
  // Append a slash to make it easier to list contents
  std::string full_dir = AppendSlash(dir_path);
  options.Prefix = full_dir;

  try {
    for (auto blobPage = container_client.ListBlobsByHierarchy("/", options);
         blobPage.HasPage(); blobPage.MoveToNextPage()) {
      // per-page per-blob
      RETURN_IF_ERROR(callback(blobPage.Blobs, blobPage.BlobPrefixes));
    }
  }
  catch (as::StorageException& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get contents of directory " + dir_path + ":" + ex.what());
  }

  return Status::Success;
}

Status
ASFileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  auto func = [&](const std::vector<asb::Models::BlobItem>& blobs,
                  const std::vector<std::string>& blob_prefixes) {
    for (const auto& blob_item : blobs) {
      // Fail-safe check to ensure the item name is not empty
      if (blob_item.Name.empty()) {
        return Status(
            Status::Code::INTERNAL,
            "Cannot handle item with empty name at " + path);
      }
      contents->insert(BaseName(blob_item.Name));
    }
    for (const auto& directory_item : blob_prefixes) {
      // Fail-safe check to ensure the item name is not empty
      if (directory_item.empty()) {
        return Status(
            Status::Code::INTERNAL,
            "Cannot handle item with empty name at " + path);
      }
      contents->insert(BaseName(directory_item));
    }
    return Status::Success;
  };
  std::string container, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &dir_path));
  return ListDirectory(container, dir_path, func);
}

Status
ASFileSystem::GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs)
{
  auto func = [&](const std::vector<asb::Models::BlobItem>& blobs,
                  const std::vector<std::string>& blob_prefixes) {
    for (const auto& directory_item : blob_prefixes) {
      // Fail-safe check to ensure the item name is not empty
      if (directory_item.empty()) {
        return Status(
            Status::Code::INTERNAL,
            "Cannot handle item with empty name at " + path);
      }
      subdirs->insert(BaseName(directory_item));
    }
    return Status::Success;
  };
  std::string container, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &dir_path));
  return ListDirectory(container, dir_path, func);
}

Status
ASFileSystem::GetDirectoryFiles(
    const std::string& path, std::set<std::string>* files)
{
  auto func = [&](const std::vector<asb::Models::BlobItem>& blobs,
                  const std::vector<std::string>& blob_prefixes) {
    for (const auto& blob_item : blobs) {
      // Fail-safe check to ensure the item name is not empty
      if (blob_item.Name.empty()) {
        return Status(
            Status::Code::INTERNAL,
            "Cannot handle item with empty name at " + path);
      }
      files->insert(BaseName(blob_item.Name));
    }
    return Status::Success;
  };
  std::string container, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &dir_path));
  return ListDirectory(container, dir_path, func);
}

Status
ASFileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;
  std::string container, blob_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &blob_path));

  auto container_client = client_->GetBlobContainerClient(container);
  auto options = asb::ListBlobsOptions();
  // Append a slash to make it easier to list contents
  std::string full_dir = AppendSlash(blob_path);
  options.Prefix = full_dir;
  try {
    for (auto blobPage = container_client.ListBlobsByHierarchy("/", options);
         blobPage.HasPage(); blobPage.MoveToNextPage()) {
      if ((blobPage.Blobs.size() == 1) &&
          (blobPage.Blobs[0].Name == blob_path)) {
        // It's a file
        return Status::Success;
      }
      *is_dir =
          ((blobPage.Blobs.size() > 0) || (blobPage.BlobPrefixes.size() > 0));
      break;
    }
  }
  catch (as::StorageException& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to check if directory at " + path + ":" + ex.what());
  }

  return Status::Success;
};

Status
ASFileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  std::string container, blob_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &blob_path));
  try {
    auto res = client_->GetBlobContainerClient(container)
                   .GetBlobClient(blob_path)
                   .Download();
    *contents = std::string(
        (const char*)res.Value.BodyStream->ReadToEnd().data(),
        res.Value.BlobSize);
  }
  catch (as::StorageException& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to read text file at " + path + ":" + ex.what());
  }
  return Status::Success;
}

Status
ASFileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = false;

  std::string container, blob;
  RETURN_IF_ERROR(ParsePath(path, &container, &blob));

  auto container_client = client_->GetBlobContainerClient(container);
  auto options = asb::ListBlobsOptions();
  options.Prefix = blob;
  try {
    for (auto blobPage = container_client.ListBlobsByHierarchy("/", options);
         blobPage.HasPage(); blobPage.MoveToNextPage()) {
      // If any entries are returned from ListBlobs, the file / directory exists
      *exists =
          ((blobPage.Blobs.size() > 0) || (blobPage.BlobPrefixes.size() > 0));
      break;
    }
  }
  catch (as::StorageException& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to check if file exists at " + path + ":" + ex.what());
  }

  return Status::Success;
}

Status
ASFileSystem::DownloadFolder(
    const std::string& container, const std::string& path,
    const std::string& dest)
{
  auto container_client = client_->GetBlobContainerClient(container);
  auto func = [&](const std::vector<asb::Models::BlobItem>& blobs,
                  const std::vector<std::string>& blob_prefixes) {
    for (const auto& blob_item : blobs) {
      const auto& local_path = JoinPath({dest, BaseName(blob_item.Name)});
      try {
        container_client.GetBlobClient(blob_item.Name).DownloadTo(local_path);
      }
      catch (as::StorageException& ex) {
        return Status(
            Status::Code::INTERNAL,
            "Failed to download file at " + blob_item.Name + ":" + ex.what());
      }
    }
    for (const auto& directory_item : blob_prefixes) {
      const auto& local_path = JoinPath({dest, BaseName(directory_item)});
      int status = mkdir(
          const_cast<char*>(local_path.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);
      if (status == -1) {
        return Status(
            Status::Code::INTERNAL,
            "Failed to create local folder: " + local_path +
                ", errno:" + strerror(errno));
      }
      RETURN_IF_ERROR(DownloadFolder(container, directory_item, local_path));
    }
    return Status::Success;
  };
  return ListDirectory(container, path, func);
}

Status
ASFileSystem::LocalizePath(
    const std::string& path, std::shared_ptr<LocalizedPath>* localized)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));
  if (!exists) {
    return Status(
        Status::Code::INTERNAL, "directory or file does not exist at " + path);
  }

  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (!is_dir) {
    return Status(
        Status::Code::UNSUPPORTED,
        "AS file localization not yet implemented " + path);
  }

  // Create a local directory for azure model store.
  // If ENV variable are not set, creates a temporary directory
  // under `/tmp` with the format: "folderXXXXXX".
  // Otherwise, will create a folder under specified directory with the same
  // format.
  std::string env_mount_dir = GetEnvironmentVariableOrDefault(
      "TRITON_AZURE_MOUNT_DIRECTORY", kDefaultMountDirectory);
  std::string tmp_folder;
  RETURN_IF_ERROR(triton::core::MakeTemporaryDirectory(
      FileSystemType::LOCAL, env_mount_dir, &tmp_folder));

  localized->reset(new LocalizedPath(path, tmp_folder));

  std::string dest(tmp_folder);

  std::string container, blob;
  RETURN_IF_ERROR(ParsePath(path, &container, &blob));
  return DownloadFolder(container, blob, dest);
}

Status
ASFileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  std::string container, blob;
  RETURN_IF_ERROR(ParsePath(path, &container, &blob));
  try {
    client_->GetBlobContainerClient(container)
        .GetBlockBlobClient(blob)
        .UploadFrom((const uint8_t*)contents.data(), contents.size());
  }
  catch (as::StorageException& ex) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to write file to " + path + ":" + ex.what());
  }
  return Status::Success;
}

Status
ASFileSystem::WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Write text file operation not yet implemented " + path);
}

Status
ASFileSystem::MakeDirectory(const std::string& dir, const bool recursive)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Make directory operation not yet implemented");
}

Status
ASFileSystem::MakeTemporaryDirectory(
    std::string dir_path, std::string* temp_dir)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Make temporary directory operation not yet implemented");
}

Status
ASFileSystem::DeletePath(const std::string& path)
{
  return Status(
      Status::Code::UNSUPPORTED, "Delete path operation not yet implemented");
}

}}  // namespace triton::core
