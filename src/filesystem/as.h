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

#include "common.h"

#include <blob/blob_client.h>
#include <storage_account.h>
#include <storage_credential.h>
// [WIP] below needed?
#undef LOG_INFO
#undef LOG_WARNING

namespace triton { namespace core {

namespace as = azure::storage_lite;
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
  Status MakeTemporaryDirectory(std::string* temp_dir) override;
  Status DeletePath(const std::string& path) override;

 private:
  Status ParsePath(
      const std::string& path, std::string* bucket, std::string* object);
  std::shared_ptr<as::blob_client> client_;

  Status ListDirectory(
      const std::string& path, const std::string& dir_path,
      std::function<
          Status(const as::list_blobs_segmented_item&, const std::string&)>
          func);

  Status DownloadFolder(
      const std::string& container, const std::string& path,
      const std::string& dest);
  re2::RE2 as_regex_;
};

Status
ASFileSystem::ParsePath(
    const std::string& path, std::string* container, std::string* object)
{
  std::string host_name, query;
  if (!RE2::FullMatch(path, as_regex_, &host_name, container, object, &query)) {
    return Status(
        Status::Code::INTERNAL, "Invalid azure storage path: " + path);
  }
  return Status::Success;
}

ASFileSystem::ASFileSystem(const std::string& path, const ASCredential& as_cred)
    : as_regex_(AS_URL_PATTERN)
{
  std::shared_ptr<as::storage_account> account = nullptr;
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

    std::shared_ptr<as::storage_credential> cred;
    if (!as_cred.account_key_.empty()) {
      // Shared Key
      cred = std::make_shared<as::shared_key_credential>(
          account_name, as_cred.account_key_);
    } else {
      cred = std::make_shared<as::anonymous_credential>();
    }
    account = std::make_shared<as::storage_account>(
        account_name, cred, /* use_https */ true);
    client_ =
        std::make_shared<as::blob_client>(account, /*max_concurrency*/ 16);
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
  as::blob_client_wrapper bc(client_);
  std::string container, object_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &object_path));

  auto blobProperty = bc.get_blob_property(container, object_path);
  if (errno != 0) {
    return Status(
        Status::Code::INTERNAL, "Unable to get blob property for file at " +
                                    path + ", errno:" + strerror(errno));
  }

  auto time =
      std::chrono::system_clock::from_time_t(blobProperty.last_modified);
  auto update_time =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(time)
          .time_since_epoch()
          .count();

  *mtime_ns = update_time;
  return Status::Success;
};

Status
ASFileSystem::ListDirectory(
    const std::string& container, const std::string& dir_path,
    std::function<
        Status(const as::list_blobs_segmented_item&, const std::string&)>
        func)
{
  as::blob_client_wrapper bc(client_);

  // Append a slash to make it easier to list contents
  std::string full_dir = AppendSlash(dir_path);
  auto blobs = bc.list_blobs_segmented(container, "/", "", full_dir);
  if (errno != 0) {
    return Status(
        Status::Code::INTERNAL, "Failed to get contents of directory " +
                                    dir_path + ", errno:" + strerror(errno));
  }

  for (auto&& item : blobs.blobs) {
    std::string name = item.name;
    int item_start = name.find(full_dir) + full_dir.size();
    int item_end = name.find("/", item_start);
    // Let set take care of subdirectory contents
    std::string subfile = name.substr(item_start, item_end - item_start);
    auto status = func(item, subfile);
    if (!status.IsOk()) {
      return status;
    }
  }
  return Status::Success;
}

Status
ASFileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  auto func = [&](const as::list_blobs_segmented_item& item,
                  const std::string& dir) {
    contents->insert(dir);
    // Fail-safe check to ensure the item name is not empty
    if (dir.empty()) {
      return Status(
          Status::Code::INTERNAL,
          "Cannot handle item with empty name at " + path);
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
  auto func = [&](const as::list_blobs_segmented_item& item,
                  const std::string& dir) {
    if (item.is_directory) {
      subdirs->insert(dir);
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
  auto func = [&](const as::list_blobs_segmented_item& item,
                  const std::string& file) {
    if (!item.is_directory) {
      files->insert(file);
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
  std::string container, object_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &object_path));

  as::blob_client_wrapper bc(client_);
  auto blobs = bc.list_blobs_segmented(container, "/", "", object_path, 1);
  if (errno != 0) {
    return Status(
        Status::Code::INTERNAL, "Failed to check if directory at " + path +
                                    ", errno:" + strerror(errno));
  }
  *is_dir = blobs.blobs.size() > 0;

  return Status::Success;
};

Status
ASFileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  as::blob_client_wrapper bc(client_);
  std::string container, object_path;
  RETURN_IF_ERROR(ParsePath(path, &container, &object_path));
  using namespace azure::storage_lite;
  std::ostringstream out_stream;
  bc.download_blob_to_stream(container, object_path, 0, 0, out_stream);
  if (errno != 0) {
    return Status(
        Status::Code::INTERNAL, "Failed to fetch file stream at " + path +
                                    ", errno:" + strerror(errno));
  }
  *contents = out_stream.str();

  return Status::Success;
}

Status
ASFileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = false;

  std::string container, object;
  RETURN_IF_ERROR(ParsePath(path, &container, &object));
  as::blob_client_wrapper bc(client_);
  auto blobs = bc.list_blobs_segmented(container, "/", "", object, 1);
  if (errno != 0) {
    return Status(
        Status::Code::INTERNAL, "Failed to check if file exists at " + path +
                                    ", errno:" + strerror(errno));
  }
  if (blobs.blobs.size() > 0) {
    *exists = true;
  }
  return Status::Success;
}

Status
ASFileSystem::DownloadFolder(
    const std::string& container, const std::string& path,
    const std::string& dest)
{
  as::blob_client_wrapper bc(client_);
  auto func = [&](const as::list_blobs_segmented_item& item,
                  const std::string& dir) {
    auto local_path = JoinPath({dest, dir});
    auto blob_path = JoinPath({path, dir});
    if (item.is_directory) {
      int status = mkdir(
          const_cast<char*>(local_path.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);
      if (status == -1) {
        return Status(
            Status::Code::INTERNAL,
            "Failed to create local folder: " + local_path +
                ", errno:" + strerror(errno));
      }
      auto ret = DownloadFolder(container, blob_path, local_path);
      if (!ret.IsOk()) {
        return ret;
      }
    } else {
      time_t last_modified;
      bc.download_blob_to_file(container, blob_path, local_path, last_modified);
      if (errno != 0) {
        return Status(
            Status::Code::INTERNAL, "Failed to download file at " + blob_path +
                                        ", errno:" + strerror(errno));
      }
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

  std::string folder_template = "/tmp/folderXXXXXX";
  char* tmp_folder = mkdtemp(const_cast<char*>(folder_template.c_str()));
  if (tmp_folder == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + folder_template +
            ", errno:" + strerror(errno));
  }
  localized->reset(new LocalizedPath(path, tmp_folder));

  std::string dest(folder_template);

  as::blob_client_wrapper bc(client_);

  std::string container, object;
  RETURN_IF_ERROR(ParsePath(path, &container, &object));
  return DownloadFolder(container, object, dest);
}

Status
ASFileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  std::stringstream ss(contents);
  std::istream is(ss.rdbuf());
  std::string container, object;
  RETURN_IF_ERROR(ParsePath(path, &container, &object));
  std::vector<std::pair<std::string, std::string>> metadata;
  auto ret =
      client_->upload_block_blob_from_stream(container, object, is, metadata)
          .get();
  if (!ret.success()) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to upload blob, Error: " + ret.error().code + ", " +
            ret.error().code_name);
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
ASFileSystem::MakeTemporaryDirectory(std::string* temp_dir)
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