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

#include <google/cloud/storage/client.h>

#include "common.h"

namespace triton { namespace core {

namespace gcs = google::cloud::storage;

struct GCSCredential {
  std::string path_;

  GCSCredential();  // from env var
  GCSCredential(triton::common::TritonJson::Value& cred_json);
};

GCSCredential::GCSCredential()
{
  const char* path = std::getenv("GOOGLE_APPLICATION_CREDENTIALS");
  path_ = (path != nullptr ? std::string(path) : "");
}

GCSCredential::GCSCredential(triton::common::TritonJson::Value& cred_json)
{
  cred_json.AsString(&path_);
}

class GCSFileSystem : public FileSystem {
 public:
  GCSFileSystem(const GCSCredential& gs_cred);
  // unify with S3/azure interface
  GCSFileSystem(const std::string& path, const GCSCredential& gs_cred)
      : GCSFileSystem(gs_cred)
  {
  }
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
      const std::string& path, std::string* bucket, std::string* object);
  Status MetaDataExists(
      const std::string path, bool* exists,
      google::cloud::StatusOr<gcs::ObjectMetadata>* metadata);

  std::unique_ptr<gcs::Client> client_;
};

GCSFileSystem::GCSFileSystem(const GCSCredential& gs_cred)
{
  google::cloud::Options options;
  auto creds = gcs::oauth2::CreateServiceAccountCredentialsFromJsonFilePath(
      gs_cred.path_);
  if (creds) {
    options.set<gcs::Oauth2CredentialsOption>(*creds);  // json credential
  } else {
    auto creds = gcs::oauth2::CreateComputeEngineCredentials();
    if (creds->AuthorizationHeader()) {
      options.set<gcs::Oauth2CredentialsOption>(creds);  // metadata service
    } else {
      options.set<gcs::Oauth2CredentialsOption>(
          gcs::oauth2::CreateAnonymousCredentials());  // no credential
    }
  }
  client_ = std::make_unique<gcs::Client>(options);
}

Status
GCSFileSystem::CheckClient()
{
  if (!client_) {
    return Status(
        Status::Code::INTERNAL,
        "Unable to create GCS client. Check account credentials.");
  }
  return Status::Success;
}

Status
GCSFileSystem::ParsePath(
    const std::string& path, std::string* bucket, std::string* object)
{
  // Get the bucket name and the object path. Return error if input is malformed
  int bucket_start = path.find("gs://") + strlen("gs://");
  int bucket_end = path.find("/", bucket_start);

  // If there isn't a second slash, the address has only the bucket
  if (bucket_end > bucket_start) {
    *bucket = path.substr(bucket_start, bucket_end - bucket_start);
    *object = path.substr(bucket_end + 1);
  } else {
    *bucket = path.substr(bucket_start);
    *object = "";
  }

  if (bucket->empty()) {
    return Status(
        Status::Code::INTERNAL, "No bucket name found in path: " + path);
  }

  return Status::Success;
}

Status
GCSFileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = false;

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Make a request for metadata and check the response
  google::cloud::StatusOr<gcs::ObjectMetadata> object_metadata =
      client_->GetObjectMetadata(bucket, object);

  if (object_metadata) {
    *exists = true;
    return Status::Success;
  }

  // GCS doesn't make objects for directories, so it could still be a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  *exists = is_dir;

  return Status::Success;
}

Status
GCSFileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;
  std::string bucket, object_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object_path));

  // Check if the bucket exists
  google::cloud::StatusOr<gcs::BucketMetadata> bucket_metadata =
      client_->GetBucketMetadata(bucket);

  if (!bucket_metadata) {
    return Status(
        Status::Code::INTERNAL, "Could not get MetaData for bucket with name " +
                                    bucket + " : " +
                                    bucket_metadata.status().message());
  }

  // Root case - bucket exists and object path is empty
  if (object_path.empty()) {
    *is_dir = true;
    return Status::Success;
  }

  // Check whether it has children. If at least one child, it is a directory
  for (auto&& object_metadata :
       client_->ListObjects(bucket, gcs::Prefix(AppendSlash(object_path)))) {
    if (object_metadata) {
      *is_dir = true;
      break;
    }
  }
  return Status::Success;
}

Status
GCSFileSystem::FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  // We don't need to worry about the case when this is a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (is_dir) {
    *mtime_ns = 0;
    return Status::Success;
  }

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Otherwise check the object metadata for update time
  google::cloud::StatusOr<gcs::ObjectMetadata> object_metadata =
      client_->GetObjectMetadata(bucket, object);

  if (!object_metadata) {
    return Status(
        Status::Code::INTERNAL, "Failed to get metadata for " + object + " : " +
                                    object_metadata.status().message());
  }

  // Get duration from time point with respect to object clock
  auto update_time = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                         object_metadata->updated())
                         .time_since_epoch()
                         .count();

  *mtime_ns = update_time;
  return Status::Success;
}

Status
GCSFileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  std::string bucket, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  // Append a slash to make it easier to list contents
  std::string full_dir = AppendSlash(dir_path);

  // Get objects with prefix equal to full directory path
  for (auto&& object_metadata :
       client_->ListObjects(bucket, gcs::Prefix(full_dir))) {
    if (!object_metadata) {
      return Status(
          Status::Code::INTERNAL, "Could not list contents of directory at " +
                                      path + " : " +
                                      object_metadata.status().message());
    }

    // In the case of empty directories, the directory itself will appear here
    if (object_metadata->name() == full_dir) {
      continue;
    }

    // We have to make sure that subdirectory contents do not appear here
    std::string name = object_metadata->name();
    int item_start = name.find(full_dir) + full_dir.size();
    // GCS response prepends parent directory name
    int item_end = name.find("/", item_start);

    // Let set take care of subdirectory contents
    std::string item = name.substr(item_start, item_end - item_start);
    contents->insert(item);

    // Fail-safe check to ensure the item name is not empty
    if (item.empty()) {
      return Status(
          Status::Code::INTERNAL,
          "Cannot handle item with empty name at " + path);
    }
  }
  return Status::Success;
}

Status
GCSFileSystem::GetDirectorySubdirs(
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
GCSFileSystem::GetDirectoryFiles(
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
GCSFileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));

  if (!exists) {
    return Status(Status::Code::INTERNAL, "File does not exist at " + path);
  }

  std::string bucket, object;
  ParsePath(path, &bucket, &object);

  gcs::ObjectReadStream stream = client_->ReadObject(bucket, object);

  if (!stream) {
    return Status(
        Status::Code::INTERNAL, "Failed to open object read stream for " +
                                    path + " : " + stream.status().message());
  }

  std::string data = "";
  char c;
  while (stream.get(c)) {
    data += c;
  }

  *contents = data;

  return Status::Success;
}

Status
GCSFileSystem::LocalizePath(
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
        "GCS file localization not yet implemented " + path);
  }

  // Create a local directory for GCS model store.
  std::string env_mount_dir = GetEnvironmentVariableOrDefault(
      "TRITON_GCS_MOUNT_DIRECTORY", kDefaultMountDirectory);
  std::string tmp_folder;
  RETURN_IF_ERROR(triton::core::MakeTemporaryDirectory(
      FileSystemType::LOCAL, env_mount_dir, &tmp_folder));

  localized->reset(new LocalizedPath(path, tmp_folder));

  std::set<std::string> contents, filenames;
  RETURN_IF_ERROR(GetDirectoryContents(path, &filenames));
  for (auto itr = filenames.begin(); itr != filenames.end(); ++itr) {
    contents.insert(JoinPath({path, *itr}));
  }

  while (contents.size() != 0) {
    std::set<std::string> tmp_contents = contents;
    contents.clear();
    for (auto iter = tmp_contents.begin(); iter != tmp_contents.end(); ++iter) {
      bool is_subdir;
      std::string gcs_fpath = *iter;
      std::string gcs_removed_path = gcs_fpath.substr(path.size());
      std::string local_fpath =
          JoinPath({(*localized)->Path(), gcs_removed_path});
      RETURN_IF_ERROR(IsDirectory(gcs_fpath, &is_subdir));
      if (is_subdir) {
        // Create local mirror of sub-directories
#ifdef _WIN32
        int status = mkdir(const_cast<char*>(local_fpath.c_str()));
#else
        int status = mkdir(
            const_cast<char*>(local_fpath.c_str()),
            S_IRUSR | S_IWUSR | S_IXUSR);
#endif
        if (status == -1) {
          return Status(
              Status::Code::INTERNAL,
              "Failed to create local folder: " + local_fpath +
                  ", errno:" + strerror(errno));
        }

        // Add sub-directories and deeper files to contents
        std::set<std::string> subdir_contents;
        RETURN_IF_ERROR(GetDirectoryContents(gcs_fpath, &subdir_contents));
        for (auto itr = subdir_contents.begin(); itr != subdir_contents.end();
             ++itr) {
          contents.insert(JoinPath({gcs_fpath, *itr}));
        }
      } else {
        // Create local copy of file
        std::string file_bucket, file_object;
        RETURN_IF_ERROR(ParsePath(gcs_fpath, &file_bucket, &file_object));

        // Send a request to read the object
        gcs::ObjectReadStream filestream =
            client_->ReadObject(file_bucket, file_object);
        if (!filestream) {
          return Status(
              Status::Code::INTERNAL, "Failed to get object at " + *iter +
                                          " : " +
                                          filestream.status().message());
        }

        std::string gcs_removed_path = (*iter).substr(path.size());
        std::string local_file_path =
            JoinPath({(*localized)->Path(), gcs_removed_path});
        std::ofstream output_file(local_file_path.c_str(), std::ios::binary);
        output_file << filestream.rdbuf();
        output_file.close();
      }
    }
  }

  return Status::Success;
}

Status
GCSFileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Write text file operation not yet implemented " + path);
}

Status
GCSFileSystem::WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Write text file operation not yet implemented " + path);
}

Status
GCSFileSystem::MakeDirectory(const std::string& dir, const bool recursive)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Make temporary directory operation not yet implemented");
}

Status
GCSFileSystem::MakeTemporaryDirectory(
    std::string dir_path, std::string* temp_dir)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Make temporary directory operation not yet implemented");
}

Status
GCSFileSystem::DeletePath(const std::string& path)
{
  return Status(
      Status::Code::UNSUPPORTED, "Delete path operation not yet implemented");
}

}}  // namespace triton::core
