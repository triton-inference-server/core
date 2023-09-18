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

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>

#include "common.h"

// [FIXME: DLIS-4973]
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/ListObjectsV2Result.h>

namespace triton { namespace core {

namespace s3 = Aws::S3;

// Override the default S3 Curl initialization for disabling HTTP/2 on s3.
// Remove once s3 fully supports HTTP/2 [FIXME: DLIS-4973].
// Reference:
// https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/cpp/example_code/s3/list_buckets_disabling_dns_cache.cpp
static const char S3_ALLOCATION_TAG[] = "OverrideDefaultHttpClient";
class S3CurlHttpClient : public Aws::Http::CurlHttpClient {
 public:
  explicit S3CurlHttpClient(
      const Aws::Client::ClientConfiguration& client_config)
      : Aws::Http::CurlHttpClient(client_config)
  {
  }

 protected:
  void OverrideOptionsOnConnectionHandle(CURL* connectionHandle) const override
  {
    curl_easy_setopt(
        connectionHandle, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
  }
};
class S3HttpClientFactory : public Aws::Http::HttpClientFactory {
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& client_config) const override
  {
    return Aws::MakeShared<S3CurlHttpClient>(S3_ALLOCATION_TAG, client_config);
  }
  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::String& uri, Aws::Http::HttpMethod method,
      const Aws::IOStreamFactory& stream_factory) const override
  {
    return CreateHttpRequest(Aws::Http::URI(uri), method, stream_factory);
  }
  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::Http::URI& uri, Aws::Http::HttpMethod method,
      const Aws::IOStreamFactory& stream_factory) const override
  {
    auto req = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
        S3_ALLOCATION_TAG, uri, method);
    req->SetResponseStreamFactory(stream_factory);
    return req;
  }
  void InitStaticState() override { S3CurlHttpClient::InitGlobalState(); }
  void CleanupStaticState() override { S3CurlHttpClient::CleanupGlobalState(); }
};

struct S3Credential {
  std::string secret_key_;
  std::string key_id_;
  std::string region_;
  std::string session_token_;
  std::string profile_name_;

  S3Credential();  // from env var
  S3Credential(triton::common::TritonJson::Value& cred_json);
};

S3Credential::S3Credential()
{
  const auto to_str = [](const char* s) -> std::string {
    return (s != nullptr ? std::string(s) : "");
  };
  const char* secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");
  const char* key_id = std::getenv("AWS_ACCESS_KEY_ID");
  const char* region = std::getenv("AWS_DEFAULT_REGION");
  const char* session_token = std::getenv("AWS_SESSION_TOKEN");
  const char* profile = std::getenv("AWS_PROFILE");
  secret_key_ = to_str(secret_key);
  key_id_ = to_str(key_id);
  region_ = to_str(region);
  session_token_ = to_str(session_token);
  profile_name_ = to_str(profile);
}

S3Credential::S3Credential(triton::common::TritonJson::Value& cred_json)
{
  triton::common::TritonJson::Value secret_key_json, key_id_json, region_json,
      session_token_json, profile_json;
  if (cred_json.Find("secret_key", &secret_key_json))
    secret_key_json.AsString(&secret_key_);
  if (cred_json.Find("key_id", &key_id_json))
    key_id_json.AsString(&key_id_);
  if (cred_json.Find("region", &region_json))
    region_json.AsString(&region_);
  if (cred_json.Find("session_token", &session_token_json))
    session_token_json.AsString(&session_token_);
  if (cred_json.Find("profile", &profile_json))
    profile_json.AsString(&profile_name_);
}

class S3FileSystem : public FileSystem {
 public:
  S3FileSystem(const std::string& s3_path, const S3Credential& s3_cred);
  Status CheckClient(const std::string& s3_path);

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
  Status CleanPath(const std::string& s3_path, std::string* clean_path);
  std::unique_ptr<s3::S3Client> client_;  // init after Aws::InitAPI is called
  re2::RE2 s3_regex_;
};

Status
S3FileSystem::ParsePath(
    const std::string& path, std::string* bucket, std::string* object)
{
  // Cleanup extra slashes
  std::string clean_path;
  RETURN_IF_ERROR(CleanPath(path, &clean_path));

  // Get the bucket name and the object path. Return error if path is malformed
  std::string protocol, host_name, host_port;
  if (!RE2::FullMatch(
          clean_path, s3_regex_, &protocol, &host_name, &host_port, bucket,
          object)) {
    int bucket_start = clean_path.find("s3://") + strlen("s3://");
    int bucket_end = clean_path.find("/", bucket_start);

    // If there isn't a slash, the address has only the bucket
    if (bucket_end > bucket_start) {
      *bucket = clean_path.substr(bucket_start, bucket_end - bucket_start);
      *object = clean_path.substr(bucket_end + 1);
    } else {
      *bucket = clean_path.substr(bucket_start);
      *object = "";
    }
  } else {
    // Erase leading '/' that is left behind in object name
    if ((*object)[0] == '/') {
      object->erase(0, 1);
    }
  }

  if (bucket->empty()) {
    return Status(
        Status::Code::INTERNAL, "No bucket name found in path: " + path);
  }

  return Status::Success;
}

Status
S3FileSystem::CleanPath(const std::string& s3_path, std::string* clean_path)
{
  // Must handle paths with s3 prefix
  size_t start = s3_path.find("s3://");
  std::string path = "";
  if (start != std::string::npos) {
    path = s3_path.substr(start + strlen("s3://"));
    *clean_path = "s3://";
  } else {
    path = s3_path;
    *clean_path = "";
  }

  // Must handle paths with https:// or http:// prefix
  size_t https_start = path.find("https://");
  if (https_start != std::string::npos) {
    path = path.substr(https_start + strlen("https://"));
    *clean_path += "https://";
  } else {
    size_t http_start = path.find("http://");
    if (http_start != std::string::npos) {
      path = path.substr(http_start + strlen("http://"));
      *clean_path += "http://";
    }
  }

  // Remove trailing slashes
  size_t rtrim_length = path.find_last_not_of('/');
  if (rtrim_length == std::string::npos) {
    return Status(
        Status::Code::INVALID_ARG, "Invalid bucket name: '" + path + "'");
  }

  // Remove leading slashes
  size_t ltrim_length = path.find_first_not_of('/');
  if (ltrim_length == std::string::npos) {
    return Status(
        Status::Code::INVALID_ARG, "Invalid bucket name: '" + path + "'");
  }

  // Remove extra internal slashes
  std::string true_path = path.substr(ltrim_length, rtrim_length + 1);
  std::vector<int> slash_locations;
  bool previous_slash = false;
  for (size_t i = 0; i < true_path.size(); i++) {
    if (true_path[i] == '/') {
      if (!previous_slash) {
        *clean_path += true_path[i];
      }
      previous_slash = true;
    } else {
      *clean_path += true_path[i];
      previous_slash = false;
    }
  }

  return Status::Success;
}

S3FileSystem::S3FileSystem(
    const std::string& s3_path, const S3Credential& s3_cred)
    : s3_regex_(
          "s3://(http://|https://|)([0-9a-zA-Z\\-.]+):([0-9]+)/"
          "([0-9a-z.\\-]+)(((/[0-9a-zA-Z.\\-_]+)*)?)")
{
  // init aws api if not already
  Aws::SDKOptions options;
  static std::once_flag onceFlag;
  std::call_once(onceFlag, [&options] { Aws::InitAPI(options); });

  // [FIXME: DLIS-4973]
  Aws::Http::SetHttpClientFactory(
      Aws::MakeShared<S3HttpClientFactory>(S3_ALLOCATION_TAG));

  Aws::Client::ClientConfiguration config;
  Aws::Auth::AWSCredentials credentials;

  // check vars for S3 credentials -> aws profile -> default
  if (!s3_cred.secret_key_.empty() && !s3_cred.key_id_.empty()) {
    credentials.SetAWSAccessKeyId(s3_cred.key_id_.c_str());
    credentials.SetAWSSecretKey(s3_cred.secret_key_.c_str());
    if (!s3_cred.session_token_.empty()) {
      credentials.SetSessionToken(s3_cred.session_token_.c_str());
    }
    config = Aws::Client::ClientConfiguration();
    if (!s3_cred.region_.empty()) {
      config.region = s3_cred.region_.c_str();
    }
  } else if (!s3_cred.profile_name_.empty()) {
    config = Aws::Client::ClientConfiguration(s3_cred.profile_name_.c_str());
  } else {
    config = Aws::Client::ClientConfiguration("default");
  }

  // Cleanup extra slashes
  std::string clean_path;
  LOG_STATUS_ERROR(CleanPath(s3_path, &clean_path), "failed to parse S3 path");

  std::string protocol, host_name, host_port, bucket, object;
  if (RE2::FullMatch(
          clean_path, s3_regex_, &protocol, &host_name, &host_port, &bucket,
          &object)) {
    config.endpointOverride = Aws::String(host_name + ":" + host_port);
    if (protocol == "https://") {
      config.scheme = Aws::Http::Scheme::HTTPS;
    } else {
      config.scheme = Aws::Http::Scheme::HTTP;
    }
  }

  if (!s3_cred.secret_key_.empty() && !s3_cred.key_id_.empty()) {
    client_ = std::make_unique<s3::S3Client>(
        credentials, config,
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        /*useVirtualAdressing*/ false);
  } else {
    client_ = std::make_unique<s3::S3Client>(
        config, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        /*useVirtualAdressing*/ false);
  }
}

Status
S3FileSystem::CheckClient(const std::string& s3_path)
{
  std::string bucket, object_path;
  RETURN_IF_ERROR(ParsePath(s3_path, &bucket, &object_path));
  // check if can connect to the bucket
  s3::Model::HeadBucketRequest head_request;
  head_request.WithBucket(bucket.c_str());
  auto head_object_outcome = client_->HeadBucket(head_request);
  if (!head_object_outcome.IsSuccess()) {
    auto err = head_object_outcome.GetError();
    return Status(
        Status::Code::INTERNAL,
        "Unable to create S3 filesystem client. Check account credentials. "
        "Exception: '" +
            err.GetExceptionName() + "' Message: '" + err.GetMessage() + "'");
  }
  return Status::Success;
}

Status
S3FileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = false;

  // S3 doesn't make objects for directories, so it could still be a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (is_dir) {
    *exists = is_dir;
    return Status::Success;
  }

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Construct request for object metadata
  s3::Model::HeadObjectRequest head_request;
  head_request.SetBucket(bucket.c_str());
  head_request.SetKey(object.c_str());

  auto head_object_outcome = client_->HeadObject(head_request);
  if (!head_object_outcome.IsSuccess()) {
    if (head_object_outcome.GetError().GetErrorType() !=
        s3::S3Errors::RESOURCE_NOT_FOUND) {
      return Status(
          Status::Code::INTERNAL,
          "Could not get MetaData for object at " + path +
              " due to exception: " +
              head_object_outcome.GetError().GetExceptionName() +
              ", error message: " +
              head_object_outcome.GetError().GetMessage());
    }
  } else {
    *exists = true;
  }

  return Status::Success;
}

Status
S3FileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;
  std::string bucket, object_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object_path));

  // Check if the bucket exists
  s3::Model::HeadBucketRequest head_request;
  head_request.WithBucket(bucket.c_str());

  auto head_bucket_outcome = client_->HeadBucket(head_request);
  if (!head_bucket_outcome.IsSuccess()) {
    return Status(
        Status::Code::INTERNAL,
        "Could not get MetaData for bucket with name " + bucket +
            " due to exception: " +
            head_bucket_outcome.GetError().GetExceptionName() +
            ", error message: " + head_bucket_outcome.GetError().GetMessage());
  }

  // Root case - bucket exists and object path is empty
  if (object_path.empty()) {
    *is_dir = true;
    return Status::Success;
  }

  // List the objects in the bucket
  s3::Model::ListObjectsV2Request list_objects_request;
  list_objects_request.SetBucket(bucket.c_str());
  list_objects_request.SetPrefix(AppendSlash(object_path).c_str());
  auto list_objects_outcome = client_->ListObjectsV2(list_objects_request);

  if (list_objects_outcome.IsSuccess()) {
    *is_dir = !list_objects_outcome.GetResult().GetContents().empty();
  } else {
    return Status(
        Status::Code::INTERNAL,
        "Failed to list objects with prefix " + path + " due to exception: " +
            list_objects_outcome.GetError().GetExceptionName() +
            ", error message: " + list_objects_outcome.GetError().GetMessage());
  }
  return Status::Success;
}

Status
S3FileSystem::FileModificationTime(const std::string& path, int64_t* mtime_ns)
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

  // Send a request for the objects metadata
  s3::Model::HeadObjectRequest head_request;
  head_request.SetBucket(bucket.c_str());
  head_request.SetKey(object.c_str());

  // If request succeeds, copy over the modification time
  auto head_object_outcome = client_->HeadObject(head_request);
  if (head_object_outcome.IsSuccess()) {
    *mtime_ns = head_object_outcome.GetResult().GetLastModified().Millis() *
                NANOS_PER_MILLIS;
  } else {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get modification time for object at " + path +
            " due to exception: " +
            head_object_outcome.GetError().GetExceptionName() +
            ", error message: " + head_object_outcome.GetError().GetMessage());
  }
  return Status::Success;
}

Status
S3FileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  // Parse bucket and dir_path
  std::string bucket, dir_path, full_dir;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  std::string true_path = "s3://" + bucket + '/' + dir_path;

  // Capture the full path to facilitate content listing
  full_dir = AppendSlash(dir_path);

  // Issue request for objects with prefix
  s3::Model::ListObjectsV2Request objects_request;
  objects_request.SetBucket(bucket.c_str());
  objects_request.SetPrefix(full_dir.c_str());

  bool done_listing = false;
  while (!done_listing) {
    auto list_objects_outcome = client_->ListObjectsV2(objects_request);

    if (!list_objects_outcome.IsSuccess()) {
      return Status(
          Status::Code::INTERNAL,
          "Could not list contents of directory at " + true_path +
              " due to exception: " +
              list_objects_outcome.GetError().GetExceptionName() +
              ", error message: " +
              list_objects_outcome.GetError().GetMessage());
    }
    const auto& list_objects_result = list_objects_outcome.GetResult();
    for (const auto& s3_object : list_objects_result.GetContents()) {
      // In the case of empty directories, the directory itself will appear
      // here
      if (s3_object.GetKey().c_str() == full_dir) {
        continue;
      }

      // We have to make sure that subdirectory contents do not appear here
      std::string name(s3_object.GetKey().c_str());
      int item_start = name.find(full_dir) + full_dir.size();
      // S3 response prepends parent directory name
      int item_end = name.find("/", item_start);

      // Let set take care of subdirectory contents
      std::string item = name.substr(item_start, item_end - item_start);
      contents->insert(item);

      // Fail-safe check to ensure the item name is not empty
      if (item.empty()) {
        return Status(
            Status::Code::INTERNAL,
            "Cannot handle item with empty name at " + true_path);
      }
    }
    // If there are more pages to retrieve, set the marker to the next page.
    if (list_objects_result.GetIsTruncated()) {
      objects_request.SetContinuationToken(
          list_objects_result.GetNextContinuationToken());
    } else {
      done_listing = true;
    }
  }
  return Status::Success;
}

Status
S3FileSystem::GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs)
{
  // Parse bucket and dir_path
  std::string bucket, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  std::string true_path = "s3://" + bucket + '/' + dir_path;

  RETURN_IF_ERROR(GetDirectoryContents(true_path, subdirs));

  // Erase non-directory entries...
  for (auto iter = subdirs->begin(); iter != subdirs->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({true_path, *iter}), &is_dir));
    if (!is_dir) {
      iter = subdirs->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}
Status
S3FileSystem::GetDirectoryFiles(
    const std::string& path, std::set<std::string>* files)
{
  // Parse bucket and dir_path
  std::string bucket, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  std::string true_path = "s3://" + bucket + '/' + dir_path;
  RETURN_IF_ERROR(GetDirectoryContents(true_path, files));

  // Erase directory entries...
  for (auto iter = files->begin(); iter != files->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({true_path, *iter}), &is_dir));
    if (is_dir) {
      iter = files->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
S3FileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));

  if (!exists) {
    return Status(Status::Code::INTERNAL, "File does not exist at " + path);
  }

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Send a request for the objects metadata
  s3::Model::GetObjectRequest object_request;
  object_request.SetBucket(bucket.c_str());
  object_request.SetKey(object.c_str());

  auto get_object_outcome = client_->GetObject(object_request);
  if (get_object_outcome.IsSuccess()) {
    auto& object_result = get_object_outcome.GetResultWithOwnership().GetBody();

    std::string data = "";
    char c;
    while (object_result.get(c)) {
      data += c;
    }

    *contents = data;
  } else {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get object at " + path + " due to exception: " +
            get_object_outcome.GetError().GetExceptionName() +
            ", error message: " + get_object_outcome.GetError().GetMessage());
  }

  return Status::Success;
}

Status
S3FileSystem::LocalizePath(
    const std::string& path, std::shared_ptr<LocalizedPath>* localized)
{
  // Check if the directory or file exists
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));
  if (!exists) {
    return Status(
        Status::Code::INTERNAL, "directory or file does not exist at " + path);
  }

  // Cleanup extra slashes
  std::string clean_path;
  RETURN_IF_ERROR(CleanPath(path, &clean_path));

  // Remove protocol and host name and port
  std::string effective_path, protocol, host_name, host_port, bucket, object;
  if (RE2::FullMatch(
          clean_path, s3_regex_, &protocol, &host_name, &host_port, &bucket,
          &object)) {
    effective_path = "s3://" + bucket + object;
  } else {
    effective_path = path;
  }

  // Create a local directory for AWS model store.
  // If ENV variable are not set, creates a temporary directory
  // under `/tmp` with the format: "folderXXXXXX".
  // Otherwise, will create a folder under specified directory with the same
  // format.
  std::string env_mount_dir = GetEnvironmentVariableOrDefault(
      "TRITON_AWS_MOUNT_DIRECTORY", kDefaultMountDirectory);
  std::string tmp_folder;
  RETURN_IF_ERROR(triton::core::MakeTemporaryDirectory(
      FileSystemType::LOCAL, env_mount_dir, &tmp_folder));

  // Specify contents to be downloaded
  std::set<std::string> contents;
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (is_dir) {
    // Set localized path
    localized->reset(new LocalizedPath(effective_path, tmp_folder));
    // Specify the entire directory to be downloaded
    std::set<std::string> filenames;
    RETURN_IF_ERROR(GetDirectoryContents(effective_path, &filenames));
    for (auto itr = filenames.begin(); itr != filenames.end(); ++itr) {
      contents.insert(JoinPath({effective_path, *itr}));
    }
  } else {
    // Set localized path
    std::string filename =
        effective_path.substr(effective_path.find_last_of('/') + 1);
    localized->reset(
        new LocalizedPath(effective_path, JoinPath({tmp_folder, filename})));
    // Specify only the file to be downloaded
    contents.insert(effective_path);
  }

  // Download all specified contents and nested contents
  while (contents.size() != 0) {
    std::set<std::string> tmp_contents = contents;
    contents.clear();
    for (auto iter = tmp_contents.begin(); iter != tmp_contents.end(); ++iter) {
      std::string s3_fpath = *iter;
      std::string s3_removed_path = s3_fpath.substr(effective_path.size());
      std::string local_fpath =
          s3_removed_path.empty()
              ? (*localized)->Path()
              : JoinPath({(*localized)->Path(), s3_removed_path});
      bool is_subdir;
      RETURN_IF_ERROR(IsDirectory(s3_fpath, &is_subdir));
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
        RETURN_IF_ERROR(GetDirectoryContents(s3_fpath, &subdir_contents));
        for (auto itr = subdir_contents.begin(); itr != subdir_contents.end();
             ++itr) {
          contents.insert(JoinPath({s3_fpath, *itr}));
        }
      } else {
        // Create local copy of file
        std::string file_bucket, file_object;
        RETURN_IF_ERROR(ParsePath(s3_fpath, &file_bucket, &file_object));

        s3::Model::GetObjectRequest object_request;
        object_request.SetBucket(file_bucket.c_str());
        object_request.SetKey(file_object.c_str());

        auto get_object_outcome = client_->GetObject(object_request);
        if (get_object_outcome.IsSuccess()) {
          auto& retrieved_file =
              get_object_outcome.GetResultWithOwnership().GetBody();
          std::ofstream output_file(local_fpath.c_str(), std::ios::binary);
          output_file << retrieved_file.rdbuf();
          output_file.close();
        } else {
          return Status(
              Status::Code::INTERNAL,
              "Failed to get object at " + s3_fpath + " due to exception: " +
                  get_object_outcome.GetError().GetExceptionName() +
                  ", error message: " +
                  get_object_outcome.GetError().GetMessage());
        }
      }
    }
  }

  return Status::Success;
}

Status
S3FileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Write text file operation not yet implemented " + path);
}

Status
S3FileSystem::WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Write text file operation not yet implemented " + path);
}

Status
S3FileSystem::MakeDirectory(const std::string& dir, const bool recursive)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Make directory operation not yet implemented");
}

Status
S3FileSystem::MakeTemporaryDirectory(
    std::string dir_path, std::string* temp_dir)
{
  return Status(
      Status::Code::UNSUPPORTED,
      "Make temporary directory operation not yet implemented");
}

Status
S3FileSystem::DeletePath(const std::string& path)
{
  return Status(
      Status::Code::UNSUPPORTED, "Delete path operation not yet implemented");
}

}}  // namespace triton::core
