// Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "triton/core/tritonserver.h"

namespace {

class FileSystemTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    char directory[] = "/tmp/triton-credentials-XXXXXX";
    const char* result = mkdtemp(directory);
    ASSERT_NE(result, nullptr);
    directory_ = result;
  }

  void TearDown() override { std::filesystem::remove_all(directory_); }

  std::string directory_;
};

TEST_F(FileSystemTest, EqualLengthCloudCredentialNames)
{
  // Each child starts with an uncached FileSystemManager. An unmatched path
  // exercises credential loading and sorting without creating an S3 client.
  const std::string path = directory_ + "/credentials.json";
  for (const size_t count : {0, 1, 16, 17, 100, 1000}) {
    SCOPED_TRACE(count);
    std::ostringstream credentials;
    credentials << "{\"s3\": {";
    for (size_t i = 0; i < count; ++i) {
      if (i != 0) {
        credentials << ",";
      }
      credentials << "\"s3://bucket/model" << std::setw(4) << std::setfill('0')
                  << i << "\": {\"key_id\": \"k\", \"secret_key\": \"s\", "
                  << "\"region\": \"us-east-1\"}";
    }
    credentials << "}}";
    {
      std::ofstream file(path);
      file << credentials.str();
      ASSERT_TRUE(file.good());
    }

    ASSERT_EXIT(
        {
          if (setenv("TRITON_CLOUD_CREDENTIAL_PATH", path.c_str(), 1) != 0) {
            std::exit(2);
          }
          TRITONSERVER_ServerOptions* options = nullptr;
          if (TRITONSERVER_ServerOptionsNew(&options) != nullptr) {
            std::exit(3);
          }
          if (TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                  options, "s3://unmatched/model") != nullptr) {
            std::exit(4);
          }
          TRITONSERVER_Server* server = nullptr;
          TRITONSERVER_Error* error = TRITONSERVER_ServerNew(&server, options);
          const bool expected_error =
              (error != nullptr) &&
              (std::string(TRITONSERVER_ErrorMessage(error))
                   .find("Cannot match credential for path  "
                         "s3://unmatched/model") != std::string::npos);
          TRITONSERVER_ErrorDelete(error);
          TRITONSERVER_ServerOptionsDelete(options);
          if (server != nullptr) {
            TRITONSERVER_ServerDelete(server);
          }
          std::exit(expected_error ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
  }
}

}  // namespace
