// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_config_utils.h"

#include <sys/stat.h>

#include <fstream>
#include <string>

#include "constants.h"
#include "filesystem/api.h"
#include "gtest/gtest.h"

namespace tc = triton::core;

namespace {

// Helper to create a temporary model directory with a version subdirectory
// and an optional file inside it.
class TempModelDir {
 public:
  TempModelDir()
  {
    auto status =
        tc::MakeTemporaryDirectory(tc::FileSystemType::LOCAL, &root_path_);
    EXPECT_TRUE(status.IsOk()) << status.AsString();
  }

  ~TempModelDir()
  {
    // Best-effort cleanup
    std::string cmd = "rm -rf " + root_path_;
    (void)system(cmd.c_str());
  }

  // Create version subdir (e.g., "1") and optionally place a file in it.
  void AddVersionWithFile(
      const std::string& version, const std::string& filename)
  {
    std::string version_dir = tc::JoinPath({root_path_, version});
    mkdir(version_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (!filename.empty()) {
      std::ofstream f(tc::JoinPath({version_dir, filename}));
      f << "# placeholder";
    }
  }

  const std::string& Path() const { return root_path_; }

 private:
  std::string root_path_;
};

class AutoCompleteBackendFieldsTest : public ::testing::Test {};

// When backend is "python" and default_model_filename is empty and
// cc_model_filenames is empty, default_model_filename should be set to
// "model.py".
TEST_F(AutoCompleteBackendFieldsTest, PythonBackendSetsDefaultFilename)
{
  TempModelDir dir;
  dir.AddVersionWithFile("1", "model.py");

  inference::ModelConfig config;
  config.set_backend("python");
  // default_model_filename and cc_model_filenames are both empty

  auto status = tc::AutoCompleteBackendFields("test_model", dir.Path(), &config);
  ASSERT_TRUE(status.IsOk()) << status.AsString();
  EXPECT_EQ(config.default_model_filename(), "model.py");
}

// When backend is "python" and default_model_filename is empty but
// cc_model_filenames is populated, default_model_filename should NOT be
// auto-filled to "model.py".
TEST_F(
    AutoCompleteBackendFieldsTest,
    PythonBackendSkipsDefaultFilenameWhenCcModelFilenamesSet)
{
  TempModelDir dir;
  dir.AddVersionWithFile("1", "custom_model.py");

  inference::ModelConfig config;
  config.set_backend("python");
  (*config.mutable_cc_model_filenames())["gpu"] = "custom_model.py";
  // default_model_filename is empty, cc_model_filenames is set

  auto status = tc::AutoCompleteBackendFields("test_model", dir.Path(), &config);
  ASSERT_TRUE(status.IsOk()) << status.AsString();
  EXPECT_EQ(config.default_model_filename(), "")
      << "default_model_filename should remain empty when cc_model_filenames "
         "is set";
}

// When backend is "python" and default_model_filename is already set,
// it should be preserved regardless of cc_model_filenames.
TEST_F(
    AutoCompleteBackendFieldsTest,
    PythonBackendPreservesExplicitDefaultFilename)
{
  TempModelDir dir;
  dir.AddVersionWithFile("1", "my_model.py");

  inference::ModelConfig config;
  config.set_backend("python");
  config.set_default_model_filename("my_model.py");

  auto status = tc::AutoCompleteBackendFields("test_model", dir.Path(), &config);
  ASSERT_TRUE(status.IsOk()) << status.AsString();
  EXPECT_EQ(config.default_model_filename(), "my_model.py");
}

// When backend is empty but version dir contains model.py, backend should be
// auto-detected as "python" and default_model_filename set to "model.py".
TEST_F(AutoCompleteBackendFieldsTest, AutoDetectPythonBackendFromModelFile)
{
  TempModelDir dir;
  dir.AddVersionWithFile("1", "model.py");

  inference::ModelConfig config;
  // backend, platform, default_model_filename all empty

  auto status = tc::AutoCompleteBackendFields("test_model", dir.Path(), &config);
  ASSERT_TRUE(status.IsOk()) << status.AsString();
  EXPECT_EQ(config.backend(), "python");
  EXPECT_EQ(config.default_model_filename(), "model.py");
}

}  // namespace
