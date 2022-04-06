// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <thread>
#include "gtest/gtest.h"
#include "triton/core/tritonserver.h"

namespace {

#define FAIL_TEST_IF_ERR(X, MSG)                                              \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ == nullptr))                                           \
        << "error: " << (MSG) << ": "                                         \
        << TRITONSERVER_ErrorCodeString(err__.get()) << " - "                 \
        << TRITONSERVER_ErrorMessage(err__.get());                            \
  } while (false)

#define FAIL_TEST_IF_NOT_ERR(X, CODE, ERR_MSG, MSG)                           \
  do {                                                                        \
    std::shared_ptr<TRITONSERVER_Error> err__((X), TRITONSERVER_ErrorDelete); \
    ASSERT_TRUE((err__ != nullptr)) << "expected error on: " << (MSG);        \
    if (err__ != nullptr) {                                                   \
      EXPECT_EQ(TRITONSERVER_ErrorCode(err__.get()), (CODE)) << (MSG);        \
      EXPECT_STREQ(TRITONSERVER_ErrorMessage(err__.get()), (ERR_MSG))         \
          << (MSG);                                                           \
    }                                                                         \
  } while (false)

// Test Fixture, this test suit expects the current direcotry to
// have the following file structure:
//  - empty_models (empty directory)
//  - models_0 (contain model directory "model_0")
//  - models_1 (contain model directories "model_0", "model_1")
class RegisterApiTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Create running server object.
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsNew(&server_options),
        "creating server options");
    // Triton expects at least one model repository is set at start, set to
    // an empty repository set ModelControlMode to EXPLICIT to avoid attempting
    // to load models.
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(
            server_options, "empty_models"),
        "setting model repository path");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelControlMode(
            server_options, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
        "setting model control mode");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerNew(&server_, server_options), "creating server");
    FAIL_TEST_IF_ERR(
        TRITONSERVER_ServerOptionsDelete(server_options),
        "deleting server options");
    ASSERT_TRUE(server_ != nullptr) << "server not created";
    bool live = false;
    for (int i = 10; ((i > 0) && !live); --i) {
      FAIL_TEST_IF_ERR(
          TRITONSERVER_ServerIsLive(server_, &live), "Is server live");
    }
    ASSERT_TRUE(live) << "server not live";
  }

  void TearDown() override
  {
    FAIL_TEST_IF_ERR(TRITONSERVER_ServerDelete(server_), "deleting server");
  }

  TRITONSERVER_Server* server_ = nullptr;
};

TEST_F(RegisterApiTest, Register)
{
  // Request to load "model_0" which should fail
  FAIL_TEST_IF_NOT_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_0"),
      TRITONSERVER_ERROR_INTERNAL,
      "failed to load 'model_0', failed to poll from model repository",
      "loading model 'model_0'");

  // Registering a repository "models_0" where contains "model_0"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", nullptr, 0),
      "registering model repository 'models_0'");
  // Request to load "model_0"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_0"),
      "loading model 'model_0'");
}

TEST_F(RegisterApiTest, RegisterWithMap)
{
  // Registering a repository "models_0" where contains "model_0", but with
  // different name mapping
  const char* override_name = "name_0";
  std::shared_ptr<TRITONSERVER_Parameter> managed_param(
      TRITONSERVER_ParameterNew(
          "model_0", TRITONSERVER_PARAMETER_STRING, override_name),
      TRITONSERVER_ParameterDelete);
  ASSERT_TRUE(managed_param != nullptr) << "failed to create name mapping pair";
  std::vector<const TRITONSERVER_Parameter*> name_map{managed_param.get()};

  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", name_map.data(), name_map.size()),
      "registering model repository 'models_0'");

  // Request to load "model_0" which should fail
  FAIL_TEST_IF_NOT_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_0"),
      TRITONSERVER_ERROR_INTERNAL,
      "failed to load 'model_0', failed to poll from model repository",
      "loading model 'model_0'");
  // Request to load "name_0"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "name_0"),
      "loading model 'name_0'");
}

TEST_F(RegisterApiTest, RegisterTwice)
{
  // Registering a startup repository
  FAIL_TEST_IF_NOT_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "empty_models", nullptr, 0),
      TRITONSERVER_ERROR_ALREADY_EXISTS,
      "model repository 'empty_models' has been registered",
      "registering model repository 'empty_models'");
}

TEST_F(RegisterApiTest, RegisterTwice2)
{
  // Registering the same repository twice
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", nullptr, 0),
      "registering model repository 'models_0'");

  FAIL_TEST_IF_NOT_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", nullptr, 0),
      TRITONSERVER_ERROR_ALREADY_EXISTS,
      "model repository 'models_0' has been registered",
      "registering model repository 'models_0'");
}

TEST_F(RegisterApiTest, RegisterWithMultiMap)
{
  // Registering a repository "models_0" where contains "model_0",
  // and "model_0" is mapped to two different names
  std::vector<std::string> override_names{"name_0", "name_1"};
  std::vector<std::shared_ptr<TRITONSERVER_Parameter>> managed_params;
  std::vector<const TRITONSERVER_Parameter*> name_map;
  for (const auto& name : override_names) {
    managed_params.emplace_back(
        TRITONSERVER_ParameterNew(
            "model_0", TRITONSERVER_PARAMETER_STRING, name.c_str()),
        TRITONSERVER_ParameterDelete);
    ASSERT_TRUE(managed_params.back() != nullptr)
        << "failed to create name mapping pair";
    name_map.emplace_back(managed_params.back().get());
  }

  // Such mapping should be allow as it is mapping to unique names
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", name_map.data(), name_map.size()),
      "registering model repository 'models_0'");

  // Request to load "name_0"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "name_0"),
      "loading model 'name_0'");
  // Request to load "name_1"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "name_1"),
      "loading model 'name_1'");
}

TEST_F(RegisterApiTest, RegisterWithRepeatedMap)
{
  // Registering a repository "models_1" where contains "model_0" and "model_1",
  // map "model_0" to "model_1" which creates confliction, however,
  // in EXPLICIT mode, mapping lookup will have higher priority than
  // repository polling so the confliction will be resolved by always loading
  // the model from mapped directory.
  // [FIXME] should this be expected behavior? Should this be the same for
  // POLLING mode? Or it should be treated as having dupliciate models since
  // model manager always iterate the repositories in POLLING.
  std::vector<std::string> override_names{"model_1"};
  std::vector<std::shared_ptr<TRITONSERVER_Parameter>> managed_params;
  std::vector<const TRITONSERVER_Parameter*> name_map;
  managed_params.emplace_back(
      TRITONSERVER_ParameterNew(
          "model_0", TRITONSERVER_PARAMETER_STRING, override_names[0].c_str()),
      TRITONSERVER_ParameterDelete);
  ASSERT_TRUE(managed_params.back() != nullptr)
      << "failed to create name mapping pair";
  name_map.emplace_back(managed_params.back().get());

  // Such mapping should be allow as it is mapping to unique names
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_1", name_map.data(), name_map.size()),
      "registering model repository 'models_1'");

  // Request to load "model_1"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_1"),
      "loading model 'model_1'");
  // [FIXME] check if the correct model is loaded (directory "model_0")
}

TEST_F(RegisterApiTest, RegisterWithRepeatedMap2)
{
  // Registering a repository "models_1" where contains "model_0" and "model_1",
  // map both directories to the same name which creates confliction. Different
  // from 'RegisterWithRepeatedMap', the confliction within the mapping can't be
  // resolved and error should be returend
  std::vector<std::string> dir_names{"model_0", "model_1"};
  std::vector<std::shared_ptr<TRITONSERVER_Parameter>> managed_params;
  std::vector<const TRITONSERVER_Parameter*> name_map;
  for (const auto& name : dir_names) {
    managed_params.emplace_back(
        TRITONSERVER_ParameterNew(
            name.c_str(), TRITONSERVER_PARAMETER_STRING, "name_0"),
        TRITONSERVER_ParameterDelete);
    ASSERT_TRUE(managed_params.back() != nullptr)
        << "failed to create name mapping pair";
    name_map.emplace_back(managed_params.back().get());
  }

  // Register should fail
  FAIL_TEST_IF_NOT_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_1", name_map.data(), name_map.size()),
      TRITONSERVER_ERROR_INVALID_ARG,
      "failed to register 'models_1', there is conflicting mapping",
      "registering model repository 'models_1'");
}

TEST_F(RegisterApiTest, RegisterMulti)
{
  // Registering repository "models_0" and "model_1" without mappings,
  // there are duplicate models but it won't be checked until load
  std::vector<const TRITONSERVER_Parameter*> name_map;
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", name_map.data(), name_map.size()),
      "registering model repository 'models_0'");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_1", name_map.data(), name_map.size()),
      "registering model repository 'models_1'");

  // Request to load "model_0" which should fail
  FAIL_TEST_IF_NOT_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_0"),
      TRITONSERVER_ERROR_INTERNAL,
      "failed to load 'model_0', failed to poll from model repository",
      "loading model 'model_0'");
  // Request to load "model_1"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_1"),
      "loading model 'model_1'");
}

TEST_F(RegisterApiTest, RegisterMultiWithMap)
{
  // Registering repository "models_0" and "model_1" without mappings,
  // there are duplicate models but we provides a "override" map for "models_0",
  // from "model_0" to "model_0" which sets priority to resolve the conflict.
  std::vector<std::string> override_names{"model_0"};
  std::vector<std::shared_ptr<TRITONSERVER_Parameter>> managed_params;
  std::vector<const TRITONSERVER_Parameter*> name_map;
  managed_params.emplace_back(
      TRITONSERVER_ParameterNew(
          "model_0", TRITONSERVER_PARAMETER_STRING, override_names[0].c_str()),
      TRITONSERVER_ParameterDelete);
  ASSERT_TRUE(managed_params.back() != nullptr)
      << "failed to create name mapping pair";
  name_map.emplace_back(managed_params.back().get());
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", name_map.data(), name_map.size()),
      "registering model repository 'models_0'");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_1", nullptr, 0),
      "registering model repository 'models_1'");

  // Request to load "model_0", "model_1"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_0"),
      "loading model 'model_0'");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_1"),
      "loading model 'model_1'");
}

TEST_F(RegisterApiTest, RegisterMultiWithMap2)
{
  // Registering repository "models_0" and "model_1" without mappings,
  // there are duplicate models but we provides a map for "models_1"
  // so they all have different name.
  std::vector<std::string> override_names{"model_2"};
  std::vector<std::shared_ptr<TRITONSERVER_Parameter>> managed_params;
  std::vector<const TRITONSERVER_Parameter*> name_map;
  managed_params.emplace_back(
      TRITONSERVER_ParameterNew(
          "model_0", TRITONSERVER_PARAMETER_STRING, override_names[0].c_str()),
      TRITONSERVER_ParameterDelete);
  ASSERT_TRUE(managed_params.back() != nullptr)
      << "failed to create name mapping pair";
  name_map.emplace_back(managed_params.back().get());
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_0", nullptr, 0),
      "registering model repository 'models_0'");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerRegisterModelRepository(
          server_, "models_1", name_map.data(), name_map.size()),
      "registering model repository 'models_1'");

  // Request to load "model_0", "model_1", "model_2"
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_0"),
      "loading model 'model_0'");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_1"),
      "loading model 'model_1'");
  FAIL_TEST_IF_ERR(
      TRITONSERVER_ServerLoadModel(server_, "model_2"),
      "loading model 'model_2'");
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
