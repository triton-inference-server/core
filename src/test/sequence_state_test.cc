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
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "sequence_state.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "triton/common/model_config.h"
#include "triton/core/tritonserver.h"

namespace tc = triton::core;

namespace {

using StateConfig = inference::ModelSequenceBatching_State;

StateConfig
CreateStateConfig(
    const std::string& input_name, const std::string& output_name,
    inference::DataType data_type, const std::vector<int64_t>& dims)
{
  StateConfig state;
  state.set_input_name(input_name);
  state.set_output_name(output_name);
  state.set_data_type(data_type);
  for (const auto dim : dims) {
    state.add_dims(dim);
  }
  return state;
}

TEST(SequenceStatesTest, OutputStateUsesConfiguredInputMapping)
{
  std::vector<StateConfig> state_configs;
  state_configs.emplace_back(CreateStateConfig(
      "cache_last_channel_len", "cache_last_channel_len_next",
      inference::DataType::TYPE_INT64, {1}));
  state_configs.emplace_back(CreateStateConfig(
      "cache_last_channel", "cache_last_channel_next",
      inference::DataType::TYPE_FP32, {4}));

  std::unordered_map<std::string, const StateConfig&> state_config_map;
  for (const auto& state_config : state_configs) {
    state_config_map.emplace(state_config.output_name(), state_config);
  }

  tc::SequenceStates states;
  auto status = states.Initialize(
      state_config_map, 0 /* max_batch_size */, {} /* initial_state */,
      TRITONSERVER_INSTANCEGROUPKIND_CPU, 0 /* device_id */,
      {} /* cuda_virtual_address_size */);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  tc::SequenceState* output_state = nullptr;
  const std::vector<int64_t> channel_update_shape{8};
  status = states.OutputState(
      "cache_last_channel_next", inference::DataType::TYPE_FP32,
      channel_update_shape, &output_state);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_NE(output_state, nullptr);

  auto output_memory = std::make_shared<tc::AllocatedMemory>(
      sizeof(float) * channel_update_shape[0], TRITONSERVER_MEMORY_CPU, 0);
  status = output_state->SetData(output_memory);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  status = output_state->Update();
  ASSERT_TRUE(status.IsOk()) << status.Message();

  const auto& input_states = states.InputStates();
  const auto& channel_state = input_states.at("cache_last_channel");
  const auto& len_state = input_states.at("cache_last_channel_len");

  EXPECT_EQ(channel_state->DType(), inference::DataType::TYPE_FP32);
  EXPECT_EQ(channel_state->Shape(), channel_update_shape);
  EXPECT_EQ(len_state->DType(), inference::DataType::TYPE_INT64);
  EXPECT_EQ(len_state->Shape(), std::vector<int64_t>({1}));
}

}  // namespace
