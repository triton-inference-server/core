// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "model.h"

#include <chrono>
#include <future>

#include "constants.h"
#include "filesystem/api.h"
#include "infer_request.h"
#include "model_config_utils.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

Status
Model::GetInput(
    const std::string& name, const inference::ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected inference input '" + name + "' for model '" + Name() + "'");
  }

  *input = &itr->second;
  return Status::Success;
}

Status
Model::GetOutput(
    const std::string& name, const inference::ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "unexpected inference output '" + name +
                                       "' for model '" + Name() + "'");
  }

  *output = &itr->second;
  return Status::Success;
}

Status
Model::SetModelConfig(const inference::ModelConfig& config)
{
  config_ = config;
  set_model_config_ = true;

  return Status::Success;
}

Status
Model::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return Status(
        Status::Code::INTERNAL, "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return Status::Success;
}

Status
Model::Init(const bool is_config_provided)
{
  if (!set_model_config_ && !is_config_provided) {
    return Status(
        Status::Code::NOT_FOUND,
        "model configuration is not provided for model '" + Name() + "'");
  }

  RETURN_IF_ERROR(ValidateModelConfig(config_, min_compute_capability_));
  RETURN_IF_ERROR(ValidateModelIOConfig(config_));

  // Initialize the input map
  for (const auto& io : config_.input()) {
    input_map_.insert(std::make_pair(io.name(), io));
    if (!io.optional()) {
      ++required_input_count_;
    }
  }

  // Initialize the output map and label provider for each output
  label_provider_ = std::make_shared<LabelProvider>();
  for (const auto& io : config_.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      auto label_path = JoinPath({model_dir_, io.label_filename()});
      bool is_escaped = false;
      RETURN_IF_ERROR(
          IsChildPathEscapingParentPath(label_path, model_dir_, &is_escaped));
      if (is_escaped) {
        return Status(
            Status::Code::UNSUPPORTED,
            "label file path '" + label_path + "' for output '" + io.name() +
                "' in model '" + Name() + "' is outside model directory");
      }
      RETURN_IF_ERROR(label_provider_->AddLabels(io.name(), label_path));
    }
  }

  if (config_.has_dynamic_batching()) {
    default_priority_level_ =
        config_.dynamic_batching().default_priority_level();
    max_priority_level_ = config_.dynamic_batching().priority_levels();
  } else if (config_.has_ensemble_scheduling()) {
    // For ensemble, allow any priority level to pass through
    default_priority_level_ = 0;
    max_priority_level_ = UINT64_MAX;
  } else {
    default_priority_level_ = 0;
    max_priority_level_ = 0;
  }

#ifdef TRITON_ENABLE_METRICS
  MetricModelReporter::Create(
      ModelId(), Version(), METRIC_REPORTER_ID_UTILITY, ResponseCacheEnabled(),
      Config().metric_tags(), Config().model_metrics(), &reporter_);
#endif  // TRITON_ENABLE_METRICS

  return Status::Success;
}

}}  // namespace triton::core
