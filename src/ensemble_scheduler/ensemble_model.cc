// SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "ensemble_model.h"

#include <stdint.h>

#include "constants.h"
#include "ensemble_scheduler.h"
#include "model_config_utils.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

Status
EnsembleModel::Create(
    InferenceServer* server, const std::string& path,
    const ModelIdentifier& model_id, const int64_t version,
    const inference::ModelConfig& model_config, const bool is_config_provided,
    const double min_compute_capability, std::unique_ptr<Model>* model)
{
  // Create the ensemble model.
  std::unique_ptr<EnsembleModel> local_model(new EnsembleModel(
      min_compute_capability, path, model_id, version, model_config));

  RETURN_IF_ERROR(local_model->Init(is_config_provided));

  std::unique_ptr<Scheduler> scheduler;
  RETURN_IF_ERROR(
      EnsembleScheduler::Create(
          local_model->MutableStatsAggregator(), server, local_model->ModelId(),
          model_config, &scheduler));
  RETURN_IF_ERROR(local_model->SetScheduler(std::move(scheduler)));

  LOG_VERBOSE(1) << "ensemble model for " << local_model->Name() << std::endl;

  *model = std::move(local_model);
  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const EnsembleModel& pb)
{
  out << "name=" << pb.Name() << std::endl;
  return out;
}

}}  // namespace triton::core
