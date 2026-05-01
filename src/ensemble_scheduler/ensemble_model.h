// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "model.h"
#include "model_config.pb.h"
#include "scheduler.h"
#include "status.h"

namespace triton { namespace core {

class InferenceServer;

class EnsembleModel : public Model {
 public:
  EnsembleModel(EnsembleModel&&) = default;

  static Status Create(
      InferenceServer* server, const std::string& path,
      const ModelIdentifier& model_id, const int64_t version,
      const inference::ModelConfig& model_config, const bool is_config_provided,
      const double min_compute_capability, std::unique_ptr<Model>* model);

 private:
  DISALLOW_COPY_AND_ASSIGN(EnsembleModel);

  explicit EnsembleModel(
      const double min_compute_capability, const std::string& model_dir,
      const ModelIdentifier& model_id, const int64_t version,
      const inference::ModelConfig& config)
      : Model(min_compute_capability, model_dir, model_id, version, config)
  {
  }
  friend std::ostream& operator<<(std::ostream&, const EnsembleModel&);
};

std::ostream& operator<<(std::ostream& out, const EnsembleModel& pb);

}}  // namespace triton::core
