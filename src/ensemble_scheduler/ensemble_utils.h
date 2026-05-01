// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef TRITON_ENABLE_ENSEMBLE

#include <deque>
#include <unordered_map>

#include "model_config.pb.h"
#include "model_repository_manager/model_repository_manager.h"
#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

/// Validate that the ensemble are specified correctly. Assuming that the
/// inputs and outputs specified in depending model configurations are accurate.
/// \param model_repository_manager The model manager to acquire model config.
/// \param ensemble The ensemble to be validated.
/// \return The error status.
Status ValidateEnsembleConfig(
    ModelRepositoryManager* model_repository_manager,
    ModelRepositoryManager::DependencyNode* ensemble);

}}  // namespace triton::core

#endif  // TRITON_ENABLE_ENSEMBLE
