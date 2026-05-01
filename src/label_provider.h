// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "constants.h"
#include "status.h"

namespace triton { namespace core {

// Provides classification labels.
class LabelProvider {
 public:
  LabelProvider() = default;

  // Return the label associated with 'name' for a given
  // 'index'. Return empty string if no label is available.
  const std::string& GetLabel(const std::string& name, size_t index) const;

  // Associate with 'name' a set of labels initialized from a given
  // 'filepath'. Within the file each label is specified on its own
  // line. The first label (line 0) is the index-0 label, the second
  // label (line 1) is the index-1 label, etc.
  Status AddLabels(const std::string& name, const std::string& filepath);

  // Return the labels associated with 'name'. Return empty vector if no labels
  // are available.
  const std::vector<std::string>& GetLabels(const std::string& name);

  // Associate with 'name' a set of 'labels'
  Status AddLabels(
      const std::string& name, const std::vector<std::string>& labels);

 private:
  DISALLOW_COPY_AND_ASSIGN(LabelProvider);

  std::unordered_map<std::string, std::vector<std::string>> label_map_;
};

}}  // namespace triton::core
