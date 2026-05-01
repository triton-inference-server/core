// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "label_provider.h"

#include <iostream>
#include <iterator>
#include <sstream>

#include "filesystem/api.h"

namespace triton { namespace core {

const std::string&
LabelProvider::GetLabel(const std::string& name, size_t index) const
{
  static const std::string not_found;

  auto itr = label_map_.find(name);
  if (itr == label_map_.end()) {
    return not_found;
  }

  if (itr->second.size() <= index) {
    return not_found;
  }

  return itr->second[index];
}

Status
LabelProvider::AddLabels(const std::string& name, const std::string& filepath)
{
  std::string label_file_contents;
  RETURN_IF_ERROR(ReadTextFile(filepath, &label_file_contents));

  auto p = label_map_.insert(std::make_pair(name, std::vector<std::string>()));
  if (!p.second) {
    return Status(
        Status::Code::INTERNAL, "multiple label files for '" + name + "'");
  }

  auto itr = p.first;

  std::istringstream label_file_stream(label_file_contents);
  std::string line;
  while (std::getline(label_file_stream, line)) {
    itr->second.push_back(line);
  }

  return Status::Success;
}

const std::vector<std::string>&
LabelProvider::GetLabels(const std::string& name)
{
  static const std::vector<std::string> not_found;
  auto itr = label_map_.find(name);
  if (itr == label_map_.end()) {
    return not_found;
  }
  return itr->second;
}

Status
LabelProvider::AddLabels(
    const std::string& name, const std::vector<std::string>& labels)
{
  label_map_.emplace(name, labels);
  return Status::Success;
}

}}  // namespace triton::core
