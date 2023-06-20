// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "persistent_backend_manager.h"

#include <memory>
#include <mutex>
#include "backend_config.h"
#include "filesystem/api.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

static std::weak_ptr<PersistentBackendManager> persist_backend_manager_;
static std::mutex mu_;

Status
PersistentBackendManager::Create(
    const triton::common::BackendCmdlineConfigMap& config_map,
    std::shared_ptr<PersistentBackendManager>* manager,
    std::shared_ptr<TritonBackendManager>& backend_manager)
{
  std::lock_guard<std::mutex> lock(mu_);

  // If there is already a manager then we just use it...
  *manager = persist_backend_manager_.lock();
  if (*manager != nullptr) {
    return Status::Success;
  }

  manager->reset(new PersistentBackendManager());
  persist_backend_manager_ = *manager;

  RETURN_IF_ERROR(
      (*manager)->InitPersistentBackends(config_map, backend_manager));
  return Status::Success;
}

Status
PersistentBackendManager::InitPersistentBackends(
    const triton::common::BackendCmdlineConfigMap& config_map,
    std::shared_ptr<TritonBackendManager>& backend_manager)
{
  for (const auto& be : {"pytorch"}) {
    RETURN_IF_ERROR(InitPersistentBackend(be, config_map, backend_manager));
  }

  return Status::Success;
}

Status
PersistentBackendManager::InitPersistentBackend(
    const std::string& backend_name,
    const triton::common::BackendCmdlineConfigMap& config_map,
    std::shared_ptr<TritonBackendManager>& backend_manager)
{
  std::string backends_dir;
  std::string specialized_backend_name;
  std::string backend_libname;
  RETURN_IF_ERROR(
      BackendConfigurationGlobalBackendsDirectory(config_map, &backends_dir));
  RETURN_IF_ERROR(BackendConfigurationSpecializeBackendName(
      config_map, backend_name, &specialized_backend_name));
  RETURN_IF_ERROR(BackendConfigurationBackendLibraryName(
      specialized_backend_name, &backend_libname));

  const auto backend_dir = JoinPath({backends_dir, specialized_backend_name});
  const auto backend_libpath = JoinPath({backend_dir, backend_libname});
  bool exists = false;
  RETURN_IF_ERROR(FileExists(backend_libpath, &exists));
  if (exists) {
    triton::common::BackendCmdlineConfig empty_backend_cmdline_config;
    const triton::common::BackendCmdlineConfig* config;
    const auto& itr = config_map.find(backend_name);
    if (itr == config_map.end()) {
      config = &empty_backend_cmdline_config;
    } else {
      config = &itr->second;
    }

    std::shared_ptr<TritonBackend> persist_backend;
    RETURN_IF_ERROR(backend_manager->CreateBackend(
        backend_name, backend_dir, backend_libpath, *config, &persist_backend));
    persist_backends_.push_back(persist_backend);
  }

  return Status::Success;
}

}}  // namespace triton::core
