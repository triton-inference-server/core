// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "backend_model.h"

#include <map>
#include <vector>

#include "backend_config.h"
#include "dynamic_batch_scheduler.h"
#include "filesystem/api.h"
#include "model_config_utils.h"
#include "numa_utils.h"
#include "sequence_batch_scheduler/sequence_batch_scheduler.h"
#include "sequence_state.h"
#include "server.h"
#include "server_message.h"
#include "shared_library.h"
#include "triton/common/logging.h"
#include "tritonserver_apis.h"

// For unknown reason, windows will not export the TRITONBACKEND_*
// functions declared with dllexport in tritonbackend.h. To get those
// functions exported it is (also?) necessary to mark the definitions
// in this file with dllexport as well.
#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

namespace triton { namespace core {

Status
TritonModel::Create(
    InferenceServer* server, const std::string& model_path,
    const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
    const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
    const ModelIdentifier& model_id, const int64_t version,
    inference::ModelConfig model_config, const bool is_config_provided,
    std::unique_ptr<TritonModel>* model)
{
  model->reset();

  // The model configuration must specify a backend.
  const std::string& backend_name = model_config.backend();
  if (backend_name.empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'backend' for '" + model_config.name() + "'");
  }

  // Localize the content of the model repository corresponding to
  // 'model_path'. This model holds a handle to the localized content
  // so that it persists as long as the model is loaded.
  std::shared_ptr<LocalizedPath> localized_model_dir;
  RETURN_IF_ERROR(LocalizePath(model_path, &localized_model_dir));

  // Localize paths in backend model config
  // [FIXME] Remove once a more permanent solution is implemented (DLIS-4211)
  RETURN_IF_ERROR(LocalizePythonBackendExecutionEnvironmentPath(
      model_path, &model_config, &localized_model_dir));

  // Get some internal configuration values needed for initialization.
  std::string backend_dir;
  RETURN_IF_ERROR(BackendConfigurationGlobalBackendsDirectory(
      backend_cmdline_config_map, &backend_dir));

  bool auto_complete_config = false;
  RETURN_IF_ERROR(BackendConfigurationAutoCompleteConfig(
      backend_cmdline_config_map, &auto_complete_config));

  double min_compute_capability = 0;
  RETURN_IF_ERROR(BackendConfigurationMinComputeCapability(
      backend_cmdline_config_map, &min_compute_capability));

  std::string specialized_backend_name;
  RETURN_IF_ERROR(BackendConfigurationSpecializeBackendName(
      backend_cmdline_config_map, backend_name, &specialized_backend_name));

  bool is_python_based_backend = false;
  std::vector<std::string> search_paths = GetBackendLibrarySearchPaths(
      model_path, version, backend_dir, backend_name);
  std::string backend_libdir, backend_libpath;

  RETURN_IF_ERROR(GetBackendLibraryProperties(
      localized_model_dir->Path(), version, backend_dir,
      specialized_backend_name, &model_config, &is_python_based_backend,
      &search_paths, &backend_libdir, &backend_libpath));

  if (is_python_based_backend) {
    RETURN_IF_ERROR(SetPythonBasedBackendExecutionEnvironment(
        backend_libdir, &model_config));
  }

  // Resolve the global backend configuration with the specific backend
  // configuration
  triton::common::BackendCmdlineConfig config;
  RETURN_IF_ERROR(ResolveBackendConfigs(
      backend_cmdline_config_map,
      (is_python_based_backend ? kPythonBackend : backend_name), config));

  RETURN_IF_ERROR(SetBackendConfigDefaults(config));

  std::shared_ptr<TritonBackend> backend;
  RETURN_IF_ERROR(server->BackendManager()->CreateBackend(
      backend_name, backend_libdir, backend_libpath, config,
      is_python_based_backend, &backend));

  // Normalize backend-dependent config
  {
    const auto& attributes = backend->BackendAttributes();
    // [WIP] formalize config normalization / validation
    RETURN_IF_ERROR(NormalizeInstanceGroup(
        min_compute_capability, attributes.preferred_groups_, &model_config));
    RETURN_IF_ERROR(
        ValidateInstanceGroup(model_config, min_compute_capability));
  }

  // Create and initialize the model.
  std::unique_ptr<TritonModel> local_model(new TritonModel(
      server, localized_model_dir, backend, min_compute_capability, model_id,
      version, model_config, auto_complete_config, backend_cmdline_config_map,
      host_policy_map));

  TritonModel* raw_local_model = local_model.get();

  // Model initialization is optional... The TRITONBACKEND_Model object is this
  // TritonModel object.
  if (backend->ModelInitFn() != nullptr) {
    // We must set set shared library path to point to the backend directory in
    // case the backend library attempts to load additional shared libraries.
    // Currently, the set and reset function is effective only on Windows, so
    // there is no need to set path on non-Windows.
    // However, parallel model loading will not see any speedup on Windows and
    // the global lock inside the SharedLibrary is a WAR.
    // [FIXME] Reduce lock WAR on SharedLibrary (DLIS-4300)
#ifdef _WIN32
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->SetLibraryDirectory(backend->Directory()));
#endif

    TRITONSERVER_Error* err = backend->ModelInitFn()(
        reinterpret_cast<TRITONBACKEND_Model*>(raw_local_model));

#ifdef _WIN32
    RETURN_IF_ERROR(slib->ResetLibraryDirectory());
#endif
    RETURN_IF_TRITONSERVER_ERROR(err);
  }

  // Initialize the model for Triton core usage
  RETURN_IF_ERROR(local_model->Init(is_config_provided));

  RETURN_IF_ERROR(local_model->GetExecutionPolicy(model_config));

  // Initialize the custom batching library for the model, if provided.
  if (model_config.has_sequence_batching()) {
    if (model_config.parameters().contains("TRITON_BATCH_STRATEGY_PATH")) {
      return Status(
          Status::Code::INVALID_ARG,
          "TRITON_BATCH_STRATEGY_PATH cannot be specified with "
          "sequence batcher, using default batching strategy");
    }
  } else {
    std::string batch_libpath;
    if (model_config.parameters().contains("TRITON_BATCH_STRATEGY_PATH")) {
      batch_libpath = model_config.parameters()
                          .at("TRITON_BATCH_STRATEGY_PATH")
                          .string_value();
      bool exists = false;
      RETURN_IF_ERROR(FileExists(batch_libpath, &exists));
      if (!exists) {
        return Status(
            triton::common::Error::Code::NOT_FOUND,
            ("Batching library path not found: " + batch_libpath).c_str());
      }
    } else {
      const std::string batch_libname = "batchstrategy.so";
      for (const auto& path : search_paths) {
        const auto full_path = JoinPath({path, batch_libname});
        bool exists = false;
        RETURN_IF_ERROR(FileExists(full_path, &exists));
        if (exists) {
          batch_libpath = full_path;
          break;
        }
      }
    }

    if (!batch_libpath.empty()) {
      LOG_INFO << "Loading custom batching strategy library " << batch_libpath
               << " for model " << model_config.name();
      RETURN_IF_ERROR(local_model->SetBatchingStrategy(batch_libpath));
    }
  }

  // Create or update the model instances for this model.
  std::vector<std::shared_ptr<TritonModelInstance>> added_instances,
      removed_instances;
  RETURN_IF_ERROR(local_model->PrepareInstances(
      model_config, &added_instances, &removed_instances));
  RETURN_IF_ERROR(local_model->SetConfiguredScheduler(added_instances));
  local_model->CommitInstances();

  *model = std::move(local_model);
  return Status::Success;
}

Status
TritonModel::UpdateInstanceGroup(const inference::ModelConfig& new_model_config)
{
  // Generate normalized model config with new instance group.
  inference::ModelConfig model_config = config_;
  model_config.clear_instance_group();
  model_config.mutable_instance_group()->Add(
      new_model_config.instance_group().begin(),
      new_model_config.instance_group().end());
  RETURN_IF_ERROR(NormalizeInstanceGroup(
      min_compute_capability_, backend_->BackendAttributes().preferred_groups_,
      &model_config));
  RETURN_IF_ERROR(ValidateInstanceGroup(model_config, min_compute_capability_));

  // Prepare the new instances on the new config.
  std::vector<std::shared_ptr<TritonModelInstance>> added_instances,
      removed_instances;
  Status status =
      PrepareInstances(model_config, &added_instances, &removed_instances);
  if (!status.IsOk()) {
    ClearBackgroundInstances();
    return status;
  }

  // Update the scheduler.
  status = UpdateConfiguredScheduler(added_instances, removed_instances);
  if (!status.IsOk()) {
    ClearBackgroundInstances();
    return status;
  }

  // Commit the instance update.
  CommitInstances();
  *config_.mutable_instance_group() = model_config.instance_group();

  return Status::Success;
}

Status
TritonModel::GetExecutionPolicy(const inference::ModelConfig& model_config)
{
  // Set 'device_blocking_'
  device_blocking_ = false;
  if (backend_->ExecutionPolicy() == TRITONBACKEND_EXECUTION_DEVICE_BLOCKING) {
    if (model_config.has_sequence_batching()) {
      LOG_INFO << "Overriding execution policy to "
                  "\"TRITONBACKEND_EXECUTION_BLOCKING\" for sequence model \""
               << model_config.name() << "\"";
    } else {
      device_blocking_ = true;
    }
  }

  return Status::Success;
}

std::vector<std::string>
TritonModel::GetBackendLibrarySearchPaths(
    const std::string& model_path, int64_t version,
    const std::string& backend_dir, const std::string& backend_name)
{
  const auto version_path = JoinPath({model_path, std::to_string(version)});
  const auto backend_path = JoinPath({backend_dir, backend_name});
  std::vector<std::string> search_paths = {
      version_path, model_path, backend_path};
  return search_paths;
}

Status
TritonModel::GetBackendLibraryProperties(
    const std::string& model_path, int64_t version,
    const std::string& backend_dir, const std::string& backend_name,
    inference::ModelConfig* model_config, bool* is_python_based_backend,
    std::vector<std::string>* search_paths, std::string* backend_libdir,
    std::string* backend_libpath)
{
  std::string python_based_backend_libdir;
  std::string backend_libname = model_config->runtime();
  if (backend_libname.empty()) {
    RETURN_IF_ERROR(GetBackendRuntimeLibraryName(
        backend_dir, backend_name, *search_paths, &backend_libname,
        backend_libdir, backend_libpath, is_python_based_backend));
    if (!*is_python_based_backend) {
      // All variables are correctly set for C++ backends on initial search.
      return Status::Success;
    }
    python_based_backend_libdir = *backend_libdir;
    model_config->set_runtime(backend_libname);
  } else {
    *is_python_based_backend = backend_libname == kPythonFilename;
  }

  std::string cpp_backend_libname = backend_libname;
  if (*is_python_based_backend) {
    // Set C++ library name to Python backend.
    cpp_backend_libname = AssembleCPPRuntimeLibraryName(kPythonBackend);
    // The search paths only contain locations related to the Python backend
    // based backend, the global Python backend location has to be added.
    search_paths->emplace_back(JoinPath({backend_dir, kPythonBackend}));
  }

  RETURN_IF_ERROR(FindBackendLibraryPath(
      *search_paths, cpp_backend_libname, backend_libdir, backend_libpath));
  if (backend_libpath->empty()) {
    std::string search_paths_str = "";
    for (const auto& path : *search_paths) {
      search_paths_str += "'" + path + "' ";
    }
    return Status(
        Status::Code::INVALID_ARG, "unable to find backend library '" +
                                       cpp_backend_libname + "' for model '" +
                                       model_config->name() +
                                       "', searched: " + search_paths_str);
  }
  if (IsChildPathEscapingParentPath(
          *backend_libpath /* child_path */,
          *backend_libdir /* parent_path */)) {
    return Status(
        Status::Code::INVALID_ARG,
        "backend library name '" + cpp_backend_libname +
            "' escapes backend directory '" + *backend_libdir +
            "', for model '" + model_config->name() +
            "', check model config runtime field");
  }

  // Both 'backend_libdir' and 'backend_libpath' are now pointing to the C++
  // backend library, 'backend_libdir' needs adjustment for Python based
  // backend.
  if (*is_python_based_backend) {
    if (python_based_backend_libdir.empty()) {
      python_based_backend_libdir = JoinPath({backend_dir, backend_name});
      // Make sure the library and its directory exist.
      std::string path =
          JoinPath({python_based_backend_libdir, kPythonFilename});
      bool path_exist;
      RETURN_IF_ERROR(FileExists(path, &path_exist));
      if (!path_exist) {
        return Status(
            Status::Code::INVALID_ARG,
            "unable to find Python backend based backend library '" +
                backend_libname + "' for model '" + model_config->name() +
                "', searched: '" + path + "'");
      }
    }
    *backend_libdir = python_based_backend_libdir;
  }

  return Status::Success;
}

Status
TritonModel::GetBackendRuntimeLibraryName(
    const std::string& backend_dir, const std::string& backend_name,
    const std::vector<std::string>& search_paths, std::string* backend_libname,
    std::string* backend_libdir, std::string* backend_libpath,
    bool* is_python_based_backend)
{
  // Try C++ runtime
  *backend_libname = AssembleCPPRuntimeLibraryName(backend_name);
  RETURN_IF_ERROR(FindBackendLibraryPath(
      search_paths, *backend_libname, backend_libdir, backend_libpath));
  if (!backend_libpath->empty()) {
    *is_python_based_backend = false;
    return Status::Success;
  }
  // Try Python runtime
  std::vector<std::string> python_search_paths = {
      JoinPath({backend_dir, backend_name})};
  *backend_libname = kPythonFilename;
  RETURN_IF_ERROR(FindBackendLibraryPath(
      python_search_paths, *backend_libname, backend_libdir, backend_libpath));
  if (!backend_libpath->empty()) {
    *is_python_based_backend = true;
    return Status::Success;
  }
  // Cannot find runtime
  return Status(
      Status::Code::INVALID_ARG,
      "unable to find backend library for backend '" + backend_name +
          "', try specifying runtime on the model configuration.");
}

Status
TritonModel::FindBackendLibraryPath(
    const std::vector<std::string>& search_paths,
    const std::string& backend_libname, std::string* backend_libdir,
    std::string* backend_libpath)
{
  backend_libpath->clear();

  for (const auto& path : search_paths) {
    const auto full_path = JoinPath({path, backend_libname});
    bool exists = false;
    RETURN_IF_ERROR(FileExists(full_path, &exists));
    if (exists) {
      *backend_libdir = path;
      *backend_libpath = full_path;
      break;
    }
  }

  return Status::Success;
}

std::string
TritonModel::AssembleCPPRuntimeLibraryName(const std::string& backend_name)
{
#ifdef _WIN32
  return "triton_" + backend_name + ".dll";
#else
  return "libtriton_" + backend_name + ".so";
#endif
}

Status
TritonModel::ResolveBackendConfigs(
    const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
    const std::string& backend_name,
    triton::common::BackendCmdlineConfig& config)
{
  const auto& global_itr = backend_cmdline_config_map.find(std::string());
  const auto& specific_itr = backend_cmdline_config_map.find(backend_name);
  std::map<std::string, std::string> lconfig;
  if (global_itr != backend_cmdline_config_map.end()) {
    // Accumulate all global settings
    for (auto& setting : global_itr->second) {
      lconfig[setting.first] = setting.second;
    }
  }
  if (specific_itr != backend_cmdline_config_map.end()) {
    // Accumulate backend specific settings and override
    // global settings with specific configs if needed
    for (auto& setting : specific_itr->second) {
      lconfig[setting.first] = setting.second;
    }
  }
  for (auto& final_setting : lconfig) {
    config.emplace_back(final_setting);
  }

  return Status::Success;
}


const std::unordered_map<std::string, std::string> backend_config_defaults(
    {{"default-max-batch-size", "4"}});

Status
TritonModel::SetBackendConfigDefaults(
    triton::common::BackendCmdlineConfig& config)
{
  auto backend_config_defaults_copy = backend_config_defaults;

  for (auto& setting : config) {
    if (setting.first.compare("default-max-batch-size") == 0) {
      LOG_VERBOSE(1) << "Found overwritten default setting: " << setting.first
                     << "," << setting.second;
      backend_config_defaults_copy.erase(setting.first);
    }

    if (backend_config_defaults_copy.empty()) {
      break;
    }
  }

  // Anything left should be added to the config
  for (const auto& default_setting : backend_config_defaults_copy) {
    LOG_VERBOSE(1) << "Adding default backend config setting: "
                   << default_setting.first << "," << default_setting.second;
    config.push_back(
        std::make_pair(default_setting.first, default_setting.second));
  }

  return Status::Success;
}

std::unordered_map<
    TritonModelInstance::Signature,
    std::vector<std::shared_ptr<TritonModelInstance>>>
TritonModel::IndexInstances() const
{
  std::unordered_map<
      TritonModelInstance::Signature,
      std::vector<std::shared_ptr<TritonModelInstance>>>
      mapped_instances;
  for (auto* instances : {&instances_, &passive_instances_}) {
    for (auto& instance : (*instances)) {
      auto itr = mapped_instances
                     .emplace(
                         instance->GetSignature(),
                         std::vector<std::shared_ptr<TritonModelInstance>>())
                     .first;
      itr->second.push_back(instance);
    }
  }
  return mapped_instances;
}

Status
TritonModel::PrepareInstances(
    const inference::ModelConfig& model_config,
    std::vector<std::shared_ptr<TritonModelInstance>>* added_instances,
    std::vector<std::shared_ptr<TritonModelInstance>>* removed_instances)
{
  added_instances->clear();
  removed_instances->clear();

  std::unordered_map<
      TritonModelInstance::Signature,
      std::vector<std::shared_ptr<TritonModelInstance>>>
      existing_instances = IndexInstances();


  std::vector<std::future<Status>> creation_results;
  // Used to protect shared states for parallel instance loading
  std::mutex instance_mu;

  // Deferred will be lazily evaluated when the result is requested. Since the
  // creation_results are requested serially below, this is equivalent to making
  // the calls serially.
  auto launch_policy = std::launch::deferred;

  // Override for testing/debugging purposes
  bool parallel = backend_->BackendAttributes().parallel_instance_loading_;
  const char* env = std::getenv("TRITON_PARALLEL_INSTANCE_LOADING");
  if (env != nullptr) {
    std::string s_env = std::string(env);
    if (!s_env.empty()) {
      parallel = (s_env == "1") ? true : false;
      LOG_VERBOSE(1)
          << "Using TRITON_PARALLEL_INSTANCE_LOADING environment variable "
             "override: "
          << parallel;
    }
  }

  // If the backend supports it, std::launch::async will allow concurrent calls
  if (parallel) {
    launch_policy = std::launch::async;
  }

  // Iterates over all the requested instances on the model config, and decides
  // if each requested instance can reuse an existing instance or a new instance
  // is needed.
  for (const auto& group : model_config.instance_group()) {
    std::vector<std::string> profile_names;
    for (const auto& profile_name : group.profile()) {
      profile_names.push_back(profile_name);
    }
    std::vector<TritonModelInstance::SecondaryDevice> secondary_devices;
    for (const auto& secondary_device : group.secondary_devices()) {
      secondary_devices.emplace_back(
          inference::
              ModelInstanceGroup_SecondaryDevice_SecondaryDeviceKind_Name(
                  secondary_device.kind()),
          secondary_device.device_id());
    }
    for (int32_t c = 0; c < group.count(); ++c) {
      std::string instance_name{group.name() + "_" + std::to_string(c)};
      const bool passive = group.passive();
      struct InstanceSetting {
        InstanceSetting(
            const std::string& policy_name, TRITONSERVER_InstanceGroupKind kind,
            int32_t device_id,
            const inference::ModelRateLimiter* rate_limiter_config)
            : policy_name_(policy_name), kind_(kind), device_id_(device_id),
              rate_limiter_config_(rate_limiter_config)
        {
        }
        const std::string policy_name_;
        const TRITONSERVER_InstanceGroupKind kind_;
        const int32_t device_id_;
        const inference::ModelRateLimiter* rate_limiter_config_;
      };
      std::vector<InstanceSetting> instance_settings;
      if (group.kind() == inference::ModelInstanceGroup::KIND_CPU) {
        instance_settings.emplace_back(
            group.host_policy().empty() ? "cpu" : group.host_policy(),
            TRITONSERVER_INSTANCEGROUPKIND_CPU, 0 /* device_id */,
            &group.rate_limiter());
      } else if (group.kind() == inference::ModelInstanceGroup::KIND_GPU) {
        for (const int32_t device_id : group.gpus()) {
          instance_settings.emplace_back(
              group.host_policy().empty() ? ("gpu_" + std::to_string(device_id))
                                          : group.host_policy(),
              TRITONSERVER_INSTANCEGROUPKIND_GPU, device_id,
              &group.rate_limiter());
        }
      } else if (group.kind() == inference::ModelInstanceGroup::KIND_MODEL) {
        instance_settings.emplace_back(
            group.host_policy().empty() ? "model" : group.host_policy(),
            TRITONSERVER_INSTANCEGROUPKIND_MODEL, 0 /* device_id */,
            &group.rate_limiter());
      } else {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("instance_group kind ") +
                ModelInstanceGroup_Kind_Name(group.kind()) + " not supported");
      }
      for (const auto& is : instance_settings) {
        // All the information for the requested instance is ready. Create a
        // signature that identifies the requested instance.
        const TritonModelInstance::Signature signature(group, is.device_id_);

        // Check if the requested instance can reuse an existing instance.
        if (!TritonModelInstance::ShareBackendThread(
                DeviceBlocking(), is.kind_)) {
          auto itr = existing_instances.find(signature);
          if (itr != existing_instances.end() && !itr->second.empty()) {
            auto existing_instance = itr->second.back();
            itr->second.pop_back();
            LOG_VERBOSE(2) << "Re-using model instance named '"
                           << existing_instance->Name() << "' with device id '"
                           << existing_instance->DeviceId() << "'";
            RegisterBackgroundInstance(std::move(existing_instance), passive);

            continue;
          }
        }

        // Note that the local variables should be captured by value
        creation_results.emplace_back(
            std::async(launch_policy, [=, &instance_mu]() {
              // The requested instance did not match an existing instance.
              // Create a new instance.
              std::shared_ptr<TritonModelInstance> new_instance;
              RETURN_IF_ERROR(TritonModelInstance::CreateInstance(
                  this, instance_name, signature, is.kind_, is.device_id_,
                  profile_names, passive, is.policy_name_,
                  *is.rate_limiter_config_, secondary_devices, &new_instance));
              {
                std::lock_guard<std::mutex> lk(instance_mu);
                added_instances->push_back(new_instance);
                RegisterBackgroundInstance(std::move(new_instance), passive);
              }
              // Keep logging to a single stream operator to avoid interweaving
              const auto msg = "Created model instance named '" +
                               instance_name + "' with device id '" +
                               std::to_string(is.device_id_) + "'";
              LOG_VERBOSE(2) << msg;
              return Status::Success;
            }));
      }
    }
  }

  // Any existing instances not reused will be removed.
  for (auto pair : existing_instances) {
    removed_instances->insert(
        removed_instances->end(), pair.second.begin(), pair.second.end());
  }

  auto status = Status::Success;
  for (auto& cr : creation_results) {
    auto lstatus = cr.get();
    if (!lstatus.IsOk()) {
      LOG_ERROR << "ERROR: Failed to create instance: " << lstatus.Message();
      status = lstatus;
    }
  }

  return status;
}

void
TritonModel::CommitInstances()
{
  instances_.swap(bg_instances_);
  passive_instances_.swap(bg_passive_instances_);
  ClearBackgroundInstances();
}

void
TritonModel::RegisterBackgroundInstance(
    std::shared_ptr<TritonModelInstance>&& instance, const bool passive)
{
  if (passive) {
    bg_passive_instances_.emplace_back(std::move(instance));
  } else {
    bg_instances_.emplace_back(std::move(instance));
  }
}

void
TritonModel::ClearBackgroundInstances()
{
  bg_instances_.clear();
  bg_passive_instances_.clear();
}

std::vector<std::shared_ptr<TritonModelInstance>>
TritonModel::GetInstancesByDevice(int32_t device_id) const
{
  std::vector<std::shared_ptr<TritonModelInstance>> result;
  // Do not match passive instances, as they do not have a backend thread.
  // Do not match foreground instances, as backend threads cannot be updated.
  for (auto& instance : bg_instances_) {
    if (instance->DeviceId() == device_id) {
      result.push_back(instance);
    }
  }
  return result;
}

Status
TritonModel::UpdateModelConfig(
    const uint32_t config_version, TRITONSERVER_Message* updated_config_message)
{
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(TRITONSERVER_MessageSerializeToJson(
      updated_config_message, &buffer, &byte_size));
  inference::ModelConfig updated_config;
  RETURN_IF_ERROR(
      JsonToModelConfig({buffer, byte_size}, config_version, &updated_config));
  auto config = Config();
  config.set_max_batch_size(updated_config.max_batch_size());
  auto inputs_config = config.mutable_input();
  *inputs_config = updated_config.input();
  auto outputs_config = config.mutable_output();
  *outputs_config = updated_config.output();

  if (!config.scheduling_choice_case()) {
    if (updated_config.has_dynamic_batching()) {
      auto dynamic_batching_config = config.mutable_dynamic_batching();
      *dynamic_batching_config = updated_config.dynamic_batching();
    } else if (updated_config.has_sequence_batching()) {
      auto sequence_batching_config = config.mutable_sequence_batching();
      *sequence_batching_config = updated_config.sequence_batching();
    } else if (updated_config.has_ensemble_scheduling()) {
      auto ensemble_scheduling_config = config.mutable_ensemble_scheduling();
      *ensemble_scheduling_config = updated_config.ensemble_scheduling();
    }  // else do nothing
  } else if (
      config.scheduling_choice_case() !=
      updated_config.scheduling_choice_case()) {
    return Status(
        triton::common::Error::Code::INTERNAL,
        (std::string("Cannot update scheduling choice from ") +
         std::to_string(config.scheduling_choice_case()) + std::string(" to ") +
         std::to_string(config.scheduling_choice_case()) +
         std::string(" when auto-completing."))
            .c_str());
  }  // else do nothing

  // Update model_transaction_policy if needed
  if (updated_config.has_model_transaction_policy()) {
    bool is_decoupled = updated_config.model_transaction_policy().decoupled();
    config.mutable_model_transaction_policy()->set_decoupled(is_decoupled);
  }

  // Need to normalize the model configuration for
  // populating missing fields.
  RETURN_IF_ERROR(NormalizeModelConfig(min_compute_capability_, &config));

  RETURN_IF_ERROR(SetModelConfig(config));

  return Status::Success;
}

Status
TritonModel::SetConfiguredScheduler(
    const std::vector<std::shared_ptr<TritonModelInstance>>& new_instances)
{
  std::unique_ptr<Scheduler> scheduler;

  // Need to enforce equal shape batches (i.e. non-ragged batches) if
  // the model 1) allows one or more variable-size input tensors that
  // are not marked as 'allow_ragged_batch' or 2) has one or more
  // shape-tensor inputs. This is not needed if all input shapes are
  // non-variable and if there are no shape tensors... so we don't
  // enable it in that case for efficiency reasons.
  std::unordered_map<std::string, bool> enforce_equal_shape_tensors;
  for (const auto& input : config_.input()) {
    if (input.is_shape_tensor()) {
      enforce_equal_shape_tensors.insert({input.name(), true});
    } else if (
        !input.allow_ragged_batch() &&
        (triton::common::GetElementCount(input) == -1)) {
      enforce_equal_shape_tensors.insert({input.name(), false});
    }
  }

  // If 'sequence_batching' is configured, then use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    // Sequence batcher
    if (config_.parameters().contains("TRITON_BATCH_STRATEGY_PATH")) {
      LOG_ERROR << "TRITON_BATCH_STRATEGY_PATH cannot be specified with "
                   "sequence batcher, using default batching strategy";
    }
    RETURN_IF_ERROR(SequenceBatchScheduler::Create(
        this, new_instances, enforce_equal_shape_tensors, &scheduler));
  } else if (config_.has_dynamic_batching()) {
    // Dynamic batcher
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        this, nullptr, 0 /*nice*/, true /* dynamic_batching_enabled */,
        config_.max_batch_size(), enforce_equal_shape_tensors,
        config_.dynamic_batching(), &scheduler));
  } else {
    // Default scheduler. Use dynamic batch scheduler (with batching
    // disabled) as the default scheduler.
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        this, nullptr, 0 /*nice*/, false /* dynamic_batching_enabled */,
        1 /* max_batch_size */,
        std::unordered_map<
            std::string, bool>() /* enforce_equal_shape_tensors */,
        false /* preserve_ordering */,
        std::set<int32_t>() /* preferred_batch_sizes */,
        0 /* max_queue_delay_microseconds */, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

Status
TritonModel::UpdateConfiguredScheduler(
    const std::vector<std::shared_ptr<TritonModelInstance>>& added_instances,
    const std::vector<std::shared_ptr<TritonModelInstance>>& removed_instances)
{
  if (config_.has_sequence_batching()) {
    SequenceBatchScheduler* sched =
        dynamic_cast<SequenceBatchScheduler*>(scheduler_.get());
    if (sched == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "Unable to downcast from 'Scheduler' to 'SequenceBatchScheduler' "
          "during scheduler update");
    }
    return sched->Update(added_instances, removed_instances);
  }

  // Non-sequence scheduler does not need to be updated, because other
  // schedulers do not require the information on model instances to function,
  // and only interact with the rate limiter.
  return Status::Success;
}

Status
TritonModel::SetBatchingStrategy(const std::string& batch_libpath)
{
  // Load library and functions.
  std::unique_ptr<SharedLibrary> slib;
  RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));

  RETURN_IF_ERROR(slib->OpenLibraryHandle(batch_libpath, &batch_dlhandle_));
  RETURN_IF_ERROR(slib->GetEntrypoint(
      batch_dlhandle_, "TRITONBACKEND_ModelBatchIncludeRequest",
      true /* optional */, reinterpret_cast<void**>(&batch_incl_fn_)));
  RETURN_IF_ERROR(slib->GetEntrypoint(
      batch_dlhandle_, "TRITONBACKEND_ModelBatchInitialize",
      true /* optional */, reinterpret_cast<void**>(&batch_init_fn_)));
  RETURN_IF_ERROR(slib->GetEntrypoint(
      batch_dlhandle_, "TRITONBACKEND_ModelBatchFinalize", true /* optional */,
      reinterpret_cast<void**>(&batch_fini_fn_)));
  RETURN_IF_ERROR(slib->GetEntrypoint(
      batch_dlhandle_, "TRITONBACKEND_ModelBatcherFinalize",
      true /* optional */, reinterpret_cast<void**>(&batcher_fini_fn_)));
  RETURN_IF_ERROR(slib->GetEntrypoint(
      batch_dlhandle_, "TRITONBACKEND_ModelBatcherInitialize",
      true /* optional */, reinterpret_cast<void**>(&batcher_init_fn_)));

  // If one custom batching function is defined, all must be.
  const bool defined_some = batch_incl_fn_ || batch_init_fn_ ||
                            batch_fini_fn_ || batcher_init_fn_ ||
                            batcher_fini_fn_;
  const bool defined_all = batch_incl_fn_ && batch_init_fn_ && batch_fini_fn_ &&
                           batcher_init_fn_ && batcher_fini_fn_;
  if (defined_some && !defined_all) {
    ClearHandles();
    return Status(
        Status::Code::INVALID_ARG,
        batch_libpath +
            " does not define all "
            "required custom batching functions for model " +
            config_.name());
  }
  // If a custom batcher is provided, initialize it.
  if (batcher_init_fn_) {
    TRITONSERVER_Error* err = batcher_init_fn_(
        Batcher(), reinterpret_cast<TRITONBACKEND_Model*>(this));
    if (err) {
      auto status = Status(
          TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
          TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
      return status;
    }
  }

  return Status::Success;
}

TritonModel::TritonModel(
    InferenceServer* server,
    const std::shared_ptr<LocalizedPath>& localized_model_dir,
    const std::shared_ptr<TritonBackend>& backend,
    const double min_compute_capability, const ModelIdentifier& model_id,
    const int64_t version, const inference::ModelConfig& config,
    const bool auto_complete_config,
    const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
    const triton::common::HostPolicyCmdlineConfigMap& host_policy_map)
    : Model(
          min_compute_capability, localized_model_dir->Path(), model_id,
          version, config),
      server_(server), min_compute_capability_(min_compute_capability),
      auto_complete_config_(auto_complete_config),
      backend_cmdline_config_map_(backend_cmdline_config_map),
      host_policy_map_(host_policy_map), device_blocking_(false),
      localized_model_dir_(localized_model_dir), backend_(backend),
      state_(nullptr)
{
}

TritonModel::~TritonModel()
{
  // If there is a custom batcher, finalize it.
  if (batcher_fini_fn_) {
    TRITONSERVER_Error* err = batcher_fini_fn_(*Batcher());
    *Batcher() = nullptr;
    if (err) {
      LOG_ERROR << "Custom batcher finalization failed for model "
                << config_.name() << ": " << TRITONSERVER_ErrorMessage(err);
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Clear library handles.
  ClearHandles();

  // Explicitly delete/finalize the scheduler before the model instances.
  scheduler_.reset(nullptr);

  // Explicitly delete/finalize all model instances before finalizing
  // the model itself.
  instances_.clear();
  passive_instances_.clear();
  ClearBackgroundInstances();

  // Unregister itself from the rate limiter. Note this should happen
  // after all instances are destructed. Destrucing instances ensures
  // there are no instance threads waiting on rate limiter for
  // receiving their payloads.
  server_->GetRateLimiter()->UnregisterModel(this);

  // Model finalization is optional... The TRITONBACKEND_Model
  // object is this TritonModel object.
  if (backend_->ModelFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        backend_->ModelFiniFn()(reinterpret_cast<TRITONBACKEND_Model*>(this)),
        "failed finalizing model");
  }
}

void
TritonModel::ClearHandles()
{
  if (batch_dlhandle_ == nullptr) {
    return;
  }

  {
    std::unique_ptr<SharedLibrary> slib;
    LOG_STATUS_ERROR(
        SharedLibrary::Acquire(&slib), "~TritonModel::ClearHandles");
    LOG_STATUS_ERROR(
        slib->CloseLibraryHandle(batch_dlhandle_), "TritonModel::ClearHandles");
  }
  batch_dlhandle_ = nullptr;
  batch_incl_fn_ = nullptr;
  batch_init_fn_ = nullptr;
  batch_fini_fn_ = nullptr;
  batcher_init_fn_ = nullptr;
  batcher_fini_fn_ = nullptr;
}

extern "C" {

//
// TRITONBACKEND_Model
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelName(TRITONBACKEND_Model* model, const char** name)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *name = tm->Name().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelVersion(TRITONBACKEND_Model* model, uint64_t* version)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *version = tm->Version();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelRepository(
    TRITONBACKEND_Model* model, TRITONBACKEND_ArtifactType* artifact_type,
    const char** location)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *artifact_type = TRITONBACKEND_ARTIFACT_FILESYSTEM;
  *location = tm->LocalizedModelPath().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelConfig(
    TRITONBACKEND_Model* model, const uint32_t config_version,
    TRITONSERVER_Message** model_config)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);

  std::string model_config_json;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      ModelConfigToJson(tm->Config(), config_version, &model_config_json));

  *model_config = reinterpret_cast<TRITONSERVER_Message*>(
      new TritonServerMessage(std::move(model_config_json)));

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelAutoCompleteConfig(
    TRITONBACKEND_Model* model, bool* auto_complete_config)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *auto_complete_config = tm->AutoCompleteConfig();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelSetConfig(
    TRITONBACKEND_Model* model, const uint32_t config_version,
    TRITONSERVER_Message* model_config)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      tm->UpdateModelConfig(config_version, model_config));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelServer(
    TRITONBACKEND_Model* model, TRITONSERVER_Server** server)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *server = reinterpret_cast<TRITONSERVER_Server*>(tm->Server());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBackend(
    TRITONBACKEND_Model* model, TRITONBACKEND_Backend** backend)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *backend = reinterpret_cast<TRITONBACKEND_Backend*>(tm->Backend().get());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelState(TRITONBACKEND_Model* model, void** state)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  *state = tm->State();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelSetState(TRITONBACKEND_Model* model, void* state)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  tm->SetState(state);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelReportMemoryUsage(
    TRITONBACKEND_Model* model, TRITONSERVER_BufferAttributes** usage,
    uint32_t usage_size)
{
  TritonModel* tm = reinterpret_cast<TritonModel*>(model);
  tm->SetMemoryUsage({reinterpret_cast<BufferAttributes**>(usage), usage_size});
  return nullptr;  // success
}

///
/// TRITONBACKEND_Request
///
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestId(TRITONBACKEND_Request* request, const char** id)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  *id = tr->Id().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestCorrelationId(TRITONBACKEND_Request* request, uint64_t* id)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const InferenceRequest::SequenceId& correlation_id = tr->CorrelationId();
  if (correlation_id.Type() != InferenceRequest::SequenceId::DataType::UINT64) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (tr->LogRequest() + "correlation ID in request is not an unsigned int")
            .c_str());
  }
  *id = correlation_id.UnsignedIntValue();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InferenceRequestTimeoutMicroseconds(
    TRITONBACKEND_Request* request, uint64_t* timeout)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  *timeout = tr->TimeoutMicroseconds();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestFlags(TRITONBACKEND_Request* request, uint32_t* flags)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  *flags = tr->Flags();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestIsCancelled(
    TRITONBACKEND_Request* request, bool* is_cancelled)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);

  RETURN_TRITONSERVER_ERROR_IF_ERROR(tr->IsCancelled(is_cancelled));
  return nullptr;
}


TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestCorrelationIdString(
    TRITONBACKEND_Request* request, const char** id)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const InferenceRequest::SequenceId& correlation_id = tr->CorrelationId();
  if (correlation_id.Type() != InferenceRequest::SequenceId::DataType::STRING) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (tr->LogRequest() + "correlation ID in request is not a string")
            .c_str());
  }
  *id = correlation_id.StringValue().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestInputCount(TRITONBACKEND_Request* request, uint32_t* count)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  *count = tr->ImmutableInputs().size();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestInputName(
    TRITONBACKEND_Request* request, const uint32_t index,
    const char** input_name)
{
  *input_name = nullptr;

  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const auto& inputs = tr->ImmutableInputs();
  if (index >= inputs.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (tr->LogRequest() + "out of bounds index " + std::to_string(index) +
         ": request has " + std::to_string(inputs.size()) + " inputs")
            .c_str());
  }

  // The request inputs are not allowed to change once the request
  // makes it to the backend, so it is ok to just iterate through the
  // map. This linear search is the best we can do given the need for
  // the inputs to be in a map and given the typical small number of
  // inputs is better than having every request maintain the inputs as
  // both map and vector.
  uint32_t cnt = 0;
  for (const auto& pr : inputs) {
    if (cnt++ == index) {
      InferenceRequest::Input* in = pr.second;
      *input_name = in->Name().c_str();
      break;
    }
  }

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestInput(
    TRITONBACKEND_Request* request, const char* name,
    TRITONBACKEND_Input** input)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const auto& inputs = tr->ImmutableInputs();
  const auto& itr = inputs.find(name);
  if (itr == inputs.end()) {
    *input = nullptr;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (tr->LogRequest() + "unknown request input name " + name).c_str());
  }

  InferenceRequest::Input* in = itr->second;
  *input = reinterpret_cast<TRITONBACKEND_Input*>(in);

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestInputByIndex(
    TRITONBACKEND_Request* request, const uint32_t index,
    TRITONBACKEND_Input** input)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const auto& inputs = tr->ImmutableInputs();
  if (index >= inputs.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (tr->LogRequest() + "out of bounds index " + std::to_string(index) +
         ": request has " + std::to_string(inputs.size()) + " inputs")
            .c_str());
  }

  // The request inputs are not allowed to change once the request
  // makes it to the backend, so it is ok to just iterate through the
  // map. This linear search is the best we can do given the need for
  // the inputs to be in a map and given the typical small number of
  // inputs is better than having every request maintain the inputs as
  // both map and vector.
  uint32_t cnt = 0;
  for (const auto& pr : inputs) {
    if (cnt++ == index) {
      InferenceRequest::Input* in = pr.second;
      *input = reinterpret_cast<TRITONBACKEND_Input*>(in);
      break;
    }
  }

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestOutputCount(
    TRITONBACKEND_Request* request, uint32_t* count)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  *count = tr->ImmutableRequestedOutputs().size();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestOutputName(
    TRITONBACKEND_Request* request, const uint32_t index,
    const char** output_name)
{
  *output_name = nullptr;

  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const auto& routputs = tr->ImmutableRequestedOutputs();
  if (index >= routputs.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (tr->LogRequest() + "out of bounds index " + std::to_string(index) +
         ": request has " + std::to_string(routputs.size()) +
         " requested outputs")
            .c_str());
  }

  // The requested outputs are not allowed to change once the request
  // makes it to the backend, so it is ok to just iterate through the
  // set. This linear search is the best we can do given the requested
  // outputs being in a set and given the typical small number of
  // requested outputs it should not be a performance issue.
  uint32_t cnt = 0;
  for (const auto& rout : routputs) {
    if (cnt++ == index) {
      *output_name = rout.c_str();
      break;
    }
  }

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestOutputBufferProperties(
    TRITONBACKEND_Request* request, const char* name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      tr->OutputBufferProperties(name, byte_size, memory_type, memory_type_id));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestRelease(
    TRITONBACKEND_Request* request, uint32_t release_flags)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  std::unique_ptr<InferenceRequest> ur(tr);
  auto status = InferenceRequest::Release(std::move(ur), release_flags);
  if (!status.IsOk()) {
    // On error, ownership of request is not taken and should not be
    // managed by unique pointer.
    ur.release();
    RETURN_TRITONSERVER_ERROR_IF_ERROR(status);
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestTrace(
    TRITONBACKEND_Request* request, TRITONSERVER_InferenceTrace** trace)
{
#ifdef TRITON_ENABLE_TRACING
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  if (tr->TraceProxy() != nullptr) {
    *trace = reinterpret_cast<TRITONSERVER_InferenceTrace*>(
        tr->TraceProxy()->Trace());
  } else {
    *trace = nullptr;
  }
  return nullptr;  // success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "tracing is not supported");
#endif  // TRITON_ENABLE_TRACING
}

///
/// TRITONBACKEND_State
///

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_StateUpdate(TRITONBACKEND_State* state)
{
  SequenceState* ts = reinterpret_cast<SequenceState*>(state);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(ts->Update());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_StateNew(
    TRITONBACKEND_State** state, TRITONBACKEND_Request* request,
    const char* name, const TRITONSERVER_DataType datatype,
    const int64_t* shape, const uint32_t dims_count)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  SequenceState* lstate;
  std::vector<int64_t> lshape(shape, shape + dims_count);
  auto& sequence_state = tr->GetSequenceStates();

  if (sequence_state == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to add state '") + name +
         "'. State configuration is missing for model '" + tr->ModelName() +
         "'.")
            .c_str());
  }

  RETURN_TRITONSERVER_ERROR_IF_ERROR(sequence_state->OutputState(
      name, TritonToDataType(datatype), lshape, &lstate));
  *state = reinterpret_cast<TRITONBACKEND_State*>(lstate);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_StateBuffer(
    TRITONBACKEND_State* state, void** buffer, const uint64_t buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  SequenceState* to = reinterpret_cast<SequenceState*>(state);
  Status status = Status::Success;

  TRITONSERVER_MemoryType current_memory_type;
  int64_t current_memory_type_id;
  void* lbuffer = nullptr;
  MutableMemory* mutable_memory =
      reinterpret_cast<MutableMemory*>(to->Data().get());
  lbuffer = mutable_memory->MutableBuffer(
      &current_memory_type, &current_memory_type_id);

  // If the buffer size exactly matches the buffer available and is requesting
  // the same memory type and memory type id, reuse the currently allocated
  // buffer.
  if (to->Data()->TotalByteSize() == buffer_byte_size &&
      current_memory_type == *memory_type &&
      current_memory_type_id == *memory_type_id) {
    *buffer = lbuffer;
  } else {
    RETURN_TRITONSERVER_ERROR_IF_ERROR(to->ResizeOrReallocate(
        buffer, buffer_byte_size, memory_type, memory_type_id));
  }

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_StateBufferAttributes(
    TRITONBACKEND_State* state,
    TRITONSERVER_BufferAttributes** buffer_attributes)
{
  SequenceState* to = reinterpret_cast<SequenceState*>(state);
  to->Data()->BufferAt(
      0, reinterpret_cast<BufferAttributes**>(buffer_attributes));

  return nullptr;  // success
}

//
// TRITONBACKEND_ResponseFactory
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactoryNew(
    TRITONBACKEND_ResponseFactory** factory, TRITONBACKEND_Request* request)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  std::shared_ptr<InferenceResponseFactory>* response_factory =
      new std::shared_ptr<InferenceResponseFactory>(tr->ResponseFactory());

  *factory = reinterpret_cast<TRITONBACKEND_ResponseFactory*>(response_factory);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactoryDelete(TRITONBACKEND_ResponseFactory* factory)
{
  std::shared_ptr<InferenceResponseFactory>* response_factory =
      reinterpret_cast<std::shared_ptr<InferenceResponseFactory>*>(factory);
  delete response_factory;
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactorySendFlags(
    TRITONBACKEND_ResponseFactory* factory, const uint32_t send_flags)
{
  std::shared_ptr<InferenceResponseFactory>* response_factory =
      reinterpret_cast<std::shared_ptr<InferenceResponseFactory>*>(factory);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      (*response_factory)->SendFlags(send_flags));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactoryIsCancelled(
    TRITONBACKEND_ResponseFactory* factory, bool* is_cancelled)
{
  std::shared_ptr<InferenceResponseFactory>* response_factory =
      reinterpret_cast<std::shared_ptr<InferenceResponseFactory>*>(factory);
  *is_cancelled = (*response_factory)->IsCancelled();
  return nullptr;  // success
}


///
/// TRITONBACKEND_Response
///
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseNew(
    TRITONBACKEND_Response** response, TRITONBACKEND_Request* request)
{
  *response = nullptr;
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);

  std::unique_ptr<InferenceResponse> tresp;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(
      tr->ResponseFactory()->CreateResponse(&tresp));

  *response = reinterpret_cast<TRITONBACKEND_Response*>(tresp.release());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseNewFromFactory(
    TRITONBACKEND_Response** response, TRITONBACKEND_ResponseFactory* factory)
{
  *response = nullptr;
  std::shared_ptr<InferenceResponseFactory>* response_factory =
      reinterpret_cast<std::shared_ptr<InferenceResponseFactory>*>(factory);

  std::unique_ptr<InferenceResponse> tr;
  RETURN_TRITONSERVER_ERROR_IF_ERROR((*response_factory)->CreateResponse(&tr));
  *response = reinterpret_cast<TRITONBACKEND_Response*>(tr.release());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseDelete(TRITONBACKEND_Response* response)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  delete tr;
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetStringParameter(
    TRITONBACKEND_Response* response, const char* name, const char* value)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tr->AddParameter(name, value));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetIntParameter(
    TRITONBACKEND_Response* response, const char* name, const int64_t value)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tr->AddParameter(name, value));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetBoolParameter(
    TRITONBACKEND_Response* response, const char* name, const bool value)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tr->AddParameter(name, value));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetDoubleParameter(
    TRITONBACKEND_Response* response, const char* name, const double value)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tr->AddParameter(name, value));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseOutput(
    TRITONBACKEND_Response* response, TRITONBACKEND_Output** output,
    const char* name, const TRITONSERVER_DataType datatype,
    const int64_t* shape, const uint32_t dims_count)
{
  *output = nullptr;
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  std::vector<int64_t> lshape(shape, shape + dims_count);
  InferenceResponse::Output* loutput;
  RETURN_TRITONSERVER_ERROR_IF_ERROR(tr->AddOutput(
      name, TritonToDataType(datatype), std::move(lshape), &loutput));
  *output = reinterpret_cast<TRITONBACKEND_Output*>(loutput);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSend(
    TRITONBACKEND_Response* response, const uint32_t send_flags,
    TRITONSERVER_Error* error)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);

  std::unique_ptr<InferenceResponse> utr(tr);
  if (error == nullptr) {
    RETURN_TRITONSERVER_ERROR_IF_ERROR(
        InferenceResponse::Send(std::move(utr), send_flags));
  } else {
    RETURN_TRITONSERVER_ERROR_IF_ERROR(InferenceResponse::SendWithStatus(
        std::move(utr), send_flags,
        Status(
            TritonCodeToStatusCode(TRITONSERVER_ErrorCode(error)),
            TRITONSERVER_ErrorMessage(error))));
  }

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestParameterCount(
    TRITONBACKEND_Request* request, uint32_t* count)
{
  InferenceRequest* lrequest = reinterpret_cast<InferenceRequest*>(request);

  const auto& parameters = lrequest->Parameters();
  *count = parameters.size();

  return nullptr;  // Success
}

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestParameter(
    TRITONBACKEND_Request* request, const uint32_t index, const char** key,
    TRITONSERVER_ParameterType* type, const void** vvalue)
{
  InferenceRequest* lrequest = reinterpret_cast<InferenceRequest*>(request);

  const auto& parameters = lrequest->Parameters();
  if (index >= parameters.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        ("out of bounds index " + std::to_string(index) +
         std::string(": request has ") + std::to_string(parameters.size()) +
         " parameters")
            .c_str());
  }

  const InferenceParameter& param = parameters[index];

  *key = param.Name().c_str();
  *type = param.Type();
  *vvalue = param.ValuePointer();

  return nullptr;  // Success
}

///
/// TRITONBACKEND_Input
///
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputProperties(
    TRITONBACKEND_Input* input, const char** name,
    TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint32_t* dims_count, uint64_t* byte_size, uint32_t* buffer_count)
{
  InferenceRequest::Input* ti =
      reinterpret_cast<InferenceRequest::Input*>(input);
  if (name != nullptr) {
    *name = ti->Name().c_str();
  }
  if (datatype != nullptr) {
    *datatype = DataTypeToTriton(ti->DType());
  }
  if (shape != nullptr) {
    *shape = ti->ShapeWithBatchDim().data();
  }
  if (dims_count != nullptr) {
    *dims_count = ti->ShapeWithBatchDim().size();
  }
  if (byte_size != nullptr) {
    *byte_size = ti->Data()->TotalByteSize();
  }
  if (buffer_count != nullptr) {
    *buffer_count = ti->DataBufferCount();
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputPropertiesForHostPolicy(
    TRITONBACKEND_Input* input, const char* host_policy_name, const char** name,
    TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint32_t* dims_count, uint64_t* byte_size, uint32_t* buffer_count)
{
  InferenceRequest::Input* ti =
      reinterpret_cast<InferenceRequest::Input*>(input);
  if (name != nullptr) {
    *name = ti->Name().c_str();
  }
  if (datatype != nullptr) {
    *datatype = DataTypeToTriton(ti->DType());
  }
  if (shape != nullptr) {
    *shape = ti->ShapeWithBatchDim().data();
  }
  if (dims_count != nullptr) {
    *dims_count = ti->ShapeWithBatchDim().size();
  }
  if (host_policy_name != nullptr) {
    if (byte_size != nullptr) {
      *byte_size = ti->Data(host_policy_name)->TotalByteSize();
    }
    if (buffer_count != nullptr) {
      *buffer_count = ti->DataBufferCountForHostPolicy(host_policy_name);
    }
  } else {
    if (byte_size != nullptr) {
      *byte_size = ti->Data()->TotalByteSize();
    }
    if (buffer_count != nullptr) {
      *buffer_count = ti->DataBufferCount();
    }
  }
  return nullptr;  // success
}


TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputBuffer(
    TRITONBACKEND_Input* input, const uint32_t index, const void** buffer,
    uint64_t* buffer_byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id)
{
  InferenceRequest::Input* ti =
      reinterpret_cast<InferenceRequest::Input*>(input);
  Status status = ti->DataBuffer(
      index, buffer, reinterpret_cast<size_t*>(buffer_byte_size), memory_type,
      memory_type_id);
  if (!status.IsOk()) {
    *buffer = nullptr;
    *buffer_byte_size = 0;
    RETURN_TRITONSERVER_ERROR_IF_ERROR(status);
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputBufferAttributes(
    TRITONBACKEND_Input* input, const uint32_t index, const void** buffer,
    TRITONSERVER_BufferAttributes** buffer_attributes)
{
  InferenceRequest::Input* ti =
      reinterpret_cast<InferenceRequest::Input*>(input);
  Status status = ti->DataBufferAttributes(
      index, buffer, reinterpret_cast<BufferAttributes**>(buffer_attributes));
  if (!status.IsOk()) {
    *buffer = nullptr;
    *buffer_attributes = nullptr;
    RETURN_TRITONSERVER_ERROR_IF_ERROR(status);
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InputBufferForHostPolicy(
    TRITONBACKEND_Input* input, const char* host_policy_name,
    const uint32_t index, const void** buffer, uint64_t* buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  InferenceRequest::Input* ti =
      reinterpret_cast<InferenceRequest::Input*>(input);

  Status status =
      (host_policy_name == nullptr)
          ? ti->DataBuffer(
                index, buffer, reinterpret_cast<size_t*>(buffer_byte_size),
                memory_type, memory_type_id)
          : ti->DataBufferForHostPolicy(
                index, buffer, reinterpret_cast<size_t*>(buffer_byte_size),
                memory_type, memory_type_id, host_policy_name);
  if (!status.IsOk()) {
    *buffer = nullptr;
    *buffer_byte_size = 0;
    RETURN_TRITONSERVER_ERROR_IF_ERROR(status);
  }
  return nullptr;  // success
}

///
/// TRITONBACKEND_Output
///
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_OutputBuffer(
    TRITONBACKEND_Output* output, void** buffer,
    const uint64_t buffer_byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id)
{
  InferenceResponse::Output* to =
      reinterpret_cast<InferenceResponse::Output*>(output);
  Status status = to->AllocateDataBuffer(
      buffer, buffer_byte_size, memory_type, memory_type_id);
  if (!status.IsOk()) {
    *buffer = nullptr;
    RETURN_TRITONSERVER_ERROR_IF_ERROR(status);
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_OutputBufferAttributes(
    TRITONBACKEND_Output* output,
    TRITONSERVER_BufferAttributes** buffer_attributes)
{
  InferenceResponse::Output* to =
      reinterpret_cast<InferenceResponse::Output*>(output);

  *buffer_attributes = reinterpret_cast<TRITONSERVER_BufferAttributes*>(
      to->GetBufferAttributes());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
    TRITONBACKEND_BackendAttribute* backend_attributes,
    const TRITONSERVER_InstanceGroupKind kind, const uint64_t count,
    const uint64_t* device_ids, const uint64_t id_count)
{
  auto ba = reinterpret_cast<TritonBackend::Attribute*>(backend_attributes);
  ba->preferred_groups_.emplace_back();
  auto& pg = ba->preferred_groups_.back();
  switch (kind) {
    case TRITONSERVER_INSTANCEGROUPKIND_AUTO:
      pg.set_kind(inference::ModelInstanceGroup::KIND_AUTO);
      break;
    case TRITONSERVER_INSTANCEGROUPKIND_CPU:
      pg.set_kind(inference::ModelInstanceGroup::KIND_CPU);
      break;
    case TRITONSERVER_INSTANCEGROUPKIND_GPU:
      pg.set_kind(inference::ModelInstanceGroup::KIND_GPU);
      break;
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL:
      pg.set_kind(inference::ModelInstanceGroup::KIND_MODEL);
      break;
  }
  pg.set_count(count);
  if (device_ids != nullptr) {
    for (size_t i = 0; i < id_count; ++i) {
      pg.add_gpus(device_ids[i]);
    }
  }
  return nullptr;
}


TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
    TRITONBACKEND_BackendAttribute* backend_attributes, bool enabled)
{
  auto ba = reinterpret_cast<TritonBackend::Attribute*>(backend_attributes);
  ba->parallel_instance_loading_ = enabled;
  return nullptr;
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InferenceResponseOutputByName(
    TRITONBACKEND_Response* response, const char* name,
    TRITONSERVER_DataType* datatype, const int64_t** shape, uint64_t* dim_count)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);

  const auto& outputs = tr->Outputs();
  uint32_t output_count = outputs.size();
  std::string output_name = std::string(name);

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    if (outputs[idx].Name() == output_name) {
      *datatype = DataTypeToTriton(outputs[idx].DType());
      const std::vector<int64_t>& oshape = outputs[idx].Shape();
      *shape = &oshape[0];
      *dim_count = oshape.size();
      return nullptr;  // success
    }
  }
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_NOT_FOUND,
      ("Output name " + output_name + "not found.").c_str());
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_InferenceResponseOutput(
    TRITONBACKEND_Response* response, const uint32_t index, const char** name,
    TRITONSERVER_DataType* datatype, const int64_t** shape, uint64_t* dim_count)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);

  const auto& outputs = tr->Outputs();
  if (index >= outputs.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        ("out of bounds index " + std::to_string(index) +
         std::string(": response has ") + std::to_string(outputs.size()) +
         " outputs")
            .c_str());
  }

  const InferenceResponse::Output& output = outputs[index];

  *name = output.Name().c_str();
  *datatype = DataTypeToTriton(output.DType());

  const std::vector<int64_t>& oshape = output.Shape();
  *shape = &oshape[0];
  *dim_count = oshape.size();

  return nullptr;  // success
}

}  // extern C

}}  // namespace triton::core
