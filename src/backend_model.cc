// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "backend_config.h"
#include "backend_model_instance.h"
#include "dynamic_batch_scheduler.h"
#include "filesystem.h"
#include "model_config_utils.h"
#include "numa_utils.h"
#include "sequence_batch_scheduler.h"
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
    const std::string& model_name, const int64_t version,
    const inference::ModelConfig& model_config,
    std::unique_ptr<TritonModel>* model)
{
  model->reset();

  // The model configuration must specify a backend. The name of the
  // corresponding shared library must be libtriton_<backend>.so.
  if (model_config.backend().empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "must specify 'backend' for '" + model_config.name() + "'");
  }

  // Localize the content of the model repository corresponding to
  // 'model_name'. This model holds a handle to the localized content
  // so that it persists as long as the model is loaded.
  std::shared_ptr<LocalizedDirectory> localized_model_dir;
  RETURN_IF_ERROR(LocalizeDirectory(model_path, &localized_model_dir));

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
      backend_cmdline_config_map, model_config.backend(),
      &specialized_backend_name));

  std::string backend_libname;
  RETURN_IF_ERROR(BackendConfigurationBackendLibraryName(
      specialized_backend_name, &backend_libname));

  // Get the path to the backend shared library. Search path is
  // version directory, model directory, global backend directory.
  const auto localized_model_path = localized_model_dir->Path();
  const auto version_path =
      JoinPath({localized_model_path, std::to_string(version)});
  const std::string global_path =
      JoinPath({backend_dir, specialized_backend_name});
  const std::vector<std::string> search_paths = {
      version_path, localized_model_path, global_path};

  std::string backend_libdir;
  std::string backend_libpath;
  for (const auto& path : search_paths) {
    const auto full_path = JoinPath({path, backend_libname});
    bool exists = false;
    RETURN_IF_ERROR(FileExists(full_path, &exists));
    if (exists) {
      backend_libdir = path;
      backend_libpath = full_path;
      break;
    }
  }

  if (backend_libpath.empty()) {
    return Status(
        Status::Code::INVALID_ARG, "unable to find '" + backend_libname +
                                       "' for model '" + model_config.name() +
                                       "', searched: " + version_path + ", " +
                                       model_path + ", " + global_path);
  }

  // Resolve the global backend configuration with the specific backend
  // configuration
  triton::common::BackendCmdlineConfig config;
  RETURN_IF_ERROR(ResolveBackendConfigs(
      backend_cmdline_config_map, model_config.backend(), config));

  RETURN_IF_ERROR(SetBackendConfigDefaults(config));

  std::shared_ptr<TritonBackend> backend;
  RETURN_IF_ERROR(server->BackendManager()->CreateBackend(
      model_config.backend(), backend_libdir, backend_libpath, config,
      &backend));

  // Create and initialize the model.
  std::unique_ptr<TritonModel> local_model(new TritonModel(
      server, localized_model_dir, backend, min_compute_capability, version,
      model_config, auto_complete_config));

  TritonModel* raw_local_model = local_model.get();

  // Model initialization is optional... The TRITONBACKEND_Model
  // object is this TritonModel object. We must set set shared library
  // path to point to the backend directory in case the backend
  // library attempts to load additional shared libaries.
  if (backend->ModelInitFn() != nullptr) {
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->SetLibraryDirectory(backend->Directory()));

    TRITONSERVER_Error* err = backend->ModelInitFn()(
        reinterpret_cast<TRITONBACKEND_Model*>(raw_local_model));

    RETURN_IF_ERROR(slib->ResetLibraryDirectory());
    RETURN_IF_TRITONSERVER_ERROR(err);
  }

  // Initialize the model for Triton core usage
  RETURN_IF_ERROR(local_model->Init());

  bool device_blocking = false;
  if (local_model->backend_->ExecutionPolicy() ==
      TRITONBACKEND_EXECUTION_DEVICE_BLOCKING) {
    if (model_config.has_sequence_batching()) {
      LOG_INFO << "Overriding execution policy to "
                  "\"TRITONBACKEND_EXECUTION_BLOCKING\" for sequence model \""
               << model_config.name() << "\"";
    } else {
      device_blocking = true;
    }
  }

  // Create and initialize the model instances for this model.
  RETURN_IF_ERROR(TritonModelInstance::CreateInstances(
      raw_local_model, host_policy_map, model_config, device_blocking));

  RETURN_IF_ERROR(local_model->SetConfiguredScheduler());

  *model = std::move(local_model);
  return Status::Success;
}

Status
TritonModel::ResolveBackendConfigs(
    const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
    const std::string& backend_name,
    triton::common::BackendCmdlineConfig& config)
{
  const auto& global_itr = backend_cmdline_config_map.find(std::string());
  const auto& specific_itr = backend_cmdline_config_map.find(backend_name);
  if (specific_itr == backend_cmdline_config_map.end() &&
      global_itr != backend_cmdline_config_map.end()) {
    for (auto setting : global_itr->second) {
      config.push_back(setting);
    }
  } else if (
      specific_itr != backend_cmdline_config_map.end() &&
      global_itr == backend_cmdline_config_map.end()) {
    for (auto setting : specific_itr->second) {
      config.push_back(setting);
    }
  } else if (
      specific_itr != backend_cmdline_config_map.end() &&
      global_itr != backend_cmdline_config_map.end()) {
    triton::common::BackendCmdlineConfig global_backend_config =
        global_itr->second;
    triton::common::BackendCmdlineConfig specific_backend_config =
        specific_itr->second;

    std::sort(global_backend_config.begin(), global_backend_config.end());
    std::sort(specific_backend_config.begin(), specific_backend_config.end());

    size_t global_index = 0;
    size_t specific_index = 0;
    while (global_index < global_backend_config.size() &&
           specific_index < specific_backend_config.size()) {
      auto& current_global_setting = global_backend_config.at(global_index);
      auto& current_specific_setting =
          specific_backend_config.at(specific_index);
      if (current_specific_setting.first.compare(
              current_global_setting.first) == 0) {
        // specific setting overrides global setting
        config.push_back(current_specific_setting);
        ++global_index;
        ++specific_index;
      } else if (
          current_specific_setting.first.compare(current_global_setting.first) <
          0) {
        config.push_back(current_specific_setting);
        ++specific_index;
      } else {
        config.push_back(current_global_setting);
        ++global_index;
      }
    }

    // add the rest of the global configs
    if (global_index < global_backend_config.size()) {
      auto& current_global_setting = global_backend_config.at(global_index);
      config.push_back(current_global_setting);
    }

    // add the rest of the specific settings
    if (specific_index < specific_backend_config.size()) {
      auto& current_specific_setting =
          specific_backend_config.at(specific_index);
      config.push_back(current_specific_setting);
    }
  }  // else empty config

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

Status
TritonModel::AddInstance(
    std::unique_ptr<TritonModelInstance>&& instance, const bool passive)
{
  if (passive) {
    passive_instances_.emplace_back(std::move(instance));
  } else {
    instances_.emplace_back(std::move(instance));
  }

  return Status::Success;
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

  RETURN_IF_ERROR(SetModelConfig(config));
  return Status::Success;
}

Status
TritonModel::SetConfiguredScheduler()
{
  std::unique_ptr<Scheduler> scheduler;

  // Need to enforce equal shape batches (i.e. non-ragged batches) if
  // the model 1) allows one or more variable-size input tensors that
  // are not marked as 'allow_ragged_batch' or 2) has one or more
  // shape-tensor inputs. This is not needed if all input shapes are
  // non-variable and if there are no shape tensors... so we don't
  // enable it in that case for efficiency reasons.
  std::unordered_map<std::string, bool> enforce_equal_shape_tensors;
  for (const auto input : config_.input()) {
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
    RETURN_IF_ERROR(SequenceBatchScheduler::Create(
        this, enforce_equal_shape_tensors, &scheduler));
  } else if (config_.has_dynamic_batching()) {
    // Dynamic batcher
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        this, nullptr, 0 /*nice*/, true /* dynamic_batching_enabled */,
        config_.max_batch_size(), enforce_equal_shape_tensors,
        config_.dynamic_batching(),
        config_.response_cache().enable() /* response_cache_enable */,
        &scheduler));
  } else {
    // Default scheduler. Use dynamic batch scheduler (with batching
    // disabled) as the default scheduler.
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        this, nullptr, 0 /*nice*/, false /* dynamic_batching_enabled */,
        1 /* max_batch_size */,
        std::unordered_map<
            std::string, bool>() /* enforce_equal_shape_tensors */,
        false /* preserve_ordering */,
        config_.response_cache().enable() /* response_cache_enable */,
        std::set<int32_t>() /* preferred_batch_sizes */,
        0 /* max_queue_delay_microseconds */, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

Status
TritonModel::Initialize()
{
  for (const auto& instance : instances_) {
    RETURN_IF_ERROR(instance->Initialize());
  }

  return Status::Success;
}

Status
TritonModel::WarmUp()
{
  for (const auto& instance : instances_) {
    RETURN_IF_ERROR(instance->WarmUp());
  }

  return Status::Success;
}

TritonModel::TritonModel(
    InferenceServer* server,
    const std::shared_ptr<LocalizedDirectory>& localized_model_dir,
    const std::shared_ptr<TritonBackend>& backend,
    const double min_compute_capability, const int64_t version,
    const inference::ModelConfig& config, const bool auto_complete_config)
    : Model(
          min_compute_capability, localized_model_dir->Path(), version, config),
      server_(server), auto_complete_config_(auto_complete_config),
      localized_model_dir_(localized_model_dir), backend_(backend),
      state_(nullptr)
{
}

TritonModel::~TritonModel()
{
  // Explicitly delete/finalize all model instances before finalizing
  // the model itself.
  instances_.clear();
  passive_instances_.clear();

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
  Status status =
      ModelConfigToJson(tm->Config(), config_version, &model_config_json);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

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
  Status status = tm->UpdateModelConfig(config_version, model_config);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
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
        std::string("correlation ID in request is not an unsigned int")
            .c_str());
  }
  *id = correlation_id.UnsignedIntValue();
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
TRITONBACKEND_RequestCorrelationIdString(
    TRITONBACKEND_Request* request, const char** id)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  const InferenceRequest::SequenceId& correlation_id = tr->CorrelationId();
  if (correlation_id.Type() != InferenceRequest::SequenceId::DataType::STRING) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("correlation ID in request is not a string").c_str());
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
        (std::string("out of bounds index ") + std::to_string(index) +
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
        (std::string("unknown request input name ") + name).c_str());
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
        (std::string("out of bounds index ") + std::to_string(index) +
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
        (std::string("out of bounds index ") + std::to_string(index) +
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
  auto status =
      tr->OutputBufferProperties(name, byte_size, memory_type, memory_type_id);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestRelease(
    TRITONBACKEND_Request* request, uint32_t release_flags)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  std::unique_ptr<InferenceRequest> ur(tr);
  InferenceRequest::Release(std::move(ur), release_flags);
  return nullptr;  // success
}

///
/// TRITONBACKEND_State
///

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_StateUpdate(TRITONBACKEND_State* state)
{
  SequenceState* ts = reinterpret_cast<SequenceState*>(state);
  auto status = ts->Update();

  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

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

  Status status = sequence_state->OutputState(
      name, TritonToDataType(datatype), lshape, &lstate);
  if (!status.IsOk()) {
    *state = nullptr;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

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

  // If the buffer size exactly matches the buffer available, reuse the
  // currently allocated buffer.
  if (to->Data()->TotalByteSize() == buffer_byte_size) {
    const std::shared_ptr<AllocatedMemory>& memory =
        reinterpret_cast<const std::shared_ptr<AllocatedMemory>&>(to->Data());

    TRITONSERVER_MemoryType current_memory_type;
    int64_t current_memory_type_id;
    void* lbuffer =
        memory->MutableBuffer(&current_memory_type, &current_memory_type_id);

    // If the requested memory type doesn't match the current buffer, allocate a
    // new buffer with the requested memory type and memory type id.
    if (current_memory_type == *memory_type &&
        current_memory_type_id == *memory_type_id) {
      *buffer = lbuffer;
    } else {
      std::shared_ptr<AllocatedMemory> memory =
          std::make_shared<AllocatedMemory>(
              buffer_byte_size, *memory_type, *memory_type_id);
      *buffer = memory->MutableBuffer(memory_type, memory_type_id);
      to->RemoveAllData();
      status = to->SetData(memory);
    }
  } else {
    std::shared_ptr<AllocatedMemory> memory = std::make_shared<AllocatedMemory>(
        buffer_byte_size, *memory_type, *memory_type_id);
    *buffer = memory->MutableBuffer(memory_type, memory_type_id);
    to->RemoveAllData();
    status = to->SetData(memory);
  }

  if (!status.IsOk()) {
    *buffer = nullptr;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
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
  InferenceResponseFactory* response_factory =
      new InferenceResponseFactory(tr->ResponseFactory());
  *factory = reinterpret_cast<TRITONBACKEND_ResponseFactory*>(response_factory);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactoryDelete(TRITONBACKEND_ResponseFactory* factory)
{
  InferenceResponseFactory* tf =
      reinterpret_cast<InferenceResponseFactory*>(factory);
  delete tf;
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseFactorySendFlags(
    TRITONBACKEND_ResponseFactory* factory, const uint32_t send_flags)
{
  InferenceResponseFactory* tf =
      reinterpret_cast<InferenceResponseFactory*>(factory);
  Status status = tf->SendFlags(send_flags);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
  return nullptr;  // success
}

///
/// TRITONBACKEND_Response
///
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseNew(
    TRITONBACKEND_Response** response, TRITONBACKEND_Request* request)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);

  std::unique_ptr<InferenceResponse> tresp;
  Status status = tr->ResponseFactory().CreateResponse(&tresp);
  if (!status.IsOk()) {
    *response = nullptr;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

  *response = reinterpret_cast<TRITONBACKEND_Response*>(tresp.release());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseNewFromFactory(
    TRITONBACKEND_Response** response, TRITONBACKEND_ResponseFactory* factory)
{
  InferenceResponseFactory* tf =
      reinterpret_cast<InferenceResponseFactory*>(factory);

  std::unique_ptr<InferenceResponse> tr;
  Status status = tf->CreateResponse(&tr);
  if (!status.IsOk()) {
    *response = nullptr;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

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
  Status status = tr->AddParameter(name, value);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetIntParameter(
    TRITONBACKEND_Response* response, const char* name, const int64_t value)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  Status status = tr->AddParameter(name, value);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSetBoolParameter(
    TRITONBACKEND_Response* response, const char* name, const bool value)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  Status status = tr->AddParameter(name, value);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseOutput(
    TRITONBACKEND_Response* response, TRITONBACKEND_Output** output,
    const char* name, const TRITONSERVER_DataType datatype,
    const int64_t* shape, const uint32_t dims_count)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);
  std::vector<int64_t> lshape(shape, shape + dims_count);
  InferenceResponse::Output* loutput;
  Status status = tr->AddOutput(
      name, TritonToDataType(datatype), std::move(lshape), &loutput);
  if (!status.IsOk()) {
    *output = nullptr;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

  *output = reinterpret_cast<TRITONBACKEND_Output*>(loutput);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ResponseSend(
    TRITONBACKEND_Response* response, const uint32_t send_flags,
    TRITONSERVER_Error* error)
{
  InferenceResponse* tr = reinterpret_cast<InferenceResponse*>(response);

  Status status;

  std::unique_ptr<InferenceResponse> utr(tr);
  if (error == nullptr) {
    status = InferenceResponse::Send(std::move(utr), send_flags);
  } else {
    status = InferenceResponse::SendWithStatus(
        std::move(utr), send_flags,
        Status(
            TritonCodeToStatusCode(TRITONSERVER_ErrorCode(error)),
            TRITONSERVER_ErrorMessage(error)));
  }

  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }

  return nullptr;  // success
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
      index, buffer, buffer_byte_size, memory_type, memory_type_id);
  if (!status.IsOk()) {
    *buffer = nullptr;
    *buffer_byte_size = 0;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
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
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
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
                index, buffer, buffer_byte_size, memory_type, memory_type_id)
          : ti->DataBufferForHostPolicy(
                index, buffer, buffer_byte_size, memory_type, memory_type_id,
                host_policy_name);
  if (!status.IsOk()) {
    *buffer = nullptr;
    *buffer_byte_size = 0;
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
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
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
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
}  // extern C

}}  // namespace triton::core
