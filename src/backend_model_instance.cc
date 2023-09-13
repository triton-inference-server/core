// Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "backend_model_instance.h"

#include "status.h"

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "backend_config.h"
#include "backend_model.h"
#include "cuda_utils.h"
#include "metrics.h"
#include "model_config.pb.h"
#include "numa_utils.h"
#include "server.h"
#include "shared_library.h"
#include "triton/common/logging.h"
#include "triton/common/nvtx.h"
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

namespace {
// Utilities for warmup feature
TRITONSERVER_Error*
WarmupResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  *buffer = malloc(byte_size);
  if (*buffer != nullptr) {
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    *actual_memory_type_id = 0;
    return nullptr;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "failed to allocate output buffer for warmup.");
}

TRITONSERVER_Error*
WarmupResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  free(buffer);
  return nullptr;
}

ResponseAllocator warmup_allocator = ResponseAllocator(
    WarmupResponseAlloc, WarmupResponseRelease, nullptr /* start_fn */);

void
WarmupResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  auto res_pair = reinterpret_cast<
      std::pair<std::promise<void>, std::vector<std::string>*>*>(userp);
  if (iresponse != nullptr) {
    auto err = TRITONSERVER_InferenceResponseError(iresponse);
    if (err != nullptr) {
      // The error vector is shared by all requests in the batch for now
      static std::mutex res_mtx;
      {
        std::lock_guard<std::mutex> lk(res_mtx);
        res_pair->second->emplace_back(TRITONSERVER_ErrorMessage(err));
      }
      TRITONSERVER_ErrorDelete(err);
    }
    // Just delete the response, warmup doesn't check for correctness
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting warmup response");
  }
  // Last response
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) {
    res_pair->first.set_value();
  }
}

void
WarmupRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    // Don't need to release request here, it is managed in WarmupData
    if (userp != nullptr) {
      auto warmup_promise = reinterpret_cast<std::promise<void>*>(userp);
      warmup_promise->set_value();
    }
  }
}

// Helper function for creating an instance
Status
VerifyModelLoadGpuFraction(
    const std::string& name, TRITONSERVER_InstanceGroupKind kind,
    int32_t device_id,
    const triton::common::BackendCmdlineConfigMap& backend_config_map)
{
  size_t free, total;
  double memory_limit;
  RETURN_IF_ERROR(GetDeviceMemoryInfo(device_id, &free, &total));
  RETURN_IF_ERROR(BackendConfigurationModelLoadGpuFraction(
      backend_config_map, device_id, &memory_limit));
  const size_t allow = total * memory_limit;
  const size_t used = total - free;
  if (used > allow) {
    return Status(
        Status::Code::UNAVAILABLE,
        std::string("can not create model '") + name +
            "': memory limit set for " +
            TRITONSERVER_InstanceGroupKindString(kind) + " " +
            std::to_string(device_id) +
            " has exceeded, model loading is rejected.");
  }
  return Status::Success;
}

}  // namespace

TritonModelInstance::TritonModelInstance(
    TritonModel* model, const std::string& name, const Signature& signature,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const triton::common::HostPolicyCmdlineConfig& host_policy,
    const TritonServerMessage& host_policy_message,
    const std::vector<SecondaryDevice>& secondary_devices)
    : model_(model), name_(name), signature_(signature), kind_(kind),
      device_id_(device_id), host_policy_(host_policy),
      host_policy_message_(host_policy_message), profile_names_(profile_names),
      passive_(passive), secondary_devices_(secondary_devices), state_(nullptr)
{
#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    // Use an ID in the metric only for GPU instances. Otherwise use
    // METRIC_REPORTER_ID_CPU to indicate no device should be reported in the
    // metric.
    const int id = (kind_ == TRITONSERVER_INSTANCEGROUPKIND_GPU)
                       ? device_id_
                       : METRIC_REPORTER_ID_CPU;
    // Let every metric reporter know if caching is enabled to correctly include
    // cache miss time into request duration on cache misses.
    const bool response_cache_enabled =
        model_->ResponseCacheEnabled() &&
        model_->Server()->ResponseCacheEnabled();
    MetricModelReporter::Create(
        model_->Name(), model_->Version(), id, response_cache_enabled,
        model_->Config().metric_tags(), &reporter_);
  }
#endif  // TRITON_ENABLE_METRICS
}

TritonModelInstance::~TritonModelInstance()
{
  if (triton_backend_thread_.get() != nullptr) {
    triton_backend_thread_->StopBackendThread();
  }

  model_->Server()->GetRateLimiter()->UnregisterModelInstance(this);

  // Model finalization is optional...
  if (model_->Backend()->ModelInstanceFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        model_->Backend()->ModelInstanceFiniFn()(
            reinterpret_cast<TRITONBACKEND_ModelInstance*>(this)),
        "failed finalizing model instance");
  }
}

Status
TritonModelInstance::CreateInstance(
    TritonModel* model, const std::string& name, const Signature& signature,
    TRITONSERVER_InstanceGroupKind kind, int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const std::string& host_policy_name,
    const inference::ModelRateLimiter& rate_limiter_config,
    const std::vector<SecondaryDevice>& secondary_devices,
    std::shared_ptr<TritonModelInstance>* triton_model_instance)
{
  static triton::common::HostPolicyCmdlineConfig empty_host_policy;
  const triton::common::HostPolicyCmdlineConfig* host_policy =
      &empty_host_policy;
  const auto policy_it = model->HostPolicyMap().find(host_policy_name);
  if (policy_it != model->HostPolicyMap().end()) {
    host_policy = &policy_it->second;
  }

  RETURN_IF_ERROR(SetNumaConfigOnThread(*host_policy));
  auto err = ConstructAndInitializeInstance(
      model, name, signature, kind, device_id, profile_names, passive,
      host_policy_name, *host_policy, rate_limiter_config, secondary_devices,
      triton_model_instance);
  RETURN_IF_ERROR(ResetNumaMemoryPolicy());
  RETURN_IF_ERROR(err);

  // When deploying on GPU, we want to make sure the GPU memory usage
  // is within allowed range, otherwise, stop the creation to ensure
  // there is sufficient GPU memory for other use.
  // We check the usage after loading the instance to better enforcing
  // the limit. If we check before loading, we may create instance
  // that occupies the rest of available memory which against the purpose
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    RETURN_IF_ERROR(VerifyModelLoadGpuFraction(
        name, kind, device_id, model->BackendConfigMap()));
  }

  return Status::Success;
}

Status
TritonModelInstance::ConstructAndInitializeInstance(
    TritonModel* model, const std::string& name, const Signature& signature,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const std::string& host_policy_name,
    const triton::common::HostPolicyCmdlineConfig& host_policy,
    const inference::ModelRateLimiter& rate_limiter_config,
    const std::vector<SecondaryDevice>& secondary_devices,
    std::shared_ptr<TritonModelInstance>* triton_model_instance)
{
  // Create the JSON representation of the backend configuration.
  triton::common::TritonJson::Value host_policy_json(
      triton::common::TritonJson::ValueType::OBJECT);
  triton::common::TritonJson::Value policy_setting_json(
      host_policy_json, triton::common::TritonJson::ValueType::OBJECT);
  for (const auto& pr : host_policy) {
    RETURN_IF_ERROR(policy_setting_json.AddString(pr.first.c_str(), pr.second));
  }

  RETURN_IF_ERROR(host_policy_json.Add(
      host_policy_name.c_str(), std::move(policy_setting_json)));
  TritonServerMessage host_policy_message(host_policy_json);

  std::unique_ptr<TritonModelInstance> local_instance(new TritonModelInstance(
      model, name, signature, kind, device_id, profile_names, passive,
      host_policy, host_policy_message, secondary_devices));

  TRITONBACKEND_ModelInstance* triton_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(local_instance.get());

  // Instance initialization is optional... We must set set shared
  // library path to point to the backend directory in case the
  // backend library attempts to load additional shared libraries.
  if (model->Backend()->ModelInstanceInitFn() != nullptr) {
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
    RETURN_IF_ERROR(slib->SetLibraryDirectory(model->Backend()->Directory()));
#endif

    TRITONSERVER_Error* err =
        model->Backend()->ModelInstanceInitFn()(triton_instance);

#ifdef _WIN32
    RETURN_IF_ERROR(slib->ResetLibraryDirectory());
#endif
    RETURN_IF_TRITONSERVER_ERROR(err);
  }

  if (!passive) {
    RETURN_IF_ERROR(local_instance->GenerateWarmupData());
    RETURN_IF_ERROR(model->Server()->GetRateLimiter()->RegisterModelInstance(
        local_instance.get(), rate_limiter_config));
    RETURN_IF_ERROR(local_instance->SetBackendThread(
        kind, device_id, model->DeviceBlocking()));
  }

  triton_model_instance->reset(local_instance.release());

  return Status::Success;
}

Status
TritonModelInstance::SetBackendThread(
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const bool device_blocking)
{
  if (ShareBackendThread(device_blocking, kind)) {
    auto device_instances = model_->GetInstancesByDevice(device_id);
    if (!device_instances.empty()) {
      LOG_VERBOSE(1) << "Using already started backend thread for " << Name()
                     << " on device " << device_id;
      triton_backend_thread_ = device_instances[0]->triton_backend_thread_;
    }
  }
  if (triton_backend_thread_.get() == nullptr) {
    std::unique_ptr<TritonBackendThread> local_backend_thread;
    RETURN_IF_ERROR(TritonBackendThread::CreateBackendThread(
        Name(), this, 0 /* nice */, device_id, &local_backend_thread));
    triton_backend_thread_ = std::move(local_backend_thread);
  } else {
    triton_backend_thread_->AddModelInstance(this);
  }
  RETURN_IF_ERROR(triton_backend_thread_->InitAndWarmUpModelInstance(this));

  return Status::Success;
}

Status
TritonModelInstance::GenerateWarmupData()
{
  warmup_samples_.clear();
  for (const auto& warmup_setting : model_->Config().model_warmup()) {
    if (warmup_setting.batch_size() == 0) {
      LOG_VERBOSE(1) << "Skipping batch 0 warmup sample '"
                     << warmup_setting.name() << "'";
      continue;
    }
    LOG_VERBOSE(1) << "Generating warmup sample data for '"
                   << warmup_setting.name() << "'";

    // Two passes. First pass to get max byte size for synthetic
    // data. Second pass to add original inputs and override inputs
    // for control inputs.
    int64_t max_zero_byte_size = 0;
    int64_t max_random_byte_size = 0;
    for (const auto& input_meta : warmup_setting.inputs()) {
      auto element_count =
          triton::common::GetElementCount(input_meta.second.dims());
      if (element_count == -1) {
        return Status(
            Status::Code::INVALID_ARG,
            "warmup setting expects all variable-size dimensions are specified "
            "for input '" +
                input_meta.first + "'");
      }

      int64_t batch_byte_size =
          element_count *
          triton::common::GetDataTypeByteSize(input_meta.second.data_type());
      if (batch_byte_size == 0) {
        batch_byte_size = element_count * sizeof(int32_t);
      }

      switch (input_meta.second.input_data_type_case()) {
        case inference::ModelWarmup_Input::InputDataTypeCase::kZeroData:
          max_zero_byte_size = std::max(batch_byte_size, max_zero_byte_size);
          break;
        case inference::ModelWarmup_Input::InputDataTypeCase::kRandomData: {
          // Because Triton expects STRING type to be in special format
          // (prepend 4 bytes to specify string length), so using zero data
          // for simplicity (4 bytes * element count of zeros).
          if (input_meta.second.data_type() ==
              inference::DataType::TYPE_STRING) {
            max_zero_byte_size = std::max(batch_byte_size, max_zero_byte_size);
          } else {
            max_random_byte_size =
                std::max(batch_byte_size, max_random_byte_size);
          }
          break;
        }
        default:
          break;
      }
    }

    warmup_samples_.emplace_back(warmup_setting.name(), warmup_setting.count());
    auto& warmup_data = warmup_samples_.back();
    // Create buffers for synthetic data
    TRITONSERVER_MemoryType type;
    int64_t type_id;
    warmup_data.zero_data_.reset(new AllocatedMemory(
        max_zero_byte_size, TRITONSERVER_MEMORY_CPU_PINNED /* memory_type */,
        0 /* memory_type_id */));
    char* zero_buffer = warmup_data.zero_data_->MutableBuffer(&type, &type_id);
    memset(zero_buffer, 0, max_zero_byte_size);

    warmup_data.random_data_.reset(new AllocatedMemory(
        max_random_byte_size, TRITONSERVER_MEMORY_CPU_PINNED /* memory_type */,
        0 /* memory_type_id */));
    char* random_buffer =
        warmup_data.random_data_->MutableBuffer(&type, &type_id);
    for (int64_t offset = 0; offset < max_random_byte_size; offset++) {
      random_buffer[offset] = rand();
    }

    // Prepare the inference request for the specified sample, not using
    // in-process C API because the request doesn't go through the same pipeline
    // (i.e. no normalization / scheduler) so we need to prepare the request to
    // the state just before calling instance execute function.
    for (size_t cnt = 0; cnt < warmup_setting.batch_size(); cnt++) {
      warmup_data.requests_.emplace_back(
          new InferenceRequest(model_, model_->Version()));
      auto& lrequest = warmup_data.requests_.back();

      // Second pass to prepare original inputs.
      std::vector<std::shared_ptr<InferenceRequest::Input>> input_sps;
      for (const auto& input_meta : warmup_setting.inputs()) {
        auto batch1_element_count =
            triton::common::GetElementCount(input_meta.second.dims());
        auto batch_byte_size =
            batch1_element_count *
            triton::common::GetDataTypeByteSize(input_meta.second.data_type());
        if (batch_byte_size == 0) {
          batch_byte_size = batch1_element_count * sizeof(int32_t);
        }

        const char* allocated_ptr;
        switch (input_meta.second.input_data_type_case()) {
          case inference::ModelWarmup_Input::InputDataTypeCase::kZeroData:
            allocated_ptr = zero_buffer;
            break;
          case inference::ModelWarmup_Input::InputDataTypeCase::kRandomData: {
            if (input_meta.second.data_type() ==
                inference::DataType::TYPE_STRING) {
              allocated_ptr = zero_buffer;
            } else {
              allocated_ptr = random_buffer;
            }
            break;
          }
          case inference::ModelWarmup_Input::InputDataTypeCase::
              kInputDataFile: {
            // For data provided from file, we can set buffer in first pass
            warmup_data.provided_data_.emplace_back(new std::string());
            auto input_data = warmup_data.provided_data_.back().get();
            RETURN_IF_ERROR(ReadTextFile(
                JoinPath(
                    {model_->LocalizedModelPath(), kWarmupDataFolder,
                     input_meta.second.input_data_file()}),
                input_data));
            if (input_meta.second.data_type() ==
                inference::DataType::TYPE_STRING) {
              batch_byte_size = input_data->size();
            } else if (((size_t)batch_byte_size) > input_data->size()) {
              return Status(
                  Status::Code::INVALID_ARG,
                  lrequest->LogRequest() + "warmup setting expects " +
                      std::to_string(batch_byte_size) +
                      " bytes, but the data "
                      "provided from " +
                      input_meta.second.input_data_file() + "only has " +
                      std::to_string(input_data->size()) + " bytes");
            }
            allocated_ptr = input_data->data();
            break;
          }
          default:
            return Status(
                Status::Code::INVALID_ARG,
                lrequest->LogRequest() + "warmup setting expects input '" +
                    input_meta.first + "' to have input_data_type set");
        }

        const inference::ModelInput* input_config;
        bool is_original_input =
            model_->GetInput(input_meta.first, &input_config).IsOk();
        InferenceRequest::Input* input = nullptr;
        std::vector<int64_t> input_meta_shape;
        // Append batch size only if the model supports batching
        // and not control inpt.
        if ((model_->Config().max_batch_size() != 0) && is_original_input) {
          input_meta_shape.push_back(1);
        }
        for (auto d : input_meta.second.dims()) {
          input_meta_shape.push_back(d);
        }
        if (is_original_input) {
          RETURN_IF_ERROR(lrequest->AddOriginalInput(
              input_meta.first, input_meta.second.data_type(), input_meta_shape,
              &input));
        } else {
          input_sps.emplace_back();
          RETURN_IF_ERROR(lrequest->AddOverrideInput(
              input_meta.first, input_meta.second.data_type(),
              (model_->Config().max_batch_size() != 0 ? 1 : 0),
              input_meta_shape, &input_sps.back()));
          input = input_sps.back().get();
        }
        RETURN_IF_ERROR(input->AppendData(
            allocated_ptr, batch_byte_size,
            TRITONSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */));
      }

      RETURN_IF_ERROR(lrequest->PrepareForInference());
      // Override inputs must be added after PrepareForInference() is called
      for (const auto& sp : input_sps) {
        RETURN_IF_ERROR(lrequest->AddOverrideInput(sp));
      }
    }
  }

  return Status::Success;
}

Status
TritonModelInstance::PrepareRequestsForExecution(
    std::vector<std::unique_ptr<InferenceRequest>>& requests)
{
  for (auto& r : requests) {
    // Load the input states for the inference request.
    RETURN_IF_ERROR(r->LoadInputStates());
    // Set request state to signify that request is no longer pending.
    RETURN_IF_ERROR(r->SetState(InferenceRequest::State::EXECUTING));
  }

  return Status::Success;
}

Status
TritonModelInstance::PrepareRequestsOrRespond(
    std::vector<std::unique_ptr<InferenceRequest>>& requests)
{
  auto status = PrepareRequestsForExecution(requests);
  // If any errors occurred, respond with error for each request.
  if (!status.IsOk()) {
    for (auto& r : requests) {
      InferenceRequest::RespondIfError(r, status, true /* release_requests */);
    }
    // Log a single error for batch of requests for better visibility
    LOG_STATUS_ERROR(status, "Requests failed pre-execution checks");
  }

  return status;
}

Status
TritonModelInstance::Schedule(
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  // Prepare requests for execution, respond to requests if any error occur.
  RETURN_IF_ERROR(PrepareRequestsOrRespond(requests));

  // Use a thread local vector to avoid needing to malloc each
  // time an inference is run.
  thread_local std::vector<TRITONBACKEND_Request*> triton_requests(1024);
  triton_requests.clear();
  for (auto& r : requests) {
    triton_requests.push_back(
        reinterpret_cast<TRITONBACKEND_Request*>(r.release()));
  }

  Execute(triton_requests);
  return Status::Success;
}

Status
TritonModelInstance::Initialize()
{
  RETURN_IF_ERROR(SetNumaConfigOnThread(HostPolicy()));
  return Status::Success;
}

Status
TritonModelInstance::WarmUp()
{
  // move samples to local variable for scoped cleanup
  std::vector<triton::core::TritonModelInstance::WarmupData> lwarmup_samples;
  lwarmup_samples.swap(warmup_samples_);

  for (auto& sample : lwarmup_samples) {
    for (size_t iteration = 1; iteration <= sample.count_; ++iteration) {
      LOG_VERBOSE(1) << "model '" << sample.requests_.back()->ModelName()
                     << "' instance " << Name() << " is running warmup sample '"
                     << sample.sample_name_ << "' for iteration " << iteration;

      // request/response complete is asynchronous so use promise to wait for
      // completion. Also collects error message from the responses in a vector.
      std::vector<std::promise<void>> request_complete(sample.requests_.size());
      std::vector<std::string> response_errors;
      std::vector<std::pair<std::promise<void>, std::vector<std::string>*>>
          response_complete(sample.requests_.size());

      std::vector<TRITONBACKEND_Request*> triton_requests;
      for (size_t i = 0; i < sample.requests_.size(); ++i) {
        auto& request = sample.requests_[i];
        request->SetReleaseCallback(
            WarmupRequestComplete, &request_complete[i]);
        response_complete[i].second = &response_errors;
        request->SetResponseCallback(
            &warmup_allocator, nullptr, WarmupResponseComplete,
            &response_complete[i]);

        // For warmup requests we need to manually set ResponseFactory
        // since they modify the callback after PrepareForInference has
        // been called.
        request->SetResponseFactory();

        // Capture timestamp before run to avoid incorrect accumulation from
        // sequential warmup runs
#ifdef TRITON_ENABLE_STATS
        request->CaptureRequestStartNs();
#endif  // TRITON_ENABLE_STATS
        request->CaptureQueueStartNs();
        triton_requests.push_back(
            reinterpret_cast<TRITONBACKEND_Request*>(request.get()));
      }

      Execute(triton_requests);

      // Wait for warmup sample to complete and check error
      for (size_t i = 0; i < sample.requests_.size(); ++i) {
        request_complete[i].get_future().get();
        response_complete[i].first.get_future().get();
      }
      if (response_errors.size() != 0) {
        std::string err_str =
            "failed to run warmup sample '" + sample.sample_name_ + "': ";
        for (const auto& error : response_errors) {
          err_str += (error + "; ");
        }
        // End warmup as soon as there is failing sample
        LOG_VERBOSE(1) << "model '" << sample.requests_.back()->ModelName()
                       << "' instance " << Name()
                       << " failed to run warmup sample '"
                       << sample.sample_name_ << "'";
        return Status(Status::Code::INVALID_ARG, err_str);
      }
    }
  }

  return Status::Success;
}

void
TritonModelInstance::Execute(
    std::vector<TRITONBACKEND_Request*>& triton_requests)
{
  TRITONBACKEND_ModelInstance* triton_model_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(this);
  TritonBackend::TritonModelInstanceExecFn_t inst_exec_fn =
      model_->Backend()->ModelInstanceExecFn();

  // If there is an error then we retain ownership of 'requests'
  // and must send error responses.
  TRITONSERVER_Error* err = inst_exec_fn(
      triton_model_instance, &triton_requests[0], triton_requests.size());
  if (err != nullptr) {
    Status status = Status(
        TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
    for (TRITONBACKEND_Request* tr : triton_requests) {
      std::unique_ptr<InferenceRequest> ur(
          reinterpret_cast<InferenceRequest*>(tr));
      InferenceRequest::RespondIfError(ur, status, true /* release_requests */);
    }

    TRITONSERVER_ErrorDelete(err);
  }
}

Status
TritonModelInstance::TritonBackendThread::CreateBackendThread(
    const std::string name, TritonModelInstance* model_instance, const int nice,
    const int32_t device_id,
    std::unique_ptr<TritonBackendThread>* triton_backend_thread)
{
  TritonBackendThread* raw_triton_backend_thread =
      new TritonBackendThread(name, model_instance->Model(), nice, device_id);
  std::unique_ptr<TritonBackendThread> runner(raw_triton_backend_thread);

  runner->AddModelInstance(model_instance);
  runner->backend_thread_ = std::thread([raw_triton_backend_thread]() {
    raw_triton_backend_thread->BackendThread();
  });

  triton_backend_thread->reset(runner.release());

  return Status::Success;
}

void
TritonModelInstance::TritonBackendThread::AddModelInstance(
    TritonModelInstance* model_instance)
{
  model_instances_.push_back(model_instance);
}

Status
TritonModelInstance::TritonBackendThread::InitAndWarmUpModelInstance(
    TritonModelInstance* model_instance)
{
  // Initialize the instance on the backend thread
  auto init_payload = model_->Server()->GetRateLimiter()->GetPayload(
      Payload::Operation::INIT, model_instance);
  RETURN_IF_ERROR(
      model_->Server()->GetRateLimiter()->EnqueuePayload(model_, init_payload));
  RETURN_IF_ERROR(init_payload->Wait());

  // Warm-up the instance on the backend thread
  auto warmup_payload = model_->Server()->GetRateLimiter()->GetPayload(
      Payload::Operation::WARM_UP, model_instance);
  RETURN_IF_ERROR(model_->Server()->GetRateLimiter()->EnqueuePayload(
      model_, warmup_payload));
  RETURN_IF_ERROR(warmup_payload->Wait());

  return Status::Success;
}

TritonModelInstance::TritonBackendThread::TritonBackendThread(
    const std::string& name, TritonModel* model, const int nice,
    const int32_t device_id)
    : name_(name), nice_(nice), device_id_(device_id), model_(model)
{
}

TritonModelInstance::TritonBackendThread::~TritonBackendThread()
{
  StopBackendThread();
}

void
TritonModelInstance::TritonBackendThread::StopBackendThread()
{
  if (backend_thread_.joinable()) {
    // Signal the backend thread to exit and then wait for it...
    auto exit_payload = model_->Server()->GetRateLimiter()->GetPayload(
        Payload::Operation::EXIT, model_instances_.back());
    model_->Server()->GetRateLimiter()->EnqueuePayload(model_, exit_payload);
    backend_thread_.join();
  }
}

void
TritonModelInstance::TritonBackendThread::BackendThread()
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice_) == 0) {
    LOG_VERBOSE(1) << "Starting backend thread for " << name_ << " at nice "
                   << nice_ << " on device " << device_id_ << "...";
  } else {
    LOG_VERBOSE(1) << "Starting backend thread for " << name_
                   << " at default nice (requested nice " << nice_ << " failed)"
                   << " on device " << device_id_ << "...";
  }
#else
  LOG_VERBOSE(1) << "Starting backend thread for " << name_
                 << " at default nice on device " << device_id_ << "...";
#endif

  bool should_exit = false;
  while (!should_exit) {
    std::shared_ptr<Payload> payload;
    model_->Server()->GetRateLimiter()->DequeuePayload(
        model_instances_, &payload);
    NVTX_RANGE(nvtx_, "BackendThread " + name_);
    payload->Execute(&should_exit);
    model_instances_.push_back(payload->GetInstance());
    // Release the payload to the RateLimiter
    model_->Server()->GetRateLimiter()->PayloadRelease(payload);
  }
  LOG_VERBOSE(1) << "Stopping backend thread for " << name_ << "...";
}

extern "C" {

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceName(
    TRITONBACKEND_ModelInstance* instance, const char** name)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *name = ti->Name().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceKind(
    TRITONBACKEND_ModelInstance* instance, TRITONSERVER_InstanceGroupKind* kind)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *kind = ti->Kind();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceDeviceId(
    TRITONBACKEND_ModelInstance* instance, int32_t* device_id)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *device_id = ti->DeviceId();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceHostPolicy(
    TRITONBACKEND_ModelInstance* instance, TRITONSERVER_Message** host_policy)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *host_policy = const_cast<TRITONSERVER_Message*>(
      reinterpret_cast<const TRITONSERVER_Message*>(&ti->HostPolicyMessage()));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileCount(
    TRITONBACKEND_ModelInstance* instance, uint32_t* count)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *count = ti->Profiles().size();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceProfileName(
    TRITONBACKEND_ModelInstance* instance, const uint32_t index,
    const char** profile_name)
{
  *profile_name = nullptr;

  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  const auto& rprofiles = ti->Profiles();
  if (index >= rprofiles.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("out of bounds index ") + std::to_string(index) +
         ": instance is configured with " + std::to_string(rprofiles.size()) +
         " profiles")
            .c_str());
  }

  *profile_name = rprofiles[index].c_str();

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSecondaryDeviceCount(
    TRITONBACKEND_ModelInstance* instance, uint32_t* count)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *count = ti->SecondaryDevices().size();

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSecondaryDeviceProperties(
    TRITONBACKEND_ModelInstance* instance, uint32_t index, const char** kind,
    int64_t* id)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  const auto& rsecondarydevices = ti->SecondaryDevices();

  if (index >= rsecondarydevices.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("out of bounds index ") + std::to_string(index) +
         ": instance is configured with " +
         std::to_string(rsecondarydevices.size()) + " secondary devices")
            .c_str());
  }

  *kind = rsecondarydevices[index].kind_.c_str();
  *id = rsecondarydevices[index].id_;

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceIsPassive(
    TRITONBACKEND_ModelInstance* instance, bool* is_passive)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *is_passive = ti->IsPassive();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceModel(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Model** model)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *model = reinterpret_cast<TRITONBACKEND_Model*>(ti->Model());
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceState(
    TRITONBACKEND_ModelInstance* instance, void** state)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  *state = ti->State();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceSetState(
    TRITONBACKEND_ModelInstance* instance, void* state)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->SetState(state);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportStatistics(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request* request,
    const bool success, const uint64_t exec_start_ns,
    const uint64_t compute_start_ns, const uint64_t compute_end_ns,
    const uint64_t exec_end_ns)
{
#ifdef TRITON_ENABLE_STATS
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  tr->ReportStatistics(
      ti->MetricReporter(), success, exec_start_ns, compute_start_ns,
      compute_end_ns, exec_end_ns);
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportBatchStatistics(
    TRITONBACKEND_ModelInstance* instance, const uint64_t batch_size,
    const uint64_t exec_start_ns, const uint64_t compute_start_ns,
    const uint64_t compute_end_ns, const uint64_t exec_end_ns)
{
#ifdef TRITON_ENABLE_STATS
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->Model()->MutableStatsAggregator()->UpdateInferBatchStats(
      ti->MetricReporter(), batch_size, exec_start_ns, compute_start_ns,
      compute_end_ns, exec_end_ns);
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceReportMemoryUsage(
    TRITONBACKEND_ModelInstance* instance,
    TRITONSERVER_BufferAttributes** usage, uint32_t usage_size)
{
  TritonModelInstance* ti = reinterpret_cast<TritonModelInstance*>(instance);
  ti->SetMemoryUsage({reinterpret_cast<BufferAttributes**>(usage), usage_size});
  return nullptr;  // success
}

}  // extern C
}}  // namespace triton::core
