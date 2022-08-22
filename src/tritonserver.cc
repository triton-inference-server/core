// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>
#include "buffer_attributes.h"
#include "cuda_utils.h"
#include "infer_parameter.h"
#include "infer_request.h"
#include "infer_response.h"
#include "infer_stats.h"
#include "metric_family.h"
#include "metrics.h"
#include "model.h"
#include "model_config_utils.h"
#include "model_repository_manager.h"
#include "rate_limiter.h"
#include "response_allocator.h"
#include "server.h"
#include "server_message.h"
#include "status.h"
#include "triton/common/logging.h"
#include "triton/common/model_config.h"
#include "triton/common/nvtx.h"
#include "triton/common/table_printer.h"
#include "triton/common/triton_json.h"
#include "tritonserver_apis.h"

// For unknown reason, windows will not export some functions declared
// with dllexport in tritonrepoagent.h and tritonbackend.h. To get
// those functions exported it is (also?) necessary to mark the
// definitions in this file with dllexport as well. The TRITONSERVER_*
// functions are getting exported but for consistency adding the
// declspec to these definitions as well.
#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

namespace tc = triton::core;

namespace {

std::string
ResourceString(const std::string& name, const int count, const int device_id)
{
  return std::string(
      "{\"name\":\"" + name + "\", \"count\":" + std::to_string(count) +
      " \"device\":" + std::to_string(device_id) + "}");
}

std::string
RateLimitModeToString(const tc::RateLimitMode rate_limit_mode)
{
  std::string rl_mode_str("<unknown>");
  switch (rate_limit_mode) {
    case tc::RateLimitMode::RL_EXEC_COUNT: {
      rl_mode_str = "EXEC_COUNT";
      break;
    }
    case tc::RateLimitMode::RL_OFF: {
      rl_mode_str = "OFF";
      break;
    }
  }
  return rl_mode_str;
}

//
// TritonServerError
//
// Implementation for TRITONSERVER_Error.
//
class TritonServerError {
 public:
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const char* msg);
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const std::string& msg);
  static TRITONSERVER_Error* Create(const tc::Status& status);

  TRITONSERVER_Error_Code Code() const { return code_; }
  const std::string& Message() const { return msg_; }

 private:
  TritonServerError(TRITONSERVER_Error_Code code, const std::string& msg)
      : code_(code), msg_(msg)
  {
  }
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }

  TRITONSERVER_Error_Code code_;
  const std::string msg_;
};

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const std::string& msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(const tc::Status& status)
{
  // If 'status' is success then return nullptr as that indicates
  // success
  if (status.IsOk()) {
    return nullptr;
  }

  return Create(
      tc::StatusCodeToTritonCode(status.StatusCode()), status.Message());
}

#define RETURN_IF_STATUS_ERROR(S)                 \
  do {                                            \
    const tc::Status& status__ = (S);             \
    if (!status__.IsOk()) {                       \
      return TritonServerError::Create(status__); \
    }                                             \
  } while (false)

//
// TritonServerMetrics
//
// Implementation for TRITONSERVER_Metrics.
//
class TritonServerMetrics {
 public:
  TritonServerMetrics() = default;
  TRITONSERVER_Error* Serialize(const char** base, size_t* byte_size);

 private:
  std::string serialized_;
};

TRITONSERVER_Error*
TritonServerMetrics::Serialize(const char** base, size_t* byte_size)
{
#ifdef TRITON_ENABLE_METRICS
  serialized_ = tc::Metrics::SerializedMetrics();
  *base = serialized_.c_str();
  *byte_size = serialized_.size();
  return nullptr;  // Success
#else
  *base = nullptr;
  *byte_size = 0;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

//
// TritonServerOptions
//
// Implementation for TRITONSERVER_ServerOptions.
//
class TritonServerOptions {
 public:
  TritonServerOptions();

  const std::string& ServerId() const { return server_id_; }
  void SetServerId(const char* id) { server_id_ = id; }

  const std::set<std::string>& ModelRepositoryPaths() const
  {
    return repo_paths_;
  }
  void SetModelRepositoryPath(const char* p) { repo_paths_.insert(p); }

  tc::ModelControlMode ModelControlMode() const { return model_control_mode_; }
  void SetModelControlMode(tc::ModelControlMode m) { model_control_mode_ = m; }

  const std::set<std::string>& StartupModels() const { return models_; }
  void SetStartupModel(const char* m) { models_.insert(m); }

  bool ExitOnError() const { return exit_on_error_; }
  void SetExitOnError(bool b) { exit_on_error_ = b; }

  bool StrictModelConfig() const { return strict_model_config_; }
  void SetStrictModelConfig(bool b) { strict_model_config_ = b; }

  tc::RateLimitMode RateLimiterMode() const { return rate_limit_mode_; }
  void SetRateLimiterMode(tc::RateLimitMode m) { rate_limit_mode_ = m; }

  TRITONSERVER_Error* AddRateLimiterResource(
      const std::string& resource, const size_t count, const int device);

  // The resource map is the map from device id to the map of
  // of resources with their respective counts for that device.
  const tc::RateLimiter::ResourceMap& RateLimiterResources() const
  {
    return rate_limit_resource_map_;
  }

  uint64_t PinnedMemoryPoolByteSize() const { return pinned_memory_pool_size_; }
  void SetPinnedMemoryPoolByteSize(uint64_t s) { pinned_memory_pool_size_ = s; }

  uint64_t ResponseCacheByteSize() const { return response_cache_byte_size_; }
  void SetResponseCacheByteSize(uint64_t s) { response_cache_byte_size_ = s; }

  const std::map<int, uint64_t>& CudaMemoryPoolByteSize() const
  {
    return cuda_memory_pool_size_;
  }
  void SetCudaMemoryPoolByteSize(int id, uint64_t s)
  {
    cuda_memory_pool_size_[id] = s;
  }

  double MinSupportedComputeCapability() const
  {
    return min_compute_capability_;
  }
  void SetMinSupportedComputeCapability(double c)
  {
    min_compute_capability_ = c;
  }

  bool StrictReadiness() const { return strict_readiness_; }
  void SetStrictReadiness(bool b) { strict_readiness_ = b; }

  unsigned int ExitTimeout() const { return exit_timeout_; }
  void SetExitTimeout(unsigned int t) { exit_timeout_ = t; }

  unsigned int BufferManagerThreadCount() const
  {
    return buffer_manager_thread_count_;
  }
  void SetBufferManagerThreadCount(unsigned int c)
  {
    buffer_manager_thread_count_ = c;
  }

  unsigned int ModelLoadThreadCount() const { return model_load_thread_count_; }
  void SetModelLoadThreadCount(unsigned int c) { model_load_thread_count_ = c; }

  bool Metrics() const { return metrics_; }
  void SetMetrics(bool b) { metrics_ = b; }

  bool GpuMetrics() const { return gpu_metrics_; }
  void SetGpuMetrics(bool b) { gpu_metrics_ = b; }

  bool CpuMetrics() const { return cpu_metrics_; }
  void SetCpuMetrics(bool b) { cpu_metrics_ = b; }

  uint64_t MetricsInterval() const { return metrics_interval_; }
  void SetMetricsInterval(uint64_t m) { metrics_interval_ = m; }

  const std::string& BackendDir() const { return backend_dir_; }
  void SetBackendDir(const std::string& bd) { backend_dir_ = bd; }

  const std::string& RepoAgentDir() const { return repoagent_dir_; }
  void SetRepoAgentDir(const std::string& rad) { repoagent_dir_ = rad; }

  // The backend config map is a map from backend name to the
  // setting=value pairs for that backend. The empty backend name ("")
  // is used to communicate configuration information that is used
  // internally.
  const triton::common::BackendCmdlineConfigMap& BackendCmdlineConfigMap() const
  {
    return backend_cmdline_config_map_;
  }
  TRITONSERVER_Error* AddBackendConfig(
      const std::string& backend_name, const std::string& setting,
      const std::string& value);

  TRITONSERVER_Error* SetHostPolicy(
      const std::string& policy_name, const std::string& setting,
      const std::string& value);
  const triton::common::HostPolicyCmdlineConfigMap& HostPolicyCmdlineConfigMap()
      const
  {
    return host_policy_map_;
  }

 private:
  std::string server_id_;
  std::set<std::string> repo_paths_;
  tc::ModelControlMode model_control_mode_;
  std::set<std::string> models_;
  bool exit_on_error_;
  bool strict_model_config_;
  bool strict_readiness_;
  tc::RateLimitMode rate_limit_mode_;
  tc::RateLimiter::ResourceMap rate_limit_resource_map_;
  bool metrics_;
  bool gpu_metrics_;
  bool cpu_metrics_;
  uint64_t metrics_interval_;
  unsigned int exit_timeout_;
  uint64_t pinned_memory_pool_size_;
  uint64_t response_cache_byte_size_;
  unsigned int buffer_manager_thread_count_;
  unsigned int model_load_thread_count_;
  std::map<int, uint64_t> cuda_memory_pool_size_;
  double min_compute_capability_;
  std::string backend_dir_;
  std::string repoagent_dir_;
  triton::common::BackendCmdlineConfigMap backend_cmdline_config_map_;
  triton::common::HostPolicyCmdlineConfigMap host_policy_map_;
};

TritonServerOptions::TritonServerOptions()
    : server_id_("triton"),
      model_control_mode_(tc::ModelControlMode::MODE_POLL),
      exit_on_error_(true), strict_model_config_(true), strict_readiness_(true),
      rate_limit_mode_(tc::RateLimitMode::RL_OFF), metrics_(true),
      gpu_metrics_(true), cpu_metrics_(true), metrics_interval_(2000),
      exit_timeout_(30), pinned_memory_pool_size_(1 << 28),
      response_cache_byte_size_(0), buffer_manager_thread_count_(0),
      model_load_thread_count_(
          std::max(2u, 2 * std::thread::hardware_concurrency())),
#ifdef TRITON_ENABLE_GPU
      min_compute_capability_(TRITON_MIN_COMPUTE_CAPABILITY),
#else
      min_compute_capability_(0),
#endif  // TRITON_ENABLE_GPU
      backend_dir_("/opt/tritonserver/backends"),
      repoagent_dir_("/opt/tritonserver/repoagents")
{
#ifndef TRITON_ENABLE_METRICS
  metrics_ = false;
  gpu_metrics_ = false;
  cpu_metrics_ = false;
#endif  // TRITON_ENABLE_METRICS

#ifndef TRITON_ENABLE_METRICS_GPU
  gpu_metrics_ = false;
#endif  // TRITON_ENABLE_METRICS_GPU

#ifndef TRITON_ENABLE_METRICS_CPU
  cpu_metrics_ = false;
#endif  // TRITON_ENABLE_METRICS_CPU
}

TRITONSERVER_Error*
TritonServerOptions::AddRateLimiterResource(
    const std::string& name, const size_t count, const int device)
{
  auto ditr = rate_limit_resource_map_.find(device);
  if (ditr == rate_limit_resource_map_.end()) {
    ditr = rate_limit_resource_map_
               .emplace(device, std::map<std::string, size_t>())
               .first;
  }
  auto ritr = ditr->second.find(name);
  if (ritr == ditr->second.end()) {
    ditr->second.emplace(name, count).first;
  } else {
    // If already present then store the minimum of the two.
    if (ritr->second > count) {
      ritr->second = count;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TritonServerOptions::AddBackendConfig(
    const std::string& backend_name, const std::string& setting,
    const std::string& value)
{
  triton::common::BackendCmdlineConfig& cc =
      backend_cmdline_config_map_[backend_name];
  cc.push_back(std::make_pair(setting, value));

  return nullptr;  // success
}

TRITONSERVER_Error*
TritonServerOptions::SetHostPolicy(
    const std::string& policy_name, const std::string& setting,
    const std::string& value)
{
  // Check if supported setting is passed
  if ((setting != "numa-node") && (setting != "cpu-cores")) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        std::string(
            "Unsupported host policy setting '" + setting +
            "' is specified, supported settings are 'numa-node', 'cpu-cores'")
            .c_str());
  }

  triton::common::HostPolicyCmdlineConfig& hp = host_policy_map_[policy_name];
  hp[setting] = value;

  return nullptr;  // success
}

#define SetDurationStat(DOC, PARENT, STAT_NAME, COUNT, NS)   \
  do {                                                       \
    triton::common::TritonJson::Value dstat(                 \
        DOC, triton::common::TritonJson::ValueType::OBJECT); \
    dstat.AddUInt("count", (COUNT));                         \
    dstat.AddUInt("ns", (NS));                               \
    PARENT.Add(STAT_NAME, std::move(dstat));                 \
  } while (false)

}  // namespace

extern "C" {

//
// TRITONSERVER API Version
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ApiVersion(uint32_t* major, uint32_t* minor)
{
  *major = TRITONSERVER_API_VERSION_MAJOR;
  *minor = TRITONSERVER_API_VERSION_MINOR;
  return nullptr;  // success
}

//
// TRITONSERVER_DataType
//
TRITONAPI_DECLSPEC const char*
TRITONSERVER_DataTypeString(TRITONSERVER_DataType datatype)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL:
      return "BOOL";
    case TRITONSERVER_TYPE_UINT8:
      return "UINT8";
    case TRITONSERVER_TYPE_UINT16:
      return "UINT16";
    case TRITONSERVER_TYPE_UINT32:
      return "UINT32";
    case TRITONSERVER_TYPE_UINT64:
      return "UINT64";
    case TRITONSERVER_TYPE_INT8:
      return "INT8";
    case TRITONSERVER_TYPE_INT16:
      return "INT16";
    case TRITONSERVER_TYPE_INT32:
      return "INT32";
    case TRITONSERVER_TYPE_INT64:
      return "INT64";
    case TRITONSERVER_TYPE_FP16:
      return "FP16";
    case TRITONSERVER_TYPE_FP32:
      return "FP32";
    case TRITONSERVER_TYPE_FP64:
      return "FP64";
    case TRITONSERVER_TYPE_BYTES:
      return "BYTES";
    case TRITONSERVER_TYPE_BF16:
      return "BF16";
    default:
      break;
  }

  return "<invalid>";
}

TRITONAPI_DECLSPEC TRITONSERVER_DataType
TRITONSERVER_StringToDataType(const char* dtype)
{
  const size_t len = strlen(dtype);
  return tc::DataTypeToTriton(
      triton::common::ProtocolStringToDataType(dtype, len));
}

TRITONAPI_DECLSPEC uint32_t
TRITONSERVER_DataTypeByteSize(TRITONSERVER_DataType datatype)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL:
    case TRITONSERVER_TYPE_INT8:
    case TRITONSERVER_TYPE_UINT8:
      return 1;
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_FP16:
    case TRITONSERVER_TYPE_BF16:
      return 2;
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_FP32:
      return 4;
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_FP64:
      return 8;
    case TRITONSERVER_TYPE_BYTES:
      return 0;
    default:
      break;
  }

  return 0;
}

//
// TRITONSERVER_MemoryType
//
TRITONAPI_DECLSPEC const char*
TRITONSERVER_MemoryTypeString(TRITONSERVER_MemoryType memtype)
{
  switch (memtype) {
    case TRITONSERVER_MEMORY_CPU:
      return "CPU";
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return "CPU_PINNED";
    case TRITONSERVER_MEMORY_GPU:
      return "GPU";
    default:
      break;
  }

  return "<invalid>";
}

//
// TRITONSERVER_Parameter
//
TRITONAPI_DECLSPEC const char*
TRITONSERVER_ParameterTypeString(TRITONSERVER_ParameterType paramtype)
{
  switch (paramtype) {
    case TRITONSERVER_PARAMETER_STRING:
      return "STRING";
    case TRITONSERVER_PARAMETER_INT:
      return "INT";
    case TRITONSERVER_PARAMETER_BOOL:
      return "BOOL";
    default:
      break;
  }

  return "<invalid>";
}

TRITONAPI_DECLSPEC TRITONSERVER_Parameter*
TRITONSERVER_ParameterNew(
    const char* name, const TRITONSERVER_ParameterType type, const void* value)
{
  std::unique_ptr<tc::InferenceParameter> lparam;
  switch (type) {
    case TRITONSERVER_PARAMETER_STRING:
      lparam.reset(new tc::InferenceParameter(
          name, reinterpret_cast<const char*>(value)));
      break;
    case TRITONSERVER_PARAMETER_INT:
      lparam.reset(new tc::InferenceParameter(
          name, *reinterpret_cast<const int64_t*>(value)));
      break;
    case TRITONSERVER_PARAMETER_BOOL:
      lparam.reset(new tc::InferenceParameter(
          name, *reinterpret_cast<const bool*>(value)));
      break;
    default:
      break;
  }
  return reinterpret_cast<TRITONSERVER_Parameter*>(lparam.release());
}

TRITONAPI_DECLSPEC TRITONSERVER_Parameter*
TRITONSERVER_ParameterBytesNew(
    const char* name, const void* byte_ptr, const uint64_t size)
{
  std::unique_ptr<tc::InferenceParameter> lparam(
      new tc::InferenceParameter(name, byte_ptr, size));
  return reinterpret_cast<TRITONSERVER_Parameter*>(lparam.release());
}

TRITONAPI_DECLSPEC void
TRITONSERVER_ParameterDelete(TRITONSERVER_Parameter* parameter)
{
  delete reinterpret_cast<tc::InferenceParameter*>(parameter);
}

//
// TRITONSERVER_InstanceGroupKind
//
TRITONAPI_DECLSPEC const char*
TRITONSERVER_InstanceGroupKindString(TRITONSERVER_InstanceGroupKind kind)
{
  switch (kind) {
    case TRITONSERVER_INSTANCEGROUPKIND_AUTO:
      return "AUTO";
    case TRITONSERVER_INSTANCEGROUPKIND_CPU:
      return "CPU";
    case TRITONSERVER_INSTANCEGROUPKIND_GPU:
      return "GPU";
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL:
      return "MODEL";
    default:
      break;
  }

  return "<invalid>";
}

//
// TRITONSERVER_Log
//
TRITONAPI_DECLSPEC bool
TRITONSERVER_LogIsEnabled(TRITONSERVER_LogLevel level)
{
  switch (level) {
    case TRITONSERVER_LOG_INFO:
      return LOG_INFO_IS_ON;
    case TRITONSERVER_LOG_WARN:
      return LOG_WARNING_IS_ON;
    case TRITONSERVER_LOG_ERROR:
      return LOG_ERROR_IS_ON;
    case TRITONSERVER_LOG_VERBOSE:
      return LOG_VERBOSE_IS_ON(1);
  }

  return false;
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_LogMessage(
    TRITONSERVER_LogLevel level, const char* filename, const int line,
    const char* msg)
{
  switch (level) {
    case TRITONSERVER_LOG_INFO:
      LOG_INFO_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_WARN:
      LOG_WARNING_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_ERROR:
      LOG_ERROR_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_VERBOSE:
      LOG_VERBOSE_FL(1, filename, line) << msg;
      return nullptr;
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown logging level '" + std::to_string(level) + "'")
              .c_str());
  }
}

//
// TRITONSERVER_Error
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      TritonServerError::Create(code, msg));
}

TRITONAPI_DECLSPEC void
TRITONSERVER_ErrorDelete(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  delete lerror;
}

TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Code();
}

TRITONAPI_DECLSPEC const char*
TRITONSERVER_ErrorCodeString(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return tc::Status::CodeString(tc::TritonCodeToStatusCode(lerror->Code()));
}

TRITONAPI_DECLSPEC const char*
TRITONSERVER_ErrorMessage(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Message().c_str();
}

//
// TRITONSERVER_ResponseAllocator
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorNew(
    TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
    TRITONSERVER_ResponseAllocatorStartFn_t start_fn)
{
  *allocator = reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
      new tc::ResponseAllocator(alloc_fn, release_fn, start_fn));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorSetQueryFunction(
    TRITONSERVER_ResponseAllocator* allocator,
    TRITONSERVER_ResponseAllocatorQueryFn_t query_fn)
{
  reinterpret_cast<tc::ResponseAllocator*>(allocator)->SetQueryFunction(
      query_fn);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
    TRITONSERVER_ResponseAllocator* allocator,
    TRITONSERVER_ResponseAllocatorBufferAttributesFn_t buffer_attributes_fn)
{
  reinterpret_cast<tc::ResponseAllocator*>(allocator)
      ->SetBufferAttributesFunction(buffer_attributes_fn);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorDelete(TRITONSERVER_ResponseAllocator* allocator)
{
  tc::ResponseAllocator* lalloc =
      reinterpret_cast<tc::ResponseAllocator*>(allocator);
  delete lalloc;
  return nullptr;  // Success
}

//
// TRITONSERVER_Message
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_MessageNewFromSerializedJson(
    TRITONSERVER_Message** message, const char* base, size_t byte_size)
{
  *message = reinterpret_cast<TRITONSERVER_Message*>(
      new tc::TritonServerMessage({base, byte_size}));
  return nullptr;
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_MessageDelete(TRITONSERVER_Message* message)
{
  tc::TritonServerMessage* lmessage =
      reinterpret_cast<tc::TritonServerMessage*>(message);
  delete lmessage;
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_MessageSerializeToJson(
    TRITONSERVER_Message* message, const char** base, size_t* byte_size)
{
  tc::TritonServerMessage* lmessage =
      reinterpret_cast<tc::TritonServerMessage*>(message);
  lmessage->Serialize(base, byte_size);
  return nullptr;  // Success
}

//
// TRITONSERVER_Metrics
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_MetricsDelete(TRITONSERVER_Metrics* metrics)
{
  TritonServerMetrics* lmetrics =
      reinterpret_cast<TritonServerMetrics*>(metrics);
  delete lmetrics;
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_MetricsFormatted(
    TRITONSERVER_Metrics* metrics, TRITONSERVER_MetricFormat format,
    const char** base, size_t* byte_size)
{
  TritonServerMetrics* lmetrics =
      reinterpret_cast<TritonServerMetrics*>(metrics);

  switch (format) {
    case TRITONSERVER_METRIC_PROMETHEUS: {
      return lmetrics->Serialize(base, byte_size);
    }

    default:
      break;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("unknown metrics format '" + std::to_string(format) + "'")
          .c_str());
}

//
// TRITONSERVER_InferenceTrace
//
TRITONAPI_DECLSPEC const char*
TRITONSERVER_InferenceTraceLevelString(TRITONSERVER_InferenceTraceLevel level)
{
  switch (level) {
    case TRITONSERVER_TRACE_LEVEL_DISABLED:
      return "DISABLED";
    case TRITONSERVER_TRACE_LEVEL_MIN:
      return "MIN";
    case TRITONSERVER_TRACE_LEVEL_MAX:
      return "MAX";
    case TRITONSERVER_TRACE_LEVEL_TIMESTAMPS:
      return "TIMESTAMPS";
    case TRITONSERVER_TRACE_LEVEL_TENSORS:
      return "TENSORS";
  }

  return "<unknown>";
}

TRITONAPI_DECLSPEC const char*
TRITONSERVER_InferenceTraceActivityString(
    TRITONSERVER_InferenceTraceActivity activity)
{
  switch (activity) {
    case TRITONSERVER_TRACE_REQUEST_START:
      return "REQUEST_START";
    case TRITONSERVER_TRACE_QUEUE_START:
      return "QUEUE_START";
    case TRITONSERVER_TRACE_COMPUTE_START:
      return "COMPUTE_START";
    case TRITONSERVER_TRACE_COMPUTE_INPUT_END:
      return "COMPUTE_INPUT_END";
    case TRITONSERVER_TRACE_COMPUTE_OUTPUT_START:
      return "COMPUTE_OUTPUT_START";
    case TRITONSERVER_TRACE_COMPUTE_END:
      return "COMPUTE_END";
    case TRITONSERVER_TRACE_REQUEST_END:
      return "REQUEST_END";
    case TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT:
      return "TENSOR_QUEUE_INPUT";
    case TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT:
      return "TENSOR_BACKEND_INPUT";
    case TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT:
      return "TENSOR_BACKEND_OUTPUT";
  }

  return "<unknown>";
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceNew(
    TRITONSERVER_InferenceTrace** trace, TRITONSERVER_InferenceTraceLevel level,
    uint64_t parent_id, TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
    TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* trace_userp)
{
#ifdef TRITON_ENABLE_TRACING
  if ((level & TRITONSERVER_TRACE_LEVEL_MIN) > 0) {
    level = static_cast<TRITONSERVER_InferenceTraceLevel>(
        (level ^ TRITONSERVER_TRACE_LEVEL_MIN) |
        TRITONSERVER_TRACE_LEVEL_TIMESTAMPS);
  }
  if ((level & TRITONSERVER_TRACE_LEVEL_MAX) > 0) {
    level = static_cast<TRITONSERVER_InferenceTraceLevel>(
        (level ^ TRITONSERVER_TRACE_LEVEL_MAX) |
        TRITONSERVER_TRACE_LEVEL_TIMESTAMPS);
  }
  tc::InferenceTrace* ltrace = new tc::InferenceTrace(
      level, parent_id, activity_fn, nullptr, release_fn, trace_userp);
  *trace = reinterpret_cast<TRITONSERVER_InferenceTrace*>(ltrace);
  return nullptr;  // Success
#else
  *trace = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceTensorNew(
    TRITONSERVER_InferenceTrace** trace, TRITONSERVER_InferenceTraceLevel level,
    uint64_t parent_id, TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
    TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn,
    TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* trace_userp)
{
#ifdef TRITON_ENABLE_TRACING
  if ((level & TRITONSERVER_TRACE_LEVEL_MIN) > 0) {
    level = static_cast<TRITONSERVER_InferenceTraceLevel>(
        (level ^ TRITONSERVER_TRACE_LEVEL_MIN) |
        TRITONSERVER_TRACE_LEVEL_TIMESTAMPS);
  }
  if ((level & TRITONSERVER_TRACE_LEVEL_MAX) > 0) {
    level = static_cast<TRITONSERVER_InferenceTraceLevel>(
        (level ^ TRITONSERVER_TRACE_LEVEL_MAX) |
        TRITONSERVER_TRACE_LEVEL_TIMESTAMPS);
  }
  tc::InferenceTrace* ltrace = new tc::InferenceTrace(
      level, parent_id, activity_fn, tensor_activity_fn, release_fn,
      trace_userp);
  *trace = reinterpret_cast<TRITONSERVER_InferenceTrace*>(ltrace);
  return nullptr;  // Success
#else
  *trace = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceDelete(TRITONSERVER_InferenceTrace* trace)
{
#ifdef TRITON_ENABLE_TRACING
  tc::InferenceTrace* ltrace = reinterpret_cast<tc::InferenceTrace*>(trace);
  delete ltrace;
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceId(TRITONSERVER_InferenceTrace* trace, uint64_t* id)
{
#ifdef TRITON_ENABLE_TRACING
  tc::InferenceTrace* ltrace = reinterpret_cast<tc::InferenceTrace*>(trace);
  *id = ltrace->Id();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceParentId(
    TRITONSERVER_InferenceTrace* trace, uint64_t* parent_id)
{
#ifdef TRITON_ENABLE_TRACING
  tc::InferenceTrace* ltrace = reinterpret_cast<tc::InferenceTrace*>(trace);
  *parent_id = ltrace->ParentId();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceModelName(
    TRITONSERVER_InferenceTrace* trace, const char** model_name)
{
#ifdef TRITON_ENABLE_TRACING
  tc::InferenceTrace* ltrace = reinterpret_cast<tc::InferenceTrace*>(trace);
  *model_name = ltrace->ModelName().c_str();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceTraceModelVersion(
    TRITONSERVER_InferenceTrace* trace, int64_t* model_version)
{
#ifdef TRITON_ENABLE_TRACING
  tc::InferenceTrace* ltrace = reinterpret_cast<tc::InferenceTrace*>(trace);
  *model_version = ltrace->ModelVersion();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

//
// TRITONSERVER_ServerOptions
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsNew(TRITONSERVER_ServerOptions** options)
{
  *options =
      reinterpret_cast<TRITONSERVER_ServerOptions*>(new TritonServerOptions());
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsDelete(TRITONSERVER_ServerOptions* options)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  delete loptions;
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetServerId(
    TRITONSERVER_ServerOptions* options, const char* server_id)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetServerId(server_id);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelRepositoryPath(
    TRITONSERVER_ServerOptions* options, const char* model_repository_path)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetModelRepositoryPath(model_repository_path);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelControlMode(
    TRITONSERVER_ServerOptions* options, TRITONSERVER_ModelControlMode mode)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  // convert mode from TRITONSERVER_ to triton::core
  switch (mode) {
    case TRITONSERVER_MODEL_CONTROL_NONE: {
      loptions->SetModelControlMode(tc::ModelControlMode::MODE_NONE);
      break;
    }
    case TRITONSERVER_MODEL_CONTROL_POLL: {
      loptions->SetModelControlMode(tc::ModelControlMode::MODE_POLL);
      break;
    }
    case TRITONSERVER_MODEL_CONTROL_EXPLICIT: {
      loptions->SetModelControlMode(tc::ModelControlMode::MODE_EXPLICIT);
      break;
    }
    default: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown control mode '" + std::to_string(mode) + "'")
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStartupModel(
    TRITONSERVER_ServerOptions* options, const char* model_name)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStartupModel(model_name);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitOnError(
    TRITONSERVER_ServerOptions* options, bool exit)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetExitOnError(exit);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictModelConfig(
    TRITONSERVER_ServerOptions* options, bool strict)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStrictModelConfig(strict);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetRateLimiterMode(
    TRITONSERVER_ServerOptions* options, TRITONSERVER_RateLimitMode mode)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  // convert mode from TRITONSERVER_ to triton::core
  switch (mode) {
    case TRITONSERVER_RATE_LIMIT_EXEC_COUNT: {
      loptions->SetRateLimiterMode(tc::RateLimitMode::RL_EXEC_COUNT);
      break;
    }
    case TRITONSERVER_RATE_LIMIT_OFF: {
      loptions->SetRateLimiterMode(tc::RateLimitMode::RL_OFF);
      break;
    }
    default: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown rate limit mode '" + std::to_string(mode) + "'")
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsAddRateLimiterResource(
    TRITONSERVER_ServerOptions* options, const char* name, const size_t count,
    const int device)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  return loptions->AddRateLimiterResource(name, count, device);
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetPinnedMemoryPoolByteSize(size);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, int gpu_device, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetCudaMemoryPoolByteSize(gpu_device, size);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetResponseCacheByteSize(
    TRITONSERVER_ServerOptions* options, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetResponseCacheByteSize(size);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
    TRITONSERVER_ServerOptions* options, double cc)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMinSupportedComputeCapability(cc);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictReadiness(
    TRITONSERVER_ServerOptions* options, bool strict)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStrictReadiness(strict);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitTimeout(
    TRITONSERVER_ServerOptions* options, unsigned int timeout)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetExitTimeout(timeout);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
    TRITONSERVER_ServerOptions* options, unsigned int thread_count)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetBufferManagerThreadCount(thread_count);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
    TRITONSERVER_ServerOptions* options, unsigned int thread_count)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetModelLoadThreadCount(thread_count);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogFile(
    TRITONSERVER_ServerOptions* options, const char* file)
{
#ifdef TRITON_ENABLE_LOGGING
  std::string out_file;
  if (file != nullptr) {
    out_file = std::string(file);
  }
  const std::string& error = LOG_SET_OUT_FILE(out_file);
  if (!error.empty()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (error).c_str());
  }
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogInfo(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_INFO(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

// Enable or disable warning level logging.
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogWarn(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_WARNING(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

// Enable or disable error level logging.
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogError(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_ERROR(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

// Set verbose logging level. Level zero disables verbose logging.
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogVerbose(
    TRITONSERVER_ServerOptions* options, int level)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_SET_VERBOSE(level);
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif             // TRITON_ENABLE_LOGGING
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogFormat(
    TRITONSERVER_ServerOptions* options, const TRITONSERVER_LogFormat format)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  switch (format) {
    case TRITONSERVER_LOG_DEFAULT:
      LOG_SET_FORMAT(triton::common::Logger::Format::kDEFAULT);
      break;
    case TRITONSERVER_LOG_ISO8601:
      LOG_SET_FORMAT(triton::common::Logger::Format::kISO8601);
      break;
  }
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif             // TRITON_ENABLE_LOGGING
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetrics(
    TRITONSERVER_ServerOptions* options, bool metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMetrics(metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetGpuMetrics(
    TRITONSERVER_ServerOptions* options, bool gpu_metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetGpuMetrics(gpu_metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCpuMetrics(
    TRITONSERVER_ServerOptions* options, bool cpu_metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetCpuMetrics(cpu_metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetricsInterval(
    TRITONSERVER_ServerOptions* options, uint64_t metrics_interval_ms)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMetricsInterval(metrics_interval_ms);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBackendDirectory(
    TRITONSERVER_ServerOptions* options, const char* backend_dir)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetBackendDir(backend_dir);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
    TRITONSERVER_ServerOptions* options, const char* repoagent_dir)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetRepoAgentDir(repoagent_dir);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
    TRITONSERVER_ServerOptions* options,
    const TRITONSERVER_InstanceGroupKind kind, const int device_id,
    const double fraction)
{
  if (device_id < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("expects device ID >= 0, got ") +
         std::to_string(device_id))
            .c_str());
  } else if ((fraction < 0.0) || (fraction > 1.0)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("expects limit fraction to be in range [0.0, 1.0], got ") +
         std::to_string(fraction))
            .c_str());
  }

  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  switch (kind) {
    case TRITONSERVER_INSTANCEGROUPKIND_GPU: {
      static std::string key_prefix = "model-load-gpu-limit-device-";
      return loptions->AddBackendConfig(
          "", key_prefix + std::to_string(device_id), std::to_string(fraction));
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("given device kind is not supported, got: ") +
           TRITONSERVER_InstanceGroupKindString(kind))
              .c_str());
  }
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBackendConfig(
    TRITONSERVER_ServerOptions* options, const char* backend_name,
    const char* setting, const char* value)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  return loptions->AddBackendConfig(backend_name, setting, value);
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetHostPolicy(
    TRITONSERVER_ServerOptions* options, const char* policy_name,
    const char* setting, const char* value)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  return loptions->SetHostPolicy(policy_name, setting, value);
}

//
// TRITONSERVER_InferenceRequest
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestNew(
    TRITONSERVER_InferenceRequest** inference_request,
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  std::shared_ptr<tc::Model> model;
  RETURN_IF_STATUS_ERROR(lserver->GetModel(model_name, model_version, &model));

  *inference_request = reinterpret_cast<TRITONSERVER_InferenceRequest*>(
      new tc::InferenceRequest(model, model_version));

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestDelete(
    TRITONSERVER_InferenceRequest* inference_request)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  delete lrequest;
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestId(
    TRITONSERVER_InferenceRequest* inference_request, const char** id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  *id = lrequest->Id().c_str();
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetId(
    TRITONSERVER_InferenceRequest* inference_request, const char* id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  lrequest->SetId(id);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* flags)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  *flags = lrequest->Flags();
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t flags)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  lrequest->SetFlags(flags);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* correlation_id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  const tc::InferenceRequest::SequenceId& corr_id = lrequest->CorrelationId();
  if (corr_id.Type() != tc::InferenceRequest::SequenceId::DataType::UINT64) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("given request's correlation id is not an unsigned int")
            .c_str());
  }
  *correlation_id = corr_id.UnsignedIntValue();
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationIdString(
    TRITONSERVER_InferenceRequest* inference_request,
    const char** correlation_id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  const tc::InferenceRequest::SequenceId& corr_id = lrequest->CorrelationId();
  if (corr_id.Type() != tc::InferenceRequest::SequenceId::DataType::STRING) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("given request's correlation id is not a string").c_str());
  }
  *correlation_id = corr_id.StringValue().c_str();
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t correlation_id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  lrequest->SetCorrelationId(tc::InferenceRequest::SequenceId(correlation_id));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationIdString(
    TRITONSERVER_InferenceRequest* inference_request,
    const char* correlation_id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  if (std::string(correlation_id).length() > 128) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        std::string(
            "string correlation ID cannot be longer than 128 characters")
            .c_str());
  }
  lrequest->SetCorrelationId(tc::InferenceRequest::SequenceId(correlation_id));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* priority)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  *priority = lrequest->Priority();
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t priority)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  lrequest->SetPriority(priority);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* timeout_us)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  *timeout_us = lrequest->TimeoutMicroseconds();
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t timeout_us)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  lrequest->SetTimeoutMicroseconds(timeout_us);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    uint64_t dim_count)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->AddOriginalInput(
      name, tc::TritonToDataType(datatype), shape, dim_count));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRawInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->AddRawInput(name));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveOriginalInput(name));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveAllOriginalInputs());
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);

  tc::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(
      input->AppendData(base, byte_size, memory_type, memory_type_id));

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, const char* host_policy_name)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);

  tc::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(input->AppendDataWithHostPolicy(
      base, byte_size, memory_type, memory_type_id, host_policy_name));

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, TRITONSERVER_BufferAttributes* buffer_attributes)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);

  tc::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(
      input->AppendDataWithBufferAttributes(base, lbuffer_attributes));

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);

  tc::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(input->RemoveAllData());

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->AddOriginalRequestedOutput(name));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveOriginalRequestedOutput(name));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveAllOriginalRequestedOutputs());
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetReleaseCallback(
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceRequestReleaseFn_t request_release_fn,
    void* request_release_userp)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(
      lrequest->SetReleaseCallback(request_release_fn, request_release_userp));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetResponseCallback(
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp)
{
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);
  tc::ResponseAllocator* lallocator =
      reinterpret_cast<tc::ResponseAllocator*>(response_allocator);
  RETURN_IF_STATUS_ERROR(lrequest->SetResponseCallback(
      lallocator, response_allocator_userp, response_fn, response_userp));
  return nullptr;  // Success
}

//
// TRITONSERVER_InferenceResponse
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseDelete(
    TRITONSERVER_InferenceResponse* inference_response)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);
  delete lresponse;
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseError(
    TRITONSERVER_InferenceResponse* inference_response)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);
  RETURN_IF_STATUS_ERROR(lresponse->ResponseStatus());
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseModel(
    TRITONSERVER_InferenceResponse* inference_response, const char** model_name,
    int64_t* model_version)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  *model_name = lresponse->ModelName().c_str();
  *model_version = lresponse->ActualModelVersion();

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseId(
    TRITONSERVER_InferenceResponse* inference_response, const char** request_id)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  *request_id = lresponse->Id().c_str();

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseParameterCount(
    TRITONSERVER_InferenceResponse* inference_response, uint32_t* count)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  const auto& parameters = lresponse->Parameters();
  *count = parameters.size();

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseParameter(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const char** name, TRITONSERVER_ParameterType* type, const void** vvalue)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  const auto& parameters = lresponse->Parameters();
  if (index >= parameters.size()) {
    return TritonServerError::Create(
        TRITONSERVER_ERROR_INVALID_ARG,
        "out of bounds index " + std::to_string(index) +
            std::string(": response has ") + std::to_string(parameters.size()) +
            " parameters");
  }

  const tc::InferenceParameter& param = parameters[index];

  *name = param.Name().c_str();
  *type = param.Type();
  *vvalue = param.ValuePointer();

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputCount(
    TRITONSERVER_InferenceResponse* inference_response, uint32_t* count)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  const auto& outputs = lresponse->Outputs();
  *count = outputs.size();

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutput(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const char** name, TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint64_t* dim_count, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id, void** userp)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  const auto& outputs = lresponse->Outputs();
  if (index >= outputs.size()) {
    return TritonServerError::Create(
        TRITONSERVER_ERROR_INVALID_ARG,
        "out of bounds index " + std::to_string(index) +
            std::string(": response has ") + std::to_string(outputs.size()) +
            " outputs");
  }

  const tc::InferenceResponse::Output& output = outputs[index];

  *name = output.Name().c_str();
  *datatype = tc::DataTypeToTriton(output.DType());

  const std::vector<int64_t>& oshape = output.Shape();
  *shape = &oshape[0];
  *dim_count = oshape.size();

  RETURN_IF_STATUS_ERROR(
      output.DataBuffer(base, byte_size, memory_type, memory_type_id, userp));

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputClassificationLabel(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const size_t class_index, const char** label)
{
  tc::InferenceResponse* lresponse =
      reinterpret_cast<tc::InferenceResponse*>(inference_response);

  const auto& outputs = lresponse->Outputs();
  if (index >= outputs.size()) {
    return TritonServerError::Create(
        TRITONSERVER_ERROR_INVALID_ARG,
        "out of bounds index " + std::to_string(index) +
            std::string(": response has ") + std::to_string(outputs.size()) +
            " outputs");
  }

  const tc::InferenceResponse::Output& output = outputs[index];
  RETURN_IF_STATUS_ERROR(
      lresponse->ClassificationLabel(output, class_index, label));

  return nullptr;  // Success
}

//
// TRITONSERVER_BufferAttributes
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesNew(
    TRITONSERVER_BufferAttributes** buffer_attributes)
{
  tc::BufferAttributes* lbuffer_attributes = new tc::BufferAttributes();
  *buffer_attributes =
      reinterpret_cast<TRITONSERVER_BufferAttributes*>(lbuffer_attributes);

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesDelete(
    TRITONSERVER_BufferAttributes* buffer_attributes)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  delete lbuffer_attributes;

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetMemoryTypeId(
    TRITONSERVER_BufferAttributes* buffer_attributes, int64_t memory_type_id)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  lbuffer_attributes->SetMemoryTypeId(memory_type_id);

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetMemoryType(
    TRITONSERVER_BufferAttributes* buffer_attributes,
    TRITONSERVER_MemoryType memory_type)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  lbuffer_attributes->SetMemoryType(memory_type);

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetCudaIpcHandle(
    TRITONSERVER_BufferAttributes* buffer_attributes, void* cuda_ipc_handle)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  lbuffer_attributes->SetCudaIpcHandle(cuda_ipc_handle);

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesSetByteSize(
    TRITONSERVER_BufferAttributes* buffer_attributes, size_t byte_size)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  lbuffer_attributes->SetByteSize(byte_size);

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesMemoryTypeId(
    TRITONSERVER_BufferAttributes* buffer_attributes, int64_t* memory_type_id)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  *memory_type_id = lbuffer_attributes->MemoryTypeId();

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesMemoryType(
    TRITONSERVER_BufferAttributes* buffer_attributes,
    TRITONSERVER_MemoryType* memory_type)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  *memory_type = lbuffer_attributes->MemoryType();

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesCudaIpcHandle(
    TRITONSERVER_BufferAttributes* buffer_attributes, void** cuda_ipc_handle)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  *cuda_ipc_handle = lbuffer_attributes->CudaIpcHandle();

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesByteSize(
    TRITONSERVER_BufferAttributes* buffer_attributes, size_t* byte_size)
{
  tc::BufferAttributes* lbuffer_attributes =
      reinterpret_cast<tc::BufferAttributes*>(buffer_attributes);
  *byte_size = lbuffer_attributes->ByteSize();

  return nullptr;  // success
}

//
// TRITONSERVER_Server
//
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerNew(
    TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* options)
{
  tc::InferenceServer* lserver = new tc::InferenceServer();
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  NVTX_INITIALIZE;

#ifdef TRITON_ENABLE_METRICS
  // NOTE: Metrics must be enabled before backends are setup
  if (loptions->Metrics()) {
    tc::Metrics::EnableMetrics();
    tc::Metrics::SetMetricsInterval(loptions->MetricsInterval());
  }
#endif  // TRITON_ENABLE_METRICS

  lserver->SetId(loptions->ServerId());
  lserver->SetModelRepositoryPaths(loptions->ModelRepositoryPaths());
  lserver->SetModelControlMode(loptions->ModelControlMode());
  lserver->SetStartupModels(loptions->StartupModels());
  bool strict_model_config = loptions->StrictModelConfig();
  lserver->SetStrictModelConfigEnabled(strict_model_config);
  lserver->SetRateLimiterMode(loptions->RateLimiterMode());
  lserver->SetRateLimiterResources(loptions->RateLimiterResources());
  lserver->SetPinnedMemoryPoolByteSize(loptions->PinnedMemoryPoolByteSize());
  lserver->SetResponseCacheByteSize(loptions->ResponseCacheByteSize());
  lserver->SetCudaMemoryPoolByteSize(loptions->CudaMemoryPoolByteSize());
  double min_compute_capability = loptions->MinSupportedComputeCapability();
  lserver->SetMinSupportedComputeCapability(min_compute_capability);
  lserver->SetStrictReadinessEnabled(loptions->StrictReadiness());
  lserver->SetExitTimeoutSeconds(loptions->ExitTimeout());
  lserver->SetHostPolicyCmdlineConfig(loptions->HostPolicyCmdlineConfigMap());
  lserver->SetRepoAgentDir(loptions->RepoAgentDir());
  lserver->SetBufferManagerThreadCount(loptions->BufferManagerThreadCount());
  lserver->SetModelLoadThreadCount(loptions->ModelLoadThreadCount());

  // SetBackendCmdlineConfig must be called after all AddBackendConfig calls
  // have completed.
  // Note that the auto complete config condition is reverted
  // due to setting name being different
  loptions->AddBackendConfig(
      std::string(), "auto-complete-config",
      strict_model_config ? "false" : "true");
  loptions->AddBackendConfig(
      std::string(), "min-compute-capability",
      std::to_string(min_compute_capability));
  loptions->AddBackendConfig(
      std::string(), "backend-directory", loptions->BackendDir());
  lserver->SetBackendCmdlineConfig(loptions->BackendCmdlineConfigMap());

  // Initialize server
  tc::Status status = lserver->Init();

#ifdef TRITON_ENABLE_METRICS
  if (loptions->Metrics() && lserver->ResponseCacheEnabled()) {
    // NOTE: Cache metrics must be enabled after cache initialized in
    // server->Init()
    tc::Metrics::EnableCacheMetrics(lserver->GetResponseCache());
  }
#ifdef TRITON_ENABLE_METRICS_GPU
  if (loptions->Metrics() && loptions->GpuMetrics()) {
    tc::Metrics::EnableGPUMetrics();
  }
#endif  // TRITON_ENABLE_METRICS_GPU

#ifdef TRITON_ENABLE_METRICS_CPU
  if (loptions->Metrics() && loptions->CpuMetrics()) {
    tc::Metrics::EnableCpuMetrics();
  }
#endif  // TRITON_ENABLE_METRICS_CPU

  const bool poll_metrics =
      (lserver->ResponseCacheEnabled() || loptions->GpuMetrics() ||
       loptions->CpuMetrics());
  if (loptions->Metrics() && poll_metrics) {
    // Start thread to poll enabled metrics periodically
    tc::Metrics::StartPollingThreadSingleton(lserver->GetResponseCache());
  }
#endif  // TRITON_ENABLE_METRICS


  // Setup tritonserver options table
  std::vector<std::string> options_headers;
  options_headers.emplace_back("Option");
  options_headers.emplace_back("Value");

  triton::common::TablePrinter options_table(options_headers);
  options_table.InsertRow(std::vector<std::string>{"server_id", lserver->Id()});
  options_table.InsertRow(
      std::vector<std::string>{"server_version", lserver->Version()});

  auto extensions = lserver->Extensions();
  std::string exts;
  for (const auto& ext : extensions) {
    exts.append(ext);
    exts.append(" ");
  }

  // Remove the trailing space
  if (exts.size() > 0)
    exts.pop_back();

  options_table.InsertRow(std::vector<std::string>{"server_extensions", exts});

  size_t i = 0;
  for (const auto& model_repository_path : lserver->ModelRepositoryPaths()) {
    options_table.InsertRow(std::vector<std::string>{
        "model_repository_path[" + std::to_string(i) + "]",
        model_repository_path});
    ++i;
  }

  std::string model_control_mode;
  auto control_mode = lserver->GetModelControlMode();
  switch (control_mode) {
    case tc::ModelControlMode::MODE_NONE: {
      model_control_mode = "MODE_NONE";
      break;
    }
    case tc::ModelControlMode::MODE_POLL: {
      model_control_mode = "MODE_POLL";
      break;
    }
    case tc::ModelControlMode::MODE_EXPLICIT: {
      model_control_mode = "MODE_EXPLICIT";
      break;
    }
    default: {
      model_control_mode = "<unknown>";
    }
  }
  options_table.InsertRow(
      std::vector<std::string>{"model_control_mode", model_control_mode});

  i = 0;
  for (const auto& startup_model : lserver->StartupModels()) {
    options_table.InsertRow(std::vector<std::string>{
        "startup_models_" + std::to_string(i), startup_model});
    ++i;
  }
  options_table.InsertRow(std::vector<std::string>{
      "strict_model_config",
      std::to_string(lserver->StrictModelConfigEnabled())});
  std::string rate_limit = RateLimitModeToString(lserver->RateLimiterMode());
  options_table.InsertRow(std::vector<std::string>{"rate_limit", rate_limit});
  i = 0;
  for (const auto& device_resources : lserver->RateLimiterResources()) {
    for (const auto& resource : device_resources.second) {
      options_table.InsertRow(std::vector<std::string>{
          "rate_limit_resource[" + std::to_string(i) + "]",
          ResourceString(
              resource.first, resource.second, device_resources.first)});
      ++i;
    }
  }
  options_table.InsertRow(std::vector<std::string>{
      "pinned_memory_pool_byte_size",
      std::to_string(lserver->PinnedMemoryPoolByteSize())});
  for (const auto& cuda_memory_pool : lserver->CudaMemoryPoolByteSize()) {
    options_table.InsertRow(std::vector<std::string>{
        "cuda_memory_pool_byte_size{" + std::to_string(cuda_memory_pool.first) +
            "}",
        std::to_string(cuda_memory_pool.second)});
  }
  options_table.InsertRow(std::vector<std::string>{
      "response_cache_byte_size",
      std::to_string(lserver->ResponseCacheByteSize())});

  std::stringstream compute_capability_ss;
  compute_capability_ss.setf(std::ios::fixed);
  compute_capability_ss.precision(1);
  compute_capability_ss << lserver->MinSupportedComputeCapability();
  options_table.InsertRow(std::vector<std::string>{
      "min_supported_compute_capability", compute_capability_ss.str()});
  options_table.InsertRow(std::vector<std::string>{
      "strict_readiness", std::to_string(lserver->StrictReadinessEnabled())});
  options_table.InsertRow(std::vector<std::string>{
      "exit_timeout", std::to_string(lserver->ExitTimeoutSeconds())});

  std::string options_table_string = options_table.PrintTable();
  LOG_INFO << options_table_string;

  if (!status.IsOk()) {
    if (loptions->ExitOnError()) {
      lserver->Stop(true /* force */);
      delete lserver;
      RETURN_IF_STATUS_ERROR(status);
    }

    LOG_ERROR << status.AsString();
  }

  *server = reinterpret_cast<TRITONSERVER_Server*>(lserver);
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerDelete(TRITONSERVER_Server* server)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  delete lserver;
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerStop(TRITONSERVER_Server* server)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  return nullptr;  // Success
}

TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerRegisterModelRepository(
    TRITONSERVER_Server* server, const char* repository_path,
    const TRITONSERVER_Parameter** name_mapping, const uint32_t mapping_count)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);
  if ((name_mapping == nullptr) && (mapping_count != 0)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "model mappings are not provided while mapping count is non-zero");
  }

  std::unordered_map<std::string, std::string> model_mapping;
  for (size_t i = 0; i < mapping_count; ++i) {
    auto mapping =
        reinterpret_cast<const tc::InferenceParameter*>(name_mapping[i]);
    auto subdir = mapping->Name();

    if (mapping->Type() != TRITONSERVER_PARAMETER_STRING) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "Mapped model name must be a string, found "
              "another type for " +
              subdir)
              .c_str());
    }

    auto model_name =
        std::string(reinterpret_cast<const char*>(mapping->ValuePointer()));

    if (!(model_mapping.emplace(model_name, subdir).second)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("failed to register '") + repository_path +
           "', there is a conflicting mapping for '" + std::string(model_name) +
           "'")
              .c_str());
    }
  }
  RETURN_IF_STATUS_ERROR(
      lserver->RegisterModelRepository(repository_path, model_mapping));
  return nullptr;  // Success
}

TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerUnregisterModelRepository(
    TRITONSERVER_Server* server, const char* repository_path)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);
  RETURN_IF_STATUS_ERROR(lserver->UnregisterModelRepository(repository_path));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerPollModelRepository(TRITONSERVER_Server* server)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);
  RETURN_IF_STATUS_ERROR(lserver->PollModelRepository());
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerIsLive(TRITONSERVER_Server* server, bool* live)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->IsLive(live));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerIsReady(TRITONSERVER_Server* server, bool* ready)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->IsReady(ready));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelIsReady(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, bool* ready)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(
      lserver->ModelIsReady(model_name, model_version, ready));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelBatchProperties(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* flags, void** voidp)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  if (voidp != nullptr) {
    *voidp = nullptr;
  }

  std::shared_ptr<tc::Model> model;
  RETURN_IF_STATUS_ERROR(lserver->GetModel(model_name, model_version, &model));

  if (model->Config().max_batch_size() > 0) {
    *flags = TRITONSERVER_BATCH_FIRST_DIM;
  } else {
    *flags = TRITONSERVER_BATCH_UNKNOWN;
  }

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelTransactionProperties(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* txn_flags, void** voidp)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  if (voidp != nullptr) {
    *voidp = nullptr;
  }

  *txn_flags = 0;

  std::shared_ptr<tc::Model> model;
  RETURN_IF_STATUS_ERROR(lserver->GetModel(model_name, model_version, &model));

  if (model->Config().model_transaction_policy().decoupled()) {
    *txn_flags = TRITONSERVER_TXN_DECOUPLED;
  } else {
    *txn_flags = TRITONSERVER_TXN_ONE_TO_ONE;
  }

  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerMetadata(
    TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  triton::common::TritonJson::Value metadata(
      triton::common::TritonJson::ValueType::OBJECT);

  // Just store string reference in JSON object since it will be
  // serialized to another buffer before lserver->Id() or
  // lserver->Version() lifetime ends.
  RETURN_IF_STATUS_ERROR(metadata.AddStringRef("name", lserver->Id().c_str()));
  RETURN_IF_STATUS_ERROR(
      metadata.AddStringRef("version", lserver->Version().c_str()));

  triton::common::TritonJson::Value extensions(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  const std::vector<const char*>& exts = lserver->Extensions();
  for (const auto ext : exts) {
    RETURN_IF_STATUS_ERROR(extensions.AppendStringRef(ext));
  }

  RETURN_IF_STATUS_ERROR(metadata.Add("extensions", std::move(extensions)));

  *server_metadata = reinterpret_cast<TRITONSERVER_Message*>(
      new tc::TritonServerMessage(metadata));
  return nullptr;  // Success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelMetadata(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, TRITONSERVER_Message** model_metadata)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  std::shared_ptr<tc::Model> model;
  RETURN_IF_STATUS_ERROR(lserver->GetModel(model_name, model_version, &model));

  std::vector<int64_t> ready_versions;
  RETURN_IF_STATUS_ERROR(
      lserver->ModelReadyVersions(model_name, &ready_versions));

  triton::common::TritonJson::Value metadata(
      triton::common::TritonJson::ValueType::OBJECT);

  // Can use string ref in this function even though model can be
  // unloaded and config becomes invalid, because TritonServeMessage
  // serializes the json when it is constructed below.
  RETURN_IF_STATUS_ERROR(metadata.AddStringRef("name", model_name));

  triton::common::TritonJson::Value versions(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  if (model_version != -1) {
    RETURN_IF_STATUS_ERROR(
        versions.AppendString(std::move(std::to_string(model_version))));
  } else {
    for (const auto v : ready_versions) {
      RETURN_IF_STATUS_ERROR(
          versions.AppendString(std::move(std::to_string(v))));
    }
  }

  RETURN_IF_STATUS_ERROR(metadata.Add("versions", std::move(versions)));

  const auto& model_config = model->Config();
  if (!model_config.platform().empty()) {
    RETURN_IF_STATUS_ERROR(
        metadata.AddStringRef("platform", model_config.platform().c_str()));
  } else {
    RETURN_IF_STATUS_ERROR(
        metadata.AddStringRef("platform", model_config.backend().c_str()));
  }

  triton::common::TritonJson::Value inputs(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io : model_config.input()) {
    triton::common::TritonJson::Value io_metadata(
        metadata, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef("name", io.name().c_str()));
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef(
        "datatype", triton::common::DataTypeToProtocolString(io.data_type())));

    // Input shape. If the model supports batching then must include
    // '-1' for the batch dimension.
    triton::common::TritonJson::Value io_metadata_shape(
        metadata, triton::common::TritonJson::ValueType::ARRAY);
    if (model_config.max_batch_size() >= 1) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(-1));
    }
    for (const auto d : io.dims()) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(d));
    }
    RETURN_IF_STATUS_ERROR(
        io_metadata.Add("shape", std::move(io_metadata_shape)));

    RETURN_IF_STATUS_ERROR(inputs.Append(std::move(io_metadata)));
  }
  RETURN_IF_STATUS_ERROR(metadata.Add("inputs", std::move(inputs)));

  triton::common::TritonJson::Value outputs(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io : model_config.output()) {
    triton::common::TritonJson::Value io_metadata(
        metadata, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef("name", io.name().c_str()));
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef(
        "datatype", triton::common::DataTypeToProtocolString(io.data_type())));

    // Output shape. If the model supports batching then must include
    // '-1' for the batch dimension.
    triton::common::TritonJson::Value io_metadata_shape(
        metadata, triton::common::TritonJson::ValueType::ARRAY);
    if (model_config.max_batch_size() >= 1) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(-1));
    }
    for (const auto d : io.dims()) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(d));
    }
    RETURN_IF_STATUS_ERROR(
        io_metadata.Add("shape", std::move(io_metadata_shape)));

    RETURN_IF_STATUS_ERROR(outputs.Append(std::move(io_metadata)));
  }
  RETURN_IF_STATUS_ERROR(metadata.Add("outputs", std::move(outputs)));

  *model_metadata = reinterpret_cast<TRITONSERVER_Message*>(
      new tc::TritonServerMessage(metadata));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelStatistics(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, TRITONSERVER_Message** model_stats)
{
#ifndef TRITON_ENABLE_STATS
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "statistics not supported");
#else

  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  auto model_name_string = std::string(model_name);
  std::map<std::string, std::vector<int64_t>> ready_model_versions;
  if (model_name_string.empty()) {
    RETURN_IF_STATUS_ERROR(lserver->ModelReadyVersions(&ready_model_versions));
  } else {
    std::vector<int64_t> ready_versions;
    RETURN_IF_STATUS_ERROR(
        lserver->ModelReadyVersions(model_name_string, &ready_versions));
    if (ready_versions.empty()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "requested model '" + model_name_string + "' is not available")
              .c_str());
    }

    if (model_version == -1) {
      ready_model_versions.emplace(
          model_name_string, std::move(ready_versions));
    } else {
      bool found = false;
      for (const auto v : ready_versions) {
        if (v == model_version) {
          found = true;
          break;
        }
      }
      if (found) {
        ready_model_versions.emplace(
            model_name_string, std::vector<int64_t>{model_version});
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "requested model version is not available for model '" +
                model_name_string + "'")
                .c_str());
      }
    }
  }

  // Can use string ref in this function because TritonServeMessage
  // serializes the json when it is constructed below.
  triton::common::TritonJson::Value metadata(
      triton::common::TritonJson::ValueType::OBJECT);

  triton::common::TritonJson::Value model_stats_json(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& mv_pair : ready_model_versions) {
    for (const auto& version : mv_pair.second) {
      std::shared_ptr<tc::Model> model;
      RETURN_IF_STATUS_ERROR(lserver->GetModel(mv_pair.first, version, &model));
      const auto& infer_stats = model->StatsAggregator().ImmutableInferStats();
      const auto& infer_batch_stats =
          model->StatsAggregator().ImmutableInferBatchStats();

      triton::common::TritonJson::Value inference_stats(
          metadata, triton::common::TritonJson::ValueType::OBJECT);
      // Compute figures only calculated when not going through cache, so
      // subtract cache_hit count from success count. Cache hit count will
      // simply be 0 when cache is disabled.
      uint64_t compute_count =
          infer_stats.success_count_ - infer_stats.cache_hit_count_;
      SetDurationStat(
          metadata, inference_stats, "success", infer_stats.success_count_,
          infer_stats.request_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "fail", infer_stats.failure_count_,
          infer_stats.failure_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "queue", infer_stats.success_count_,
          infer_stats.queue_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "compute_input", compute_count,
          infer_stats.compute_input_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "compute_infer", compute_count,
          infer_stats.compute_infer_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "compute_output", compute_count,
          infer_stats.compute_output_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "cache_hit", infer_stats.cache_hit_count_,
          infer_stats.cache_hit_lookup_duration_ns_);
      // NOTE: cache_miss_count_ should equal compute_count if non-zero
      SetDurationStat(
          metadata, inference_stats, "cache_miss",
          infer_stats.cache_miss_count_,
          infer_stats.cache_miss_lookup_duration_ns_ +
              infer_stats.cache_miss_insertion_duration_ns_);

      triton::common::TritonJson::Value batch_stats(
          metadata, triton::common::TritonJson::ValueType::ARRAY);
      for (const auto& batch : infer_batch_stats) {
        triton::common::TritonJson::Value batch_stat(
            metadata, triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_STATUS_ERROR(batch_stat.AddUInt("batch_size", batch.first));
        SetDurationStat(
            metadata, batch_stat, "compute_input", batch.second.count_,
            batch.second.compute_input_duration_ns_);
        SetDurationStat(
            metadata, batch_stat, "compute_infer", batch.second.count_,
            batch.second.compute_infer_duration_ns_);
        SetDurationStat(
            metadata, batch_stat, "compute_output", batch.second.count_,
            batch.second.compute_output_duration_ns_);
        RETURN_IF_STATUS_ERROR(batch_stats.Append(std::move(batch_stat)));
      }

      triton::common::TritonJson::Value model_stat(
          metadata, triton::common::TritonJson::ValueType::OBJECT);
      RETURN_IF_STATUS_ERROR(
          model_stat.AddStringRef("name", mv_pair.first.c_str()));
      RETURN_IF_STATUS_ERROR(
          model_stat.AddString("version", std::move(std::to_string(version))));

      RETURN_IF_STATUS_ERROR(model_stat.AddUInt(
          "last_inference", model->StatsAggregator().LastInferenceMs()));
      RETURN_IF_STATUS_ERROR(model_stat.AddUInt(
          "inference_count", model->StatsAggregator().InferenceCount()));
      RETURN_IF_STATUS_ERROR(model_stat.AddUInt(
          "execution_count", model->StatsAggregator().ExecutionCount()));

      RETURN_IF_STATUS_ERROR(
          model_stat.Add("inference_stats", std::move(inference_stats)));
      RETURN_IF_STATUS_ERROR(
          model_stat.Add("batch_stats", std::move(batch_stats)));
      RETURN_IF_STATUS_ERROR(model_stats_json.Append(std::move(model_stat)));
    }
  }

  RETURN_IF_STATUS_ERROR(
      metadata.Add("model_stats", std::move(model_stats_json)));
  *model_stats = reinterpret_cast<TRITONSERVER_Message*>(
      new tc::TritonServerMessage(metadata));

  return nullptr;  // success

#endif  // TRITON_ENABLE_STATS
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelConfig(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, const uint32_t config_version,
    TRITONSERVER_Message** model_config)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  std::shared_ptr<tc::Model> model;
  RETURN_IF_STATUS_ERROR(lserver->GetModel(model_name, model_version, &model));

  std::string model_config_json;
  RETURN_IF_STATUS_ERROR(tc::ModelConfigToJson(
      model->Config(), config_version, &model_config_json));

  *model_config = reinterpret_cast<TRITONSERVER_Message*>(
      new tc::TritonServerMessage(std::move(model_config_json)));

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerModelIndex(
    TRITONSERVER_Server* server, uint32_t flags,
    TRITONSERVER_Message** repository_index)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  const bool ready_only = ((flags & TRITONSERVER_INDEX_FLAG_READY) != 0);

  std::vector<tc::ModelRepositoryManager::ModelIndex> index;
  RETURN_IF_STATUS_ERROR(lserver->RepositoryIndex(ready_only, &index));

  // Can use string ref in this function because TritonServerMessage
  // serializes the json when it is constructed below.
  triton::common::TritonJson::Value repository_index_json(
      triton::common::TritonJson::ValueType::ARRAY);

  for (const auto& in : index) {
    triton::common::TritonJson::Value model_index(
        repository_index_json, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_STATUS_ERROR(model_index.AddStringRef("name", in.name_.c_str()));
    if (!in.name_only_) {
      if (in.version_ >= 0) {
        RETURN_IF_STATUS_ERROR(model_index.AddString(
            "version", std::move(std::to_string(in.version_))));
      }
      RETURN_IF_STATUS_ERROR(model_index.AddStringRef(
          "state", tc::ModelReadyStateString(in.state_).c_str()));
      if (!in.reason_.empty()) {
        RETURN_IF_STATUS_ERROR(
            model_index.AddStringRef("reason", in.reason_.c_str()));
      }
    }

    RETURN_IF_STATUS_ERROR(
        repository_index_json.Append(std::move(model_index)));
  }

  *repository_index = reinterpret_cast<TRITONSERVER_Message*>(
      new tc::TritonServerMessage(repository_index_json));

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerLoadModel(
    TRITONSERVER_Server* server, const char* model_name)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->LoadModel({{std::string(model_name), {}}}));

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerLoadModelWithParameters(
    TRITONSERVER_Server* server, const char* model_name,
    const TRITONSERVER_Parameter** parameters, const uint64_t parameter_count)
{
  if ((parameters == nullptr) && (parameter_count != 0)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "load parameters are not provided while parameter count is non-zero");
  }

  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  std::unordered_map<std::string, std::vector<const tc::InferenceParameter*>>
      models;
  std::vector<const tc::InferenceParameter*> mp;
  for (size_t i = 0; i < parameter_count; ++i) {
    mp.emplace_back(
        reinterpret_cast<const tc::InferenceParameter*>(parameters[i]));
  }
  models[model_name] = std::move(mp);
  RETURN_IF_STATUS_ERROR(lserver->LoadModel(models));

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerUnloadModel(
    TRITONSERVER_Server* server, const char* model_name)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->UnloadModel(
      std::string(model_name), false /* unload_dependents */));

  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerUnloadModelAndDependents(
    TRITONSERVER_Server* server, const char* model_name)
{
  {
    tc::InferenceServer* lserver =
        reinterpret_cast<tc::InferenceServer*>(server);

    RETURN_IF_STATUS_ERROR(lserver->UnloadModel(
        std::string(model_name), true /* unload_dependents */));

    return nullptr;  // success
  }
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerMetrics(
    TRITONSERVER_Server* server, TRITONSERVER_Metrics** metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerMetrics* lmetrics = new TritonServerMetrics();
  *metrics = reinterpret_cast<TRITONSERVER_Metrics*>(lmetrics);
  return nullptr;  // Success
#else
  *metrics = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerInferAsync(
    TRITONSERVER_Server* server,
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceTrace* trace)
{
  tc::InferenceServer* lserver = reinterpret_cast<tc::InferenceServer*>(server);
  tc::InferenceRequest* lrequest =
      reinterpret_cast<tc::InferenceRequest*>(inference_request);

  RETURN_IF_STATUS_ERROR(lrequest->PrepareForInference());

  // Set the trace object in the request so that activity associated
  // with the request can be recorded as the request flows through
  // Triton.
  if (trace != nullptr) {
#ifdef TRITON_ENABLE_TRACING
    tc::InferenceTrace* ltrace = reinterpret_cast<tc::InferenceTrace*>(trace);
    ltrace->SetModelName(lrequest->ModelName());
    ltrace->SetModelVersion(lrequest->ActualModelVersion());

    lrequest->SetTrace(std::make_shared<tc::InferenceTraceProxy>(ltrace));
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
  }

  // We wrap the request in a unique pointer to ensure that it flows
  // through inferencing with clear ownership.
  std::unique_ptr<tc::InferenceRequest> ureq(lrequest);

  // Run inference...
  tc::Status status = lserver->InferAsync(ureq);

  // If there is an error then must explicitly release any trace
  // object associated with the inference request above.
#ifdef TRITON_ENABLE_TRACING
  if (!status.IsOk()) {
    ureq->ReleaseTrace();
  }
#endif  // TRITON_ENABLE_TRACING

  // If there is an error then ureq will still have 'lrequest' and we
  // must release it from unique_ptr since the caller should retain
  // ownership when there is error. If there is not an error then ureq
  // == nullptr and so this release is a nop.
  ureq.release();

  RETURN_IF_STATUS_ERROR(status);
  return nullptr;  // Success
}

//
// TRITONSERVER_MetricFamily
//
TRITONSERVER_Error*
TRITONSERVER_MetricFamilyNew(
    TRITONSERVER_MetricFamily** family, TRITONSERVER_MetricKind kind,
    const char* name, const char* description)
{
#ifdef TRITON_ENABLE_METRICS
  try {
    *family = reinterpret_cast<TRITONSERVER_MetricFamily*>(
        new tc::MetricFamily(kind, name, description));
  }
  catch (std::invalid_argument const& ex) {
    // Catch invalid kinds passed to constructor
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ex.what());
  }
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_MetricFamilyDelete(TRITONSERVER_MetricFamily* family)
{
#ifdef TRITON_ENABLE_METRICS
  auto lfamily = reinterpret_cast<tc::MetricFamily*>(family);
  if (lfamily->NumMetrics() > 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Must call MetricDelete on all dependent metrics before calling "
        "MetricFamilyDelete.");
  }

  delete lfamily;
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

//
// TRITONSERVER_Metric
//
TRITONSERVER_Error*
TRITONSERVER_MetricNew(
    TRITONSERVER_Metric** metric, TRITONSERVER_MetricFamily* family,
    const TRITONSERVER_Parameter** labels, const uint64_t label_count)
{
#ifdef TRITON_ENABLE_METRICS
  std::vector<const tc::InferenceParameter*> labels_vec;
  for (size_t i = 0; i < label_count; i++) {
    labels_vec.emplace_back(
        reinterpret_cast<const tc::InferenceParameter*>(labels[i]));
  }

  try {
    *metric = reinterpret_cast<TRITONSERVER_Metric*>(
        new tc::Metric(family, labels_vec));
  }
  catch (std::invalid_argument const& ex) {
    // Catch invalid kinds passed to constructor
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ex.what());
  }

  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_MetricDelete(TRITONSERVER_Metric* metric)
{
#ifdef TRITON_ENABLE_METRICS
  auto lmetric = reinterpret_cast<tc::Metric*>(metric);
  if (lmetric->Family() == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "MetricFamily reference was invalidated before Metric was deleted. "
        "Must call MetricDelete on all dependent metrics before calling "
        "MetricFamilyDelete.");
  }

  delete lmetric;
  return nullptr;  // success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_MetricValue(TRITONSERVER_Metric* metric, double* value)
{
#ifdef TRITON_ENABLE_METRICS
  return reinterpret_cast<tc::Metric*>(metric)->Value(value);
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_MetricIncrement(TRITONSERVER_Metric* metric, double value)
{
#ifdef TRITON_ENABLE_METRICS
  return reinterpret_cast<tc::Metric*>(metric)->Increment(value);
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_MetricSet(TRITONSERVER_Metric* metric, double value)
{
#ifdef TRITON_ENABLE_METRICS
  return reinterpret_cast<tc::Metric*>(metric)->Set(value);
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_GetMetricKind(
    TRITONSERVER_Metric* metric, TRITONSERVER_MetricKind* kind)
{
#ifdef TRITON_ENABLE_METRICS
  *kind = reinterpret_cast<tc::Metric*>(metric)->Kind();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

}  // extern C
