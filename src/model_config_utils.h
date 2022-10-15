// Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "model_config.pb.h"
#include "status.h"
#include "triton/common/model_config.h"
#include "tritonserver_apis.h"
#include "filesystem.h"

namespace triton { namespace core {

/// Enumeration for the different backend types.
enum BackendType {
  BACKEND_TYPE_UNKNOWN = 0,
  BACKEND_TYPE_TENSORRT = 1,
  BACKEND_TYPE_TENSORFLOW = 2,
  BACKEND_TYPE_ONNXRUNTIME = 3,
  BACKEND_TYPE_PYTORCH = 4
};

// Get version of a model from the path containing the model
/// definition file.
/// \param path The path to the model definition file.
/// \param version Returns the version.
/// \return The error status.
Status GetModelVersionFromPath(const std::string& path, int64_t* version);

/// Get the tensor name, false value, and true value for a boolean
/// sequence batcher control kind. If 'required' is true then must
/// find a tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor.
Status GetBooleanSequenceControlProperties(
    const inference::ModelSequenceBatching& batcher,
    const std::string& model_name,
    const inference::ModelSequenceBatching::Control::Kind control_kind,
    const bool required, std::string* tensor_name,
    inference::DataType* tensor_datatype, float* fp32_false_value,
    float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value, bool* bool_false_value, bool* bool_true_value);

/// Get the tensor name and datatype for a non-boolean sequence
/// batcher control kind. If 'required' is true then must find a
/// tensor for the control. If 'required' is false, return
/// 'tensor_name' as empty-string if the control is not mapped to any
/// tensor. 'tensor_datatype' returns the required datatype for the
/// control.
Status GetTypedSequenceControlProperties(
    const inference::ModelSequenceBatching& batcher,
    const std::string& model_name,
    const inference::ModelSequenceBatching::Control::Kind control_kind,
    const bool required, std::string* tensor_name,
    inference::DataType* tensor_datatype);

/// Read a ModelConfig and normalize it as expected by model backends.
/// \param path The full-path to the directory containing the
/// model configuration.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \param config Returns the normalized model configuration.
/// \return The error status.
Status GetNormalizedModelConfig(
    const std::string& model_name, const std::string& path,
    const double min_compute_capability, inference::ModelConfig* config);

/// Auto-complete backend related fields (platform, backend and default model
/// filename) if not set, note that only Triton recognized backends will be
/// checked.
/// \param model_name The name of the model.
/// \param model_path The full-path to the directory containing the
/// model configuration.
/// \param config Returns the auto-completed model configuration.
/// \return The error status.
Status AutoCompleteBackendFields(
    const std::string& model_name, const std::string& model_path,
    inference::ModelConfig* config);

/// Detects and adds missing fields in the model configuration.
/// \param min_compute_capability The minimum supported CUDA compute
/// capability.
/// \param config The model configuration
/// \return The error status
Status NormalizeModelConfig(
    const double min_compute_capability, inference::ModelConfig* config);

/// [FIXME] better formalize config normalization / validation
/// Detects and adds missing fields in instance group setting.
/// \param min_compute_capability The minimum supported CUDA compute
/// capability.
/// \param config The model configuration
/// \return The error status
Status NormalizeInstanceGroup(
    const double min_compute_capability,
    const std::vector<inference::ModelInstanceGroup>& preferred_groups,
    inference::ModelConfig* config);

/// [FIXME] Remove once a more permanent solution is implemented  (DLIS-4211)
/// Localize EXECUTION_ENV_PATH in python backend.
/// \param model_path The full-path to the directory containing the model
/// configuration, before localization.
/// \param config The model configuration
/// \param localized_model_dir The localized model directory
/// \return The error status
Status LocalizePythonBackendExecutionEnvironmentPath(
    const std::string& model_path, inference::ModelConfig* config,
    std::shared_ptr<LocalizedPath>* localized_model_dir);

/// Auto-complete the instance count based on instance kind and backend name.
/// \param group The instance group to set the count for.
/// \param backend The backend name to check against.
/// \return The error status.
Status SetDefaultInstanceCount(
    inference::ModelInstanceGroup* group, const std::string& backend);

/// Validate that a model is specified correctly, except for model inputs
/// and outputs. ValidateModelIOConfig() should be called to
/// validate model inputs and outputs.
/// \param config The model configuration to validate.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
Status ValidateModelConfig(
    const inference::ModelConfig& config, const double min_compute_capability);

/// [FIXME] better formalize config normalization / validation
/// Validate instance group setting.
/// \param config The model configuration to validate.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
Status ValidateInstanceGroup(
    const inference::ModelConfig& config, const double min_compute_capability);

/// Validate that a model inputs and outputs are specified correctly.
/// \param config The model configuration to validate.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
Status ValidateModelIOConfig(const inference::ModelConfig& config);

/// Validate that input is specified correctly in a model
/// configuration.
/// \param io The model input.
/// \param max_batch_size The max batch size specified in model configuration.
/// \param platform The platform name
/// \return The error status. A non-OK status indicates the input
/// is not valid.
Status ValidateModelInput(
    const inference::ModelInput& io, int32_t max_batch_size,
    const std::string& platform);

/// Validate that an input matches one of the allowed input names.
/// \param io The model input.
/// \param allowed The set of allowed input names.
/// \return The error status. A non-OK status indicates the input
/// is not valid.
Status CheckAllowedModelInput(
    const inference::ModelInput& io, const std::set<std::string>& allowed);

/// Validate that an output is specified correctly in a model
/// configuration.
/// \param io The model output.
/// \param max_batch_size The max batch size specified in model configuration.
/// \param platform The platform name
/// \return The error status. A non-OK status indicates the output
/// is not valid.
Status ValidateModelOutput(
    const inference::ModelOutput& io, int32_t max_batch_size,
    const std::string& platform);

/// Validate that an output matches one of the allowed output names.
/// \param io The model output.
/// \param allowed The set of allowed output names.
/// \return The error status. A non-OK status indicates the output
/// is not valid.
Status CheckAllowedModelOutput(
    const inference::ModelOutput& io, const std::set<std::string>& allowed);

/// Validate that a model batch inputs and batch outputs are specified
/// correctly.
/// \param config The model configuration to validate..
/// \return The error status. A non-OK status indicates the batch inputs or
/// batch outputs are not valid.
Status ValidateBatchIO(const inference::ModelConfig& config);

/// Parse the 'value' of the parameter 'key' into a boolean value.
/// \param key The name of the parameter.
/// \param value The value of the parameter in string.
/// \param parsed_value Return the boolean of the parameter.
/// \return The error status. A non-OK status indicates failure on parsing the
/// value.
Status ParseBoolParameter(
    const std::string& key, std::string value, bool* parsed_value);

/// Parse the 'value' of the parameter 'key' into a long long integer value.
/// \param key The name of the parameter.
/// \param value The value of the parameter in string.
/// \param parsed_value Return the numerical value of the parameter.
/// \return The error status. A non-OK status indicates failure on parsing the
/// value.
Status ParseLongLongParameter(
    const std::string& key, const std::string& value, int64_t* parsed_value);

/// Obtain the 'profile_index' of the 'profile_name'.
/// \param profile_name The name of the profile.
/// \param profile_index Return the index of the profile.
/// \return The error status. A non-OK status indicates failure on getting the
/// value.
Status GetProfileIndex(const std::string& profile_name, int* profile_index);

/// Convert a model configuration protobuf to the equivalent json.
/// \param config The protobuf model configuration.
/// \param config_version The model configuration will be returned in
/// a format matching this version. If the configuration cannot be
/// represented in the requested version's format then an error will
/// be returned.
/// \param json Returns the equivalent JSON.
/// \return The error status.
Status ModelConfigToJson(
    const inference::ModelConfig& config, const uint32_t config_version,
    std::string* json_str);

/// Convert a model configuration JSON to the equivalent protobuf.
/// \param config The JSON model configuration.
/// \param config_version The model configuration will be returned in
/// a format matching this version. If the configuration cannot be
/// represented in the requested version's format then an error will
/// be returned.
/// \param protobuf Returns the equivalent protobuf.
/// \return The error status.
Status JsonToModelConfig(
    const std::string& json_config, const uint32_t config_version,
    inference::ModelConfig* protobuf_config);

/// Get the BackendType value for a platform name.
/// \param platform_name The platform name.
/// \return The BackendType or BackendType::UNKNOWN if the platform string
/// is not recognized.
BackendType GetBackendTypeFromPlatform(const std::string& platform_name);

/// Get the BackendType value for a backend name.
/// \param backend_name The backend name.
/// \return The BackendType or BackendType::UNKNOWN if the platform string
/// is not recognized.
BackendType GetBackendType(const std::string& backend_name);

/// Get the Triton server data type corresponding to a data type.
/// \param dtype The data type.
/// \return The Triton server data type.
TRITONSERVER_DataType DataTypeToTriton(const inference::DataType dtype);

/// Get the data type corresponding to a Triton server data type.
/// \param dtype The Triton server data type.
/// \return The data type.
inference::DataType TritonToDataType(const TRITONSERVER_DataType dtype);

}}  // namespace triton::core
