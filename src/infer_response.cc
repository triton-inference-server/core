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

#include "infer_response.h"

#include "model.h"
#include "model_config_utils.h"
#include "server.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

//
// InferenceResponseFactory
//
Status
InferenceResponseFactory::CreateResponse(
    std::unique_ptr<InferenceResponse>* response) const
{
  response->reset(new InferenceResponse(
      model_, id_, allocator_, alloc_userp_, response_fn_, response_userp_,
      response_delegator_
#ifdef TRITON_ENABLE_METRICS
      ,
      responses_sent_, infer_start_ns_
#endif  // TRITON_ENABLE_METRICS
      ));
#ifdef TRITON_ENABLE_TRACING
  (*response)->SetTrace(trace_);
#endif  // TRITON_ENABLE_TRACING
  return Status::Success;
}

Status
InferenceResponseFactory::SendFlags(const uint32_t flags) const
{
  if (response_delegator_ != nullptr) {
    std::unique_ptr<InferenceResponse> response(
        new InferenceResponse(response_fn_, response_userp_));
    response_delegator_(std::move(response), flags);
  } else {
    void* userp = response_userp_;
    response_fn_(nullptr /* response */, flags, userp);
  }
  return Status::Success;
}

//
// InferenceResponse
//
InferenceResponse::InferenceResponse(
    const std::shared_ptr<Model>& model, const std::string& id,
    const ResponseAllocator* allocator, void* alloc_userp,
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp,
    const std::function<
        void(std::unique_ptr<InferenceResponse>&&, const uint32_t)>& delegator
#ifdef TRITON_ENABLE_METRICS
    ,
    std::shared_ptr<std::atomic<uint64_t>> responses_sent,
    uint64_t infer_start_ns
#endif  // TRITON_ENABLE_METRICS
    )
    : model_(model), id_(id), allocator_(allocator), alloc_userp_(alloc_userp),
      response_fn_(response_fn), response_userp_(response_userp),
      response_delegator_(delegator),
#ifdef TRITON_ENABLE_METRICS
      responses_sent_(std::move(responses_sent)),
      infer_start_ns_(infer_start_ns),
#endif  // TRITON_ENABLE_METRICS
      null_response_(false)
{
  // If the allocator has a start_fn then invoke it.
  TRITONSERVER_ResponseAllocatorStartFn_t start_fn = allocator_->StartFn();
  if (start_fn != nullptr) {
    LOG_TRITONSERVER_ERROR(
        start_fn(
            reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
                const_cast<ResponseAllocator*>(allocator_)),
            alloc_userp_),
        "response allocation start failed");
  }
}

InferenceResponse::InferenceResponse(
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp)
    : response_fn_(response_fn), response_userp_(response_userp),
#ifdef TRITON_ENABLE_METRICS
      responses_sent_(nullptr), infer_start_ns_(0),
#endif  // TRITON_ENABLE_METRICS
      null_response_(true)
{
}

const std::string&
InferenceResponse::ModelName() const
{
  static const std::string unknown("<unknown>");
  return (model_ == nullptr) ? unknown : model_->Name();
}

int64_t
InferenceResponse::ActualModelVersion() const
{
  return (model_ == nullptr) ? -1 : model_->Version();
}

Status
InferenceResponse::AddParameter(const char* name, const char* value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceResponse::AddParameter(const char* name, const int64_t value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceResponse::AddParameter(const char* name, const bool value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceResponse::AddParameter(const char* name, const double value)
{
  parameters_.emplace_back(name, value);
  return Status::Success;
}

Status
InferenceResponse::AddOutput(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, InferenceResponse::Output** output)
{
  outputs_.emplace_back(name, datatype, shape, allocator_, alloc_userp_);

  LOG_VERBOSE(1) << "add response output: " << outputs_.back();

  if (model_ != nullptr) {
    const inference::ModelOutput* output_config;
    RETURN_IF_ERROR(model_->GetOutput(name, &output_config));
    if (output_config->has_reshape()) {
      const bool has_batch_dim = (model_->Config().max_batch_size() > 0);
      outputs_.back().Reshape(has_batch_dim, output_config);
    }
  }

  if (output != nullptr) {
    *output = std::addressof(outputs_.back());
  }

  return Status::Success;
}

Status
InferenceResponse::AddOutput(
    const std::string& name, const inference::DataType datatype,
    std::vector<int64_t>&& shape, InferenceResponse::Output** output)
{
  outputs_.emplace_back(
      name, datatype, std::move(shape), allocator_, alloc_userp_);

  LOG_VERBOSE(1) << "add response output: " << outputs_.back();

  if (model_ != nullptr) {
    const inference::ModelOutput* output_config;
    RETURN_IF_ERROR(model_->GetOutput(name, &output_config));
    if (output_config->has_reshape()) {
      const bool has_batch_dim = (model_->Config().max_batch_size() > 0);
      outputs_.back().Reshape(has_batch_dim, output_config);
    }
  }

  if (output != nullptr) {
    *output = std::addressof(outputs_.back());
  }

  return Status::Success;
}

Status
InferenceResponse::ClassificationLabel(
    const InferenceResponse::Output& output, const uint32_t class_index,
    const char** label) const
{
  const auto& label_provider = model_->GetLabelProvider();
  const std::string& l = label_provider->GetLabel(output.Name(), class_index);
  if (l.empty()) {
    *label = nullptr;
  } else {
    *label = l.c_str();
  }

  return Status::Success;
}

Status
InferenceResponse::Send(
    std::unique_ptr<InferenceResponse>&& response, const uint32_t flags)
{
#ifdef TRITON_ENABLE_TRACING
  response->TraceOutputTensors(
      TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT, "InferenceResponse Send");
#endif  // TRITON_ENABLE_TRACING

#ifdef TRITON_ENABLE_METRICS
  response->UpdateResponseMetrics();
#endif  // TRITON_ENABLE_METRICS

  if (response->response_delegator_ != nullptr) {
    auto ldelegator = std::move(response->response_delegator_);
    ldelegator(std::move(response), flags);
    return Status::Success;
  }
  void* userp = response->response_userp_;
  if (response->null_response_) {
    response->response_fn_(nullptr /* response */, flags, userp);
  } else {
    auto& response_fn = response->response_fn_;
    response_fn(
        reinterpret_cast<TRITONSERVER_InferenceResponse*>(response.release()),
        flags, userp);
  }
  return Status::Success;
}

Status
InferenceResponse::SendWithStatus(
    std::unique_ptr<InferenceResponse>&& response, const uint32_t flags,
    const Status& status)
{
  response->status_ = status;
  return InferenceResponse::Send(std::move(response), flags);
}

#ifdef TRITON_ENABLE_TRACING
Status
InferenceResponse::TraceOutputTensors(
    TRITONSERVER_InferenceTraceActivity activity, const std::string& msg)
{
  const auto& outputs = this->Outputs();
  uint32_t output_count = outputs.size();

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const Output& output = outputs[idx];

    // output data
    const char* cname = output.Name().c_str();
    TRITONSERVER_DataType datatype = DataTypeToTriton(output.DType());
    const std::vector<int64_t>& oshape = output.Shape();
    const int64_t* shape = &oshape[0];
    uint64_t dim_count = oshape.size();
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    Status status = output.DataBuffer(
        &base, &byte_size, &memory_type, &memory_type_id, &userp);
    if (!status.IsOk()) {
      LOG_STATUS_ERROR(
          status,
          std::string(TRITONSERVER_InferenceTraceActivityString(activity)) +
              ": " + msg + ": fail to get data buffer: " + status.Message());
      return status;
    }

    INFER_TRACE_TENSOR_ACTIVITY(
        this->trace_, activity, cname, datatype, base, byte_size, shape,
        dim_count, memory_type, memory_type_id);
  }

  return Status::Success;
}
#endif  // TRITON_ENABLE_TRACING

#ifdef TRITON_ENABLE_METRICS
void
InferenceResponse::UpdateResponseMetrics() const
{
  // Report inference to first response duration.
  if (model_ != nullptr && responses_sent_ != nullptr &&
      responses_sent_->fetch_add(1, std::memory_order_relaxed) == 0) {
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
    if (auto reporter = model_->MetricReporter()) {
      reporter->ObserveHistogram(
          kFirstResponseHistogram,
          (now_ns - infer_start_ns_) / NANOS_PER_MILLIS);
    }
  }
}
#endif  // TRITON_ENABLE_METRICS

//
// InferenceResponse::Output
//
InferenceResponse::Output::~Output()
{
  Status status = ReleaseDataBuffer();
  if (!status.IsOk()) {
    LOG_ERROR << "failed to release buffer for output '" << name_
              << "': " << status.AsString();
  }
}

void
InferenceResponse::Output::Reshape(
    const bool has_batch_dim, const inference::ModelOutput* output_config)
{
  std::deque<int64_t> variable_size_values;

  const int64_t batch_dim =
      (has_batch_dim && (shape_.size() > 0)) ? shape_[0] : -1;
  const size_t batch_dim_offset = (has_batch_dim) ? 1 : 0;

  const auto& from_shape = output_config->reshape().shape();
  const auto& to_shape = output_config->dims();
  for (int64_t idx = 0; idx < from_shape.size(); idx++) {
    if (from_shape[idx] == -1) {
      variable_size_values.push_back(shape_[idx + batch_dim_offset]);
    }
  }

  shape_.clear();
  if (batch_dim >= 0) {
    shape_.push_back(batch_dim);
  }

  for (const auto& dim : to_shape) {
    if (dim == -1) {
      shape_.push_back(variable_size_values.front());
      variable_size_values.pop_front();
    } else {
      shape_.push_back(dim);
    }
  }
}

Status
InferenceResponse::Output::DataBuffer(
    const void** buffer, size_t* buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
    void** userp) const
{
  *buffer = allocated_buffer_;
  *buffer_byte_size = buffer_attributes_.ByteSize();
  *memory_type = buffer_attributes_.MemoryType();
  *memory_type_id = buffer_attributes_.MemoryTypeId();
  *userp = allocated_userp_;
  return Status::Success;
}

Status
InferenceResponse::Output::AllocateDataBuffer(
    void** buffer, size_t buffer_byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  if (allocated_buffer_ != nullptr) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "allocated buffer for output '" + name_ + "' already exists");
  }

  TRITONSERVER_MemoryType actual_memory_type = *memory_type;
  int64_t actual_memory_type_id = *memory_type_id;
  void* alloc_buffer_userp = nullptr;

  RETURN_IF_TRITONSERVER_ERROR(allocator_->AllocFn()(
      reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
          const_cast<ResponseAllocator*>(allocator_)),
      name_.c_str(), buffer_byte_size, *memory_type, *memory_type_id,
      alloc_userp_, buffer, &alloc_buffer_userp, &actual_memory_type,
      &actual_memory_type_id));

  // Only call the buffer attributes API if it is set.
  if (allocator_->BufferAttributesFn() != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(allocator_->BufferAttributesFn()(
        reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
            const_cast<ResponseAllocator*>(allocator_)),
        name_.c_str(),
        reinterpret_cast<TRITONSERVER_BufferAttributes*>(&buffer_attributes_),
        alloc_userp_, alloc_buffer_userp));
  }

  allocated_buffer_ = *buffer;
  buffer_attributes_.SetByteSize(buffer_byte_size);
  buffer_attributes_.SetMemoryType(actual_memory_type);
  buffer_attributes_.SetMemoryTypeId(actual_memory_type_id);

  allocated_userp_ = alloc_buffer_userp;
  *memory_type = actual_memory_type;
  *memory_type_id = actual_memory_type_id;

  return Status::Success;
}

Status
InferenceResponse::Output::ReleaseDataBuffer()
{
  TRITONSERVER_Error* err = nullptr;

  if (allocated_buffer_ != nullptr) {
    err = allocator_->ReleaseFn()(
        reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
            const_cast<ResponseAllocator*>(allocator_)),
        allocated_buffer_, allocated_userp_, buffer_attributes_.ByteSize(),
        buffer_attributes_.MemoryType(), buffer_attributes_.MemoryTypeId());
  }

  allocated_buffer_ = nullptr;
  buffer_attributes_.SetByteSize(0);
  buffer_attributes_.SetMemoryType(TRITONSERVER_MEMORY_CPU);
  buffer_attributes_.SetMemoryTypeId(0);
  allocated_userp_ = nullptr;

  RETURN_IF_TRITONSERVER_ERROR(err);

  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const InferenceResponse& response)
{
  out << "[0x" << std::addressof(response) << "] "
      << "response id: " << response.Id() << ", model: " << response.ModelName()
      << ", actual version: " << response.ActualModelVersion() << std::endl;

  out << "status:" << response.ResponseStatus().AsString() << std::endl;

  out << "outputs:" << std::endl;
  for (const auto& output : response.Outputs()) {
    out << "[0x" << std::addressof(output) << "] " << output << std::endl;
  }

  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceResponse::Output& output)
{
  out << "output: " << output.Name()
      << ", type: " << triton::common::DataTypeToProtocolString(output.DType())
      << ", shape: " << triton::common::DimsListToString(output.Shape());
  return out;
}

}}  // namespace triton::core
