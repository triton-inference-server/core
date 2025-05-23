// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "buffer_attributes.h"
#include "infer_response.h"
#include "infer_stats.h"
#include "infer_trace.h"
#include "memory.h"
#include "response_allocator.h"
#include "sequence_state.h"
#include "status.h"
#include "triton/common/logging.h"
#include "triton/common/model_config.h"
#include "tritonserver_apis.h"

namespace triton { namespace core {

class Model;
class InferenceServer;
class MetricModelReporter;

//
// An inference request. A request can be used multiple times for
// inference but before each inference run, PrepareForInference() must
// be called to verify and prepare the request. Verification involves
// ensuring that any changes made since the last inference are
// valid. Preparing involves removing/resetting any state left over
// from the previous inference.
//
class InferenceRequest {
 public:
  using InternalReleaseFn =
      std::function<Status(std::unique_ptr<InferenceRequest>&, const uint32_t)>;
  /// State for the request object.
  enum class State {
    // The request has been initialized, but not yet enqueued.
    INITIALIZED,

    // The request is pending execution.
    PENDING,

    // The request failed to enqueue.
    FAILED_ENQUEUE,

    // The request has been picked up by a backend model instance for execution,
    // but hasn't been released yet.
    EXECUTING,

    // The request has been released.
    RELEASED
  };

  // Input tensor
  class Input {
   public:
    enum class TensorType { TENSOR, SHAPE_TENSOR, NON_LINEAR };

    Input();
    Input(
        const std::string& name, const inference::DataType datatype,
        const std::vector<int64_t>& shape);
    Input(
        const std::string& name, const inference::DataType datatype,
        const int64_t* shape, const uint64_t dim_count);

    // Set the name, data type and original shape of the input tensor.
    void SetMetadata(
        const std::string& name, const inference::DataType& dt,
        const std::vector<int64_t>& shape);

    // The name of the input tensor. There is no mutable operator for
    // the name because it is used in a InferenceRequest map and a
    // mutable method would allow it to get out-of-sync.
    const std::string& Name() const { return name_; }

    // Data type of the input tensor.
    inference::DataType DType() const { return datatype_; }

    // The original shape of the input tensor.
    const std::vector<int64_t>& OriginalShape() const
    {
      return original_shape_;
    }

    // The shape of the input tensor after normalization. This shape
    // is the original shape modified as required/expected by
    // inference processing.
    const std::vector<int64_t>& Shape() const { return shape_; }
    std::vector<int64_t>* MutableShape() { return &shape_; }

    // FIXME. Should not need these functions. All shapes kept here
    // should include the batch dimension instead of breaking the same
    // into batch + shape.
    const std::vector<int64_t>& ShapeWithBatchDim() const
    {
      return shape_with_batch_dim_;
    }
    std::vector<int64_t>* MutableShapeWithBatchDim()
    {
      return &shape_with_batch_dim_;
    }

    // Return true if host-specific data was added for this input
    bool HasHostPolicySpecificData() const
    {
      return has_host_policy_specific_data_;
    }

    // Whether or not the input is a tensorrt shape tensor
    bool IsShapeTensor() const
    {
      return tensor_type_ == TensorType::SHAPE_TENSOR;
    }

    // Specifies whether the input uses a non-linear IO format
    bool IsNonLinearFormatIo() const
    {
      return tensor_type_ == TensorType::NON_LINEAR;
    }

    // Set the input to be treated as a shape tensor.
    Status SetIsShapeTensor();

    // Set the input uses a non-linear IO format
    Status SetIsNonLinearFormatIo();

    // The data for this input.
    const std::shared_ptr<Memory>& Data() const { return data_; }

    // The data for this input for a specific device
    const std::shared_ptr<Memory>& Data(
        const std::string& host_policy_name) const;

    // Return all host policy data set for this input
    const std::map<std::string, std::shared_ptr<Memory>>& HostPolicyData() const
    {
      return host_policy_data_map_;
    }

    // Set the data for this input. Error if input already has some
    // data.
    Status SetData(const std::shared_ptr<Memory>& data);

    // Set the data associated with the host policy for this input.
    // Return error if input already has some data.
    Status SetData(
        const std::string& host_policy_name,
        const std::shared_ptr<Memory>& data);

    // Append a new buffer of data to this input.
    Status AppendData(
        const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id);

    Status AppendDataWithHostPolicy(
        const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id, const char* host_policy_name);

    Status AppendDataWithBufferAttributes(
        const void* base, BufferAttributes* buffer_attributes);

    // Prepend a new buffer of data to this input.
    Status PrependData(
        const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id);

    // Remove all existing data for the input.
    Status RemoveAllData();

    // Get the number of buffers containing the input tensor data.
    size_t DataBufferCount() const { return data_->BufferCount(); }

    // Get the number of buffers containing the input tensor data with
    // host policy. If there are no buffers corresponding to the specific
    // host policy, the number of buffers in the fallback input data is
    // returned.
    size_t DataBufferCountForHostPolicy(
        const std::string& host_policy_name) const;

    // Get the 'idx' buffer containing a contiguous chunk of bytes for
    // the input. Return error is 'idx' refers to a buffer that does
    // not exist. Return a pointer to the chunk in 'base' and the
    // size of the chunk in 'byte_size'. 'memory_type' acts as
    // both input and output. On input 'memory_type' is the buffer
    // memory type preferred by the function caller. On return
    // 'memory_type' gives the actual memory type of the chunk pointed
    // to by 'base'.  'memory_type_id' acts as both input and
    // output. On input 'memory_type_id' is the buffer memory type id
    // preferred by the function caller.  On return 'memory_type_id'
    // gives the actual memory type id of the chunk pointed to by
    // 'base'.
    Status DataBuffer(
        const size_t idx, const void** base, size_t* byte_size,
        TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id) const;

    // Get the buffer attributes associated with 'idx' buffer.
    Status DataBufferAttributes(
        const size_t idx, const void** base,
        BufferAttributes** buffer_attributes) const;

    // Get the 'idx' buffer containing a contiguous chunk of bytes for
    // the input. Return error is 'idx' refers to a buffer that does
    // not exist. Return a pointer to the chunk in 'base' and the
    // size of the chunk in 'byte_size'. 'memory_type' acts as
    // both input and output. On input 'memory_type' is the buffer
    // memory type preferred by the function caller. On return
    // 'memory_type' gives the actual memory type of the chunk pointed
    // to by 'base'.  'memory_type_id' acts as both input and
    // output. On input 'memory_type_id' is the buffer memory type id
    // preferred by the function caller.  On return 'memory_type_id'
    // gives the actual memory type id of the chunk pointed to by
    // 'base'.
    Status DataBufferForHostPolicy(
        const size_t idx, const void** base, size_t* byte_size,
        TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
        const std::string& host_policy_name) const;

   private:
    DISALLOW_COPY_AND_ASSIGN(Input);
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceRequest::Input& input);

    std::string name_;
    inference::DataType datatype_;
    std::vector<int64_t> original_shape_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> shape_with_batch_dim_;
    TensorType tensor_type_;
    std::shared_ptr<Memory> data_;

    bool has_host_policy_specific_data_;
    // A map of host policy to input data memory
    std::map<std::string, std::shared_ptr<Memory>> host_policy_data_map_;
  };

  // Sequence ID can be either a 64 bit integer or a string.
  // This class implements the SequenceId type
  class SequenceId {
   public:
    enum class DataType { UINT64, STRING };

    SequenceId();
    SequenceId(const std::string& sequence_label);
    SequenceId(uint64_t sequence_index);
    SequenceId& operator=(const SequenceId& rhs) = default;
    SequenceId& operator=(const std::string& rhs);
    SequenceId& operator=(const uint64_t rhs);

    // Functions that help determine exact type of sequence Id
    DataType Type() const { return id_type_; }
    bool InSequence() const
    {
      return ((sequence_label_ != "") || (sequence_index_ != 0));
    }

    // Get the value of the SequenceId based on the type
    const std::string& StringValue() const { return sequence_label_; }
    uint64_t UnsignedIntValue() const { return sequence_index_; }

   private:
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceRequest::SequenceId& correlation_id);
    friend bool operator==(const SequenceId lhs, const SequenceId rhs);
    friend bool operator!=(const SequenceId lhs, const SequenceId rhs);

    std::string sequence_label_;
    uint64_t sequence_index_;
    DataType id_type_;
  };

  // InferenceRequest
  //
  // The two constructors are identical except one takes model as a
  // shared pointer and the other as a raw pointer. The shared pointer
  // version is the primary one and acts to keep the model alive as
  // long as the request is in flight. The raw pointer version is used
  // only for cases where the model itself is issuing a request
  // (e.g. warmup) and no shared pointer version of the model exists
  // (because we aren't using shared_from_this).
  InferenceRequest(
      const std::shared_ptr<Model>& model,
      const int64_t requested_model_version);

  InferenceRequest(Model* model, const int64_t requested_model_version);

  const std::string& ModelName() const;
  int64_t RequestedModelVersion() const { return requested_model_version_; }
  int64_t ActualModelVersion() const;

  const std::string& Id() const { return id_; }
  void SetId(const std::string& i) { id_ = i; }
  // Return string for logging request ID
  std::string LogRequest() const
  {
    std::string id = Id();
    if (id.empty()) {
      id = "<id_unknown>";
    }
    return std::string("[request id: ") + id + "] ";
  }

  // Flags for the request, union of TRITONSERVER_RequestFlag.
  uint32_t Flags() const { return flags_; }
  void SetFlags(uint32_t f) { flags_ = f; }

  const SequenceId& CorrelationId() const { return correlation_id_; }
  void SetCorrelationId(const SequenceId& c) { correlation_id_ = c; }

  // The batch size of the request, as understood by Triton. A
  // batch-size of 0 indicates that the model doesn't support batching
  // in a way that Triton understands.  Batch size is not set
  // explicitly so there is no setter for it. It is set when the
  // request is normalized.
  uint32_t BatchSize() const { return batch_size_; }

  uint64_t Priority() const { return priority_; }
  void SetPriority(uint64_t p);

  uint64_t TimeoutMicroseconds() const { return timeout_us_; }
  void SetTimeoutMicroseconds(uint64_t t) { timeout_us_ = t; }

  const std::string& CacheKey() const { return cache_key_; }
  // It is up to the user to update the cache_key_ if modifying any hashable
  // fields of the request after cache_key_is_set_ has been set to true.
  void SetCacheKey(const std::string& key)
  {
    cache_key_ = key;
    cache_key_is_set_ = true;
  }
  bool CacheKeyIsSet() const { return cache_key_is_set_; }

  // Define and validate state transitions for request.
  Status SetState(InferenceRequest::State state);

#ifdef TRITON_ENABLE_TRACING
  const std::shared_ptr<InferenceTraceProxy>& TraceProxy() const
  {
    return trace_;
  }
  std::shared_ptr<InferenceTraceProxy>* MutableTrace() { return &trace_; }
  void SetTrace(const std::shared_ptr<InferenceTraceProxy>& trace)
  {
    trace_ = trace;
    response_factory_->SetTrace(trace);
  }
  void ReleaseTrace()
  {
    trace_ = nullptr;
    response_factory_->ReleaseTrace();
  }

  Status TraceInputTensors(
      TRITONSERVER_InferenceTraceActivity activity, const std::string& msg);
#endif  // TRITON_ENABLE_TRACING

  // Add a parameter to the request.
  Status AddParameter(const char* name, const char* value);
  Status AddParameter(const char* name, const int64_t value);
  Status AddParameter(const char* name, const bool value);
  Status AddParameter(const char* name, const double value);
  Status SetParameters(const std::deque<InferenceParameter>& parameters);
  const std::deque<InferenceParameter>& Parameters() const
  {
    return parameters_;
  }


  // The original inputs are the inputs added to the request before
  // the inference execution (that is before
  // TRITONSERVER_ServerInferAsync is called). Once execution has
  // started the original inputs should not be modified until
  // execution completes (and those modifications will apply to the
  // next inference execution).
  Status MutableOriginalInput(const std::string& name, Input** input);
  std::unordered_map<std::string, Input>* MutableOriginalInputs()
  {
    return &original_inputs_;
  }
  const std::unordered_map<std::string, Input>& OriginalInputs() const
  {
    return original_inputs_;
  }

  // The override inputs are the inputs added to the request after
  // inference execution has started (that is after
  // TRITONSERVER_ServerInferAsync or equivalent is called). During
  // inference processing, if Triton needs to change an original input
  // it will add an override instead of changing the original. Triton
  // will also use an override if it needs to add a new input to the
  // request. Overrides are recorded as shared_ptr so that the same
  // override can be used efficiently multiple times or even in
  // multiple requests simultaneously. Must be careful not to modify
  // an override input if it is being shared unless you want that
  // change to be reflected in all requests that hold that override
  // input. Override inputs within a specific request are not
  // persisted across inference calls.
  std::unordered_map<std::string, std::shared_ptr<Input>>*
  MutableOverrideInputs()
  {
    return &override_inputs_;
  }
  const std::unordered_map<std::string, std::shared_ptr<Input>>&
  OverrideInputs() const
  {
    return override_inputs_;
  }

  // Get an input taking into account both original inputs and
  // overrides. If an override input is available use it, otherwise
  // use the original input. Accessing inputs via this method is not
  // valid until after PrepareForInference is called.
  Status ImmutableInput(const std::string& name, const Input** input) const;
  const std::unordered_map<std::string, Input*>& ImmutableInputs() const
  {
    return inputs_;
  }

  // The original requested outputs are the requested outputs added to
  // the request before the inference execution (that is before
  // TRITONSERVER_ServerInferAsync is called). Once execution has
  // started the original requested outputs should not be modified
  // until execution completes (and those modifications will apply to
  // the next inference execution).
  const std::set<std::string>& OriginalRequestedOutputs() const
  {
    return original_requested_outputs_;
  }

  // Get the requested outputs that should be used during
  // inference. Accessing outputs via this method is not valid until
  // after PrepareForInference is called.
  const std::set<std::string>& ImmutableRequestedOutputs() const
  {
    return (requested_outputs_.empty()) ? original_requested_outputs_
                                        : requested_outputs_;
  }

  // Get the response factory.
  const std::shared_ptr<InferenceResponseFactory>& ResponseFactory() const
  {
    return response_factory_;
  }

  // Add an original input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  Status AddOriginalInput(
      const std::string& name, const inference::DataType datatype,
      const int64_t* shape, const uint64_t dim_count, Input** input = nullptr);
  Status AddOriginalInput(
      const std::string& name, const inference::DataType datatype,
      const std::vector<int64_t>& shape, Input** input = nullptr);

  // Add an original raw input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  Status AddRawInput(const std::string& name, Input** input = nullptr);

  // Remove a single original input or all inputs.
  Status RemoveOriginalInput(const std::string& name);
  Status RemoveAllOriginalInputs();

  // Add an override input to the request. If 'input' is non-null
  // return a pointer to the newly added input.
  // FIXME passing batch size is special handling for backend API.
  // For override input, the 'shape' is without batch dimension for
  // backends that implemented w/o backend API (which need correct
  // input.Shape()), but backend API uses input.ShapeWithBatchDim().
  Status AddOverrideInput(
      const std::string& name, const inference::DataType datatype,
      const int64_t batch_size, const std::vector<int64_t>& shape,
      std::shared_ptr<Input>* input = nullptr);

  // Add an override input to the request.
  Status AddOverrideInput(const std::shared_ptr<Input>& input);

  // Request an original requested output.
  Status AddOriginalRequestedOutput(const std::string& name);

  // Remove a single original requested output or all requested
  // outputs.
  Status RemoveOriginalRequestedOutput(const std::string& name);
  Status RemoveAllOriginalRequestedOutputs();

  // Initialize the release callback for the request.
  Status SetReleaseCallback(
      TRITONSERVER_InferenceRequestReleaseFn_t release_fn, void* release_userp)
  {
    release_fn_ = release_fn;
    release_userp_ = release_userp;
    return Status::Success;
  }

  // Initialize the response factory arguments that are going to be used with
  // any responses produced for this request.
  Status SetResponseCallback(
      const ResponseAllocator* allocator, void* alloc_userp,
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp)
  {
    response_allocator_ = allocator;
    alloc_userp_ = alloc_userp;
    response_callback_ = response_fn;
    response_userp_ = response_userp;
    return Status::Success;
  }

  // Returns the preferred memory type and memory type ID of the output buffer
  // for the request. 'name' and 'byte_size' are optional and set to nullptr
  // if not specified, if provided, they give the allocator more information.
  // 'memory_type' and 'memory_type_id' are also used as input to provide types
  // preferred by the caller.
  // Status::Code::UNAVAILABLE will be returned if output properties are not
  // available.
  Status OutputBufferProperties(
      const char* name, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
      int64_t* memory_type_id);

  // Add a callback to be invoked on releasing the request object from Triton.
  // Multiple callbacks can be added by calling this function in order,
  // and they will be invoked in reversed order.
  Status AddInternalReleaseCallback(InternalReleaseFn&& callback)
  {
    release_callbacks_.emplace_back(std::move(callback));
    return Status::Success;
  }

  // Add a delegator to be invoked on sending the responses of this request.
  // The response will be passed to 'delegator' and 'delegator' must call the
  // InferenceResponse::Send() to send the response.
  Status SetResponseDelegator(
      std::function<void(
          std::unique_ptr<InferenceResponse>&&, const uint32_t)>&& delegator)
  {
    response_delegator_ = std::move(delegator);
    return response_factory_->SetResponseDelegator(response_delegator_);
  }

  Status SetSequenceStates(
      const std::shared_ptr<SequenceStates>& sequence_states)
  {
    sequence_states_ = sequence_states;
    return Status::Success;
  }

  Status LoadInputStates();

  void SetResponseFactory()
  {
    response_factory_.reset(new InferenceResponseFactory(
        model_shared_, id_, response_allocator_, alloc_userp_,
        response_callback_, response_userp_, response_delegator_));
  }

  const std::shared_ptr<SequenceStates>& GetSequenceStates() const
  {
    return sequence_states_;
  }

  // Prepare this request for inference.
  Status PrepareForInference();

  // Run this inference request using the model associated with the
  // request. If Status::Success is returned then the call has taken
  // ownership of the request object and so 'request' will be
  // nullptr. If non-success is returned then the caller still retains
  // ownership of 'request'.
  static Status Run(std::unique_ptr<InferenceRequest>& request);

  // Send an error response for this request. If 'status' is Success
  // then no response is sent and the request is not released (even if
  // 'release_request' is true). Because this is sending an error it
  // is assumed that this is the last response for the request and so
  // the FINAL flag is set in the response callback. If
  // 'release_request' is true then the release callback is called for
  // this request and ownership is given to the callback. Thus, if
  // 'release_request' is true 'request' is returned as nullptr.
  static void RespondIfError(
      std::unique_ptr<InferenceRequest>& request, const Status& status,
      const bool release_request = false,
      FailureReason reason = FailureReason::OTHER);

  // Send an error response to a set of 'requests'. If 'status' is
  // Success then no responses are sent and the requests are not
  // released (even if 'release_request' is true). Because this is
  // sending an error it is assumed that this is the last response for
  // the requests and so the FINAL flag is set in the response
  // callbacks. If 'release_request' is true then the release callback
  // is called for each request, and the request ownership is given to
  // the callback. Thus, if 'release_request' is true 'requests' is
  // returned with all nullptrs.
  static void RespondIfError(
      std::vector<std::unique_ptr<InferenceRequest>>& requests,
      const Status& status, const bool release_requests = false,
      FailureReason reason = FailureReason::OTHER);

  // Release the request. Call the release callback and transfer
  // ownership of the request to the callback. On return 'request' is
  // nullptr.
  static Status Release(
      std::unique_ptr<InferenceRequest>&& request,
      const uint32_t release_flags);

  // Create a copy of 'from' suitable for use as a "null" request as
  // required for the direct sequence batcher. The returned copy will
  // contain only the minimum content required for a null request.
  // The statistics of the copy will not be collected.
  static InferenceRequest* CopyAsNull(const InferenceRequest& from);

  uint64_t QueueStartNs() const { return queue_start_ns_; }
  uint64_t CaptureQueueStartNs()
  {
    queue_start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
    return queue_start_ns_;
  }

  uint64_t CacheLookupStartNs() const { return cache_lookup_start_ns_; }
  uint64_t CaptureCacheLookupStartNs()
  {
    cache_lookup_start_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    return cache_lookup_start_ns_;
  }

  uint64_t CacheLookupEndNs() const { return cache_lookup_end_ns_; }
  uint64_t CaptureCacheLookupEndNs()
  {
    cache_lookup_end_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    return cache_lookup_end_ns_;
  }

  uint64_t BatcherStartNs() const { return batcher_start_ns_; }
  uint64_t CaptureBatcherStartNs()
  {
    batcher_start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    return batcher_start_ns_;
  }

#ifdef TRITON_ENABLE_STATS
  uint64_t RequestStartNs() const { return request_start_ns_; }
  uint64_t CaptureRequestStartNs()
  {
    request_start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    return request_start_ns_;
  }

  // Report the statistics to stats collectors associated with the request.
  // Duration and timestamps provide two granularities for stats collectors.
  void ReportStatistics(
      MetricModelReporter* metric_reporter, bool success,
      const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
      const uint64_t compute_output_start_ns, const uint64_t compute_end_ns);

  // Report the error statistics to stats collectors associated with the
  // request.
  // FIXME: A separate function may not be necessary here, but is being used
  // cautiously in case of unforeseen issues such as possibly capturing a trace
  // twice. This should be revisited and better tested to see if the
  // ReportStatistics function can be used as-is for the newly captured failure
  // cases.
  void ReportErrorStatistics(
      MetricModelReporter* metric_reporter, FailureReason reason);

  // Report the statistics to stats collectors associated with the request.
  // Duration and timestamps provide two granularities for stats collectors.
  void ReportStatisticsWithDuration(
      MetricModelReporter* metric_reporter, bool success,
      const uint64_t compute_start_ns, const uint64_t compute_input_duration_ns,
      const uint64_t compute_infer_duration_ns,
      const uint64_t compute_output_duration_ns);

  // Report the statistics to stats collectors associated with the request on
  // response cache hits. Cache miss stats will be updated through model object
  // directly because the backend may release the request object.
  void ReportStatisticsCacheHit(MetricModelReporter* metric_reporter);

  // Statistics for each request are aggregated into the corresponding
  // model's statistics. Optionally this function may be used to
  // add an additional aggregator where statistics are also aggregated.
  void SetSecondaryStatsAggregator(
      InferenceStatsAggregator* secondary_stats_aggregator)
  {
    secondary_stats_aggregator_ = secondary_stats_aggregator;
  }
#endif  // TRITON_ENABLE_STATS

  // Mark the request as cancelled.
  Status Cancel()
  {
    if (!response_factory_) {
      return Status(
          Status::Code::INTERNAL,
          "It is not possible to cancel an inference request before calling "
          "TRITONSERVER_InferAsync.");
    }
    response_factory_->Cancel();
    return Status::Success;
  }

  // Check if the request is marked as cancelled. This does not indicate if the
  // request is actually cancelled. The request is cancelled if and only if it
  // is responded with a cancelled status.
  Status IsCancelled(bool* is_cancelled)
  {
    if (!response_factory_) {
      return Status(
          Status::Code::INTERNAL,
          "It is not possible to query cancellation status before calling "
          "TRITONSERVER_InferAsync.");
    }
    *is_cancelled = response_factory_->IsCancelled();
    return Status::Success;
  }

  bool IsCancelled()
  {
    bool is_cancelled = false;
    Status status = IsCancelled(&is_cancelled);
    if (!status.IsOk()) {
      LOG_ERROR << status.Message();
    }
    return is_cancelled;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(InferenceRequest);
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceRequest& request);
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceRequest::State& state);


  Status Normalize();

  // Helper for validating Inputs
  Status ValidateRequestInputs() const;

  Status ValidateBytesInputs(
      const std::string& input_id, const Input& input,
      const std::string& model_name) const;

  Status ValidateCorrelationId() const;

  // Helpers for pending request metrics
  void IncrementPendingRequestCount();
  void DecrementPendingRequestCount();

  // Has anything in the request potentially changed in a way that
  // causes normalization to be required when preparing the request
  // for inference.
  bool needs_normalization_;

  // The model associated with this request. For most requests
  // model_shared_ will be non-null and will act to keep the model
  // alive as long as this request is live. In this case model_raw_
  // will be the raw pointer from the shared pointer. For cases where
  // the model itself created the request (like running requests for
  // warmup), model_shared_ will be nullptr, but model_raw_ will
  // still be defined. Thus model_raw_ is always defined and should
  // always to used to access the model.
  std::shared_ptr<Model> model_shared_;
  Model* model_raw_;

  // The model version as requested and based on version policy the
  // specific version that is actually used for inference.
  int64_t requested_model_version_;
  int64_t actual_model_version_;

  std::string id_;

  uint32_t flags_;
  SequenceId correlation_id_;
  uint32_t batch_size_;
  uint64_t priority_;
  uint64_t timeout_us_;
  std::string cache_key_ = "";
  // Helper to determine if request was successfully hashed
  // and cache_key_ field is valid
  bool cache_key_is_set_ = false;

  std::unordered_map<std::string, Input> original_inputs_;
  std::unordered_map<std::string, std::shared_ptr<Input>> override_inputs_;
  std::unordered_map<std::string, Input*> inputs_;
  std::set<std::string> original_requested_outputs_;
  std::string raw_input_name_;
  uint32_t raw_input_size_;

  // requested_outputs_ is to be used post-normalization. It will be
  // empty unless it differs from original_requested_outputs_, so
  // typically should access it through ImmutableRequestedOutputs.
  std::set<std::string> requested_outputs_;

  // The release function and user pointer for this request.
  TRITONSERVER_InferenceRequestReleaseFn_t release_fn_;
  void* release_userp_;

  // Additional release callbacks invoked before 'release_fn_'.
  std::vector<InternalReleaseFn> release_callbacks_;

  // Delegator to be invoked on sending responses.
  std::function<void(std::unique_ptr<InferenceResponse>&&, const uint32_t)>
      response_delegator_;

  // The response factory associated with this request.
  std::shared_ptr<InferenceResponseFactory> response_factory_;

  // Request timestamps. Queue start is needed for schedulers even
  // when statistics are not being collected.
  uint64_t queue_start_ns_;

  // Cache lookup start/end timestamps. Cache manages its own stats even
  // when statistics are not being colleceted.
  uint64_t cache_lookup_start_ns_;
  uint64_t cache_lookup_end_ns_;

  // Cache insertion start/end timestamps. Cache manages its own stats even
  // when statistics are not being colleceted.
  uint64_t cache_insertion_start_ns_;
  uint64_t cache_insertion_end_ns_;

  // Dedicated timestamp for batcher internal which can diverge from
  // queue start timestamp to provide accurate queue time without affecting
  // batcher functionalities.
  uint64_t batcher_start_ns_;

  // Whether the stats of the request should be collected.
  bool collect_stats_;

  // The parameters of the request. Use a deque so that there is no
  // reallocation.
  std::deque<InferenceParameter> parameters_;

#ifdef TRITON_ENABLE_STATS
  uint64_t request_start_ns_;
  InferenceStatsAggregator* secondary_stats_aggregator_ = nullptr;
#endif  // TRITON_ENABLE_STATS

#ifdef TRITON_ENABLE_TRACING
  // Inference trace associated with this request.
  std::shared_ptr<InferenceTraceProxy> trace_;
#endif  // TRITON_ENABLE_TRACING

  // Sequence I/O states used for implicit state.
  std::shared_ptr<SequenceStates> sequence_states_;

  // The state of the request.
  std::atomic<InferenceRequest::State> state_;
  // Whether this is a null request used for direct sequence batch padding or
  // not.
  bool null_request_;

  // Response factory arguments
  const ResponseAllocator* response_allocator_;
  void* response_userp_;
  void* alloc_userp_;
  TRITONSERVER_InferenceResponseCompleteFn_t response_callback_;
};

std::ostream& operator<<(std::ostream& out, const InferenceRequest& request);
std::ostream& operator<<(
    std::ostream& out, const InferenceRequest::Input& input);
std::ostream& operator<<(
    std::ostream& out, const InferenceRequest::SequenceId& sequence_id);
bool operator==(
    const InferenceRequest::SequenceId lhs,
    const InferenceRequest::SequenceId rhs);
}}  // namespace triton::core

namespace std {
using namespace triton::core;
template <>
class hash<InferenceRequest::SequenceId> {
 public:
  size_t operator()(const InferenceRequest::SequenceId& sequence_id) const
  {
    if (sequence_id.Type() == InferenceRequest::SequenceId::DataType::STRING) {
      return std::hash<std::string>{}(sequence_id.StringValue());
    }
    return std::hash<uint64_t>{}(sequence_id.UnsignedIntValue());
  }
};
}  // namespace std
