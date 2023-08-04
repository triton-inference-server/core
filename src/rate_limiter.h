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
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

#include "backend_model.h"
#include "backend_model_instance.h"
#include "instance_queue.h"
#include "model_config.pb.h"
#include "payload.h"
#include "status.h"

namespace triton { namespace core {

// Limits the rate at which requests are dispatched to the model instances
class RateLimiter {
 public:
  using RateLimiterConfig = inference::ModelRateLimiter;
  using ResourceMap = std::map<int, std::map<std::string, size_t>>;
  enum RESOURCE_KIND_KEY {
    // Key for holding global resources
    GLOBAL_RESOURCE_KEY = -2,
    // Key for holding resources per each device
    PER_DEVICE_RESOURCE_KEY = -1
  };

  /// Creates a rate limiter object which will funnel the requests to
  /// the model instances. A typical lifetime of the model instance within
  /// RateLimiter transition from available -> staged -> allocated -> available.
  /// The transition from available to staged occurs when a request is
  /// registered for the model. Depending upon the resource availability and
  /// priority, the RateLimiter will transition an instance to allocated state
  /// at some point in the future. The staged state is skipped when
  /// configured to ignore the resource constraints. The cycle in this case
  /// will be available -> allocated -> available.
  /// \param ignore_resources_and_priority Whether or not to ignore resource
  /// constraints and cross-model priority. An available instance is directly
  /// allocated when true.
  /// \param resource_map The map to the available resource count provided
  /// explicitly.
  /// \return Status object indicating success or failure.
  static Status Create(
      const bool ignore_resources_and_priority, const ResourceMap& resource_map,
      std::unique_ptr<RateLimiter>* rate_limiter);

  /// Registers the model instance with the rate limiter.
  /// \param instance The pointer to the TritonModelInstance object to register
  /// with the rate limiter.
  /// \param rate_limiter_config The rate limiter configuration associated with
  /// the model instance.
  /// \return Status object indicating success or failure.
  Status RegisterModelInstance(
      TritonModelInstance* instance,
      const RateLimiterConfig& rate_limiter_config);

  /// Unregisters the model instance with the rate limiter.
  /// \param instance The pointer to the TritonModelInstance object to
  /// unregister with the rate limiter.
  void UnregisterModelInstance(TritonModelInstance* instance);

  /// Remove model from the set of models being managed by the rate limiter.
  /// \param model The pointer to TritonModel object to be removed.
  void UnregisterModel(const TritonModel* model);

  /// Returns true if there is a payload slot available for the given model.
  /// Note the function can be a blocking call when support_prefetching is
  /// false. In this case, the function will block until a slot is available to
  /// start building the payload. force_non_blocking option can be set to True
  /// to allow function to return back with availability.
  /// \param model The pointer to TritonModel object to query for.
  /// \param model_instance The pointer to TritonMode
  /// \param support_prefetching Whether or not pre-fetching of payloads is
  /// enabled.
  /// \param force_non_blocking When set true, function will not block for
  /// the availability of the slot.
  /// \return slot availability in boolean.
  bool PayloadSlotAvailable(
      const TritonModel* model, const TritonModelInstance* model_instance,
      const bool support_prefetching, const bool force_non_blocking = false);

  /// Enqueues the payload to rate limiter for scheduling on the given model.
  /// \param model The pointer to TritonModel object to be removed.
  /// \param payload The shared pointer to the payload object.
  /// \return Status object indicating success or failure.
  Status EnqueuePayload(
      const TritonModel* model, std::shared_ptr<Payload> payload);

  /// Returns the payload that has been scheduled for the given set of model
  /// instances. Note that this call is blocking and depends upon the
  /// availability of payloads in the rate limiter for the triton model
  /// instance.
  /// \param instance The pointers to TritonModelInstance objects whose
  /// payload is being requested.
  /// \param payload The shared pointer to the payload object.
  void DequeuePayload(
      std::deque<TritonModelInstance*>& instance,
      std::shared_ptr<Payload>* payload);

  /// Returns a new payload object.
  /// \param op_type The operation type for the payload.
  /// \param instance Optional field that providess the model instance that must
  /// be used for the execution of the payload. Default is nullptr which allows
  /// any model instance to execute the payload.
  /// \return The shared pointer to a new payload object.
  std::shared_ptr<Payload> GetPayload(
      const Payload::Operation op_type,
      TritonModelInstance* instance = nullptr);

  /// Releases the given payload object back to the rate limiter.
  /// \param payload The payload to release.
  void PayloadRelease(std::shared_ptr<Payload>& payload);

 private:
  class ModelInstanceContext;
  class ModelContext;
  struct PayloadQueue;
  using StandardReleaseFunc = std::function<void(ModelInstanceContext*)>;
  using StandardScheduleFunc = std::function<void(ModelInstanceContext*)>;
  using StandardStageFunc = std::function<void(ModelInstanceContext*)>;

  // Holds the state of the model instance.
  class ModelInstanceContext {
   public:
    friend class RateLimiter;
    friend class ResourceManager;
    enum State { AVAILABLE, STAGED, ALLOCATED, REMOVED };

    void Release();
    TritonModelInstance* RawInstance() const { return triton_model_instance_; }

   private:
    ModelInstanceContext(
        TritonModelInstance* triton_model_instance, ModelContext* model_context,
        const RateLimiterConfig& rate_limiter_config, StandardStageFunc OnStage,
        StandardReleaseFunc OnRelease);

    const RateLimiterConfig* GetRateLimiterConfig() const
    {
      return &rate_limiter_config_;
    }
    void MarkAvailable();
    double ScaledPriority();
    Status Stage(StandardScheduleFunc OnSchedule);
    Status Allocate();
    Status DirectAllocate(StandardScheduleFunc OnSchedule);
    void RequestRemoval();
    bool IsRemovalInProgress();

    TritonModelInstance* triton_model_instance_;
    ModelContext* model_context_;
    RateLimiterConfig rate_limiter_config_;
    StandardStageFunc OnStage_;
    StandardReleaseFunc OnRelease_;
    std::atomic<uint64_t> exec_count_;

    State state_;
    bool removal_in_progress_;
    std::mutex state_mtx_;

    StandardScheduleFunc OnSchedule_;
  };

  class ScaledPriorityComparator {
   public:
    bool operator()(ModelInstanceContext* a, ModelInstanceContext* b)
    {
      return a->ScaledPriority() > b->ScaledPriority();
    }
  };

  using PriorityQueue = std::priority_queue<
      ModelInstanceContext*, std::vector<ModelInstanceContext*>,
      ScaledPriorityComparator>;

  // Holds the active context to a model
  class ModelContext {
   public:
    ModelContext() : removal_in_progress_(false) {}

    // Enqueue request for obtaining a model instance for scheduling
    // a inference payload execution.
    Status EnqueueModelInstanceRequest(
        const StandardScheduleFunc& OnSchedule,
        TritonModelInstance* triton_model_instance);
    // Marks the given instance of the model as available and ready
    // to be staged.
    void AddAvailableInstance(ModelInstanceContext* instance);
    // Attempts to stage the instance given upon its availability.
    void StageInstanceIfAvailable(TritonModelInstance* triton_model_instance);
    // Allocates one of the staged model instance for execution if resources
    // are available in the system.
    void AllocateInstanceIfAvailable();
    // Adds a queue in the model context for holding requests meant for
    // running on the given instance.
    void AddSpecificRequestQueue(ModelInstanceContext* instance);
    // Whether or not there are any requests waiting for execution on the given
    // model instance.
    bool ContainsPendingRequests(ModelInstanceContext* instance);
    // Remove the given instance of the model. `WaitForRemoval()` on the
    // instance should have been called and returned.
    void RemoveInstance(ModelInstanceContext* instance);
    // Starts the removal of the model context from scheduling purposes.
    // Will wait for all enqueued model instance requests to complete.
    void RequestRemoval() { removal_in_progress_ = true; }
    // Whether or not model context is decommissioned
    bool IsRemovalInProgress() { return removal_in_progress_; }

   private:
    bool removal_in_progress_;

    // Queue holding pending scheduling request
    std::queue<StandardScheduleFunc> generic_sched_request_queue_;
    std::map<const TritonModelInstance*, std::queue<StandardScheduleFunc>>
        specific_sched_request_queues_;
    std::recursive_mutex sched_request_queue_mtx_;

    // The set of instances that are available at the moment
    PriorityQueue avbl_instances_;
    std::recursive_mutex avbl_instances_mtx_;
  };

  // Manages and keep track of resource allocation to the model instances.
  class ResourceManager {
   public:
    static Status Create(
        const ResourceMap& resource_map,
        std::unique_ptr<ResourceManager>* resource_manager);
    // Adds the model instance to the resource manager
    Status AddModelInstance(const ModelInstanceContext* instance);
    // Removes the model instance from the resource manager
    Status RemoveModelInstance(const ModelInstanceContext* instance);
    // Allocate resources for the given model instance. Returns
    // false if resources are not available at this time.
    bool AllocateResources(const ModelInstanceContext* instance);
    // Releases resources held by the given model instance back to
    // the available pool.
    Status ReleaseResources(const ModelInstanceContext* instance);

   private:
    ResourceManager(const ResourceMap& resource_map);
    void ComputeResourceLimits();
    void UpdateMaxResource(const ResourceMap& instance_resource_map);
    Status ParseAndValidateResources();
    Status ValidateMaxResources();
    Status ParseAndValidateExplicitResources();

    ResourceMap explicit_max_resources_;

    std::map<const ModelInstanceContext*, ResourceMap> model_resources_;
    std::mutex model_resources_mtx_;

    ResourceMap max_resources_;
    std::mutex max_resources_mtx_;

    ResourceMap allocated_resources_;
    std::mutex allocated_resources_mtx_;
  };

  RateLimiter(
      const bool ignore_resources_and_priority,
      const ResourceMap& resource_map);

  // Initializes payload queues for the given model instance. The queue
  // holds payloads that get scheduled by rate limiter.
  void InitializePayloadQueues(const TritonModelInstance* instance);

  // Should wait till a consumer registers a pending dequeue request
  // for the given instance(s) of the model. This implies that the
  // call will wait for an idle runner.
  void WaitForConsumer(
      const TritonModel* model, const TritonModelInstance* model_instance);
  // Returns the number of consumers who have a pending dequeue request for
  // the given instance(s) of the model.
  int WaitingConsumerCount(
      const TritonModel* model, const TritonModelInstance* model_instance);

  // Defers scheduling of the payload to the future. Rate Limiter will
  // schedule the payload execution based upon the resource availability/
  // Note that OnSchedule function should only schedule(enqueued in payload
  // queue) the payload and not execute it.
  Status DeferPayloadSchedule(
      const StandardScheduleFunc& OnSchedule, const TritonModel* model,
      TritonModelInstance* instance = nullptr);
  // Callback function to stage the instance.
  void OnStage(ModelInstanceContext* instance_ptr);
  // Callback function to release resources allocated to the instance
  // and attempt allocating available resources to next staged instance.
  void OnRelease(ModelInstanceContext* instance_ptr);
  // Attempt allocating the resources for the staged instance with
  // highest priority.
  void AttemptAllocation();
  // Schedules the payload for execution on model instance.
  void SchedulePayload(
      TritonModelInstance* tmi, PayloadQueue* payload_queue,
      const std::shared_ptr<Payload>& payload);

  bool ignore_resources_and_priority_;

  // Instance context for the models
  std::map<
      const TritonModel*,
      std::map<
          const TritonModelInstance*, std::unique_ptr<ModelInstanceContext>>>
      model_instance_ctxs_;
  std::mutex model_instance_ctx_mtx_;

  // Running context of the models
  std::map<const TritonModel*, ModelContext> model_contexts_;
  std::mutex model_ctx_mtx_;

  // Holds the model instances that have been staged
  PriorityQueue staged_instances_;
  std::recursive_mutex staged_instances_mtx_;

  // Manager to keep track of the resource allocations
  std::unique_ptr<ResourceManager> resource_manager_;
  std::mutex resource_manager_mtx_;

  // Mutex to serialize Payload [de]allocation
  std::mutex payload_mu_;

  // Mutex to serialize Payload Queues deallocation
  std::mutex payload_queues_mu_;

  // Keep some number of Payload objects for reuse to avoid the overhead
  // of creating a Payload for every new request.
  const size_t max_payload_bucket_count_;
  std::vector<std::shared_ptr<Payload>> payload_bucket_;
  std::deque<std::shared_ptr<Payload>> payloads_in_use_;

  struct PayloadQueue {
    explicit PayloadQueue(size_t max_batch_size, uint64_t max_queue_delay_ns)
    {
      queue_.reset(new InstanceQueue(max_batch_size, max_queue_delay_ns));
    }
    std::unique_ptr<InstanceQueue> queue_;
    std::map<const TritonModelInstance*, std::unique_ptr<InstanceQueue>>
        specific_queues_;
    std::mutex mu_;
    std::condition_variable cv_;
  };
  std::map<const TritonModel*, std::unique_ptr<PayloadQueue>> payload_queues_;
};

}}  // namespace triton::core
