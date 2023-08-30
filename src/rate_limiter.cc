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

#include "rate_limiter.h"

#include <limits>

#include "triton/common/logging.h"

namespace triton { namespace core {

constexpr size_t MAX_PAYLOAD_BUCKET_COUNT = 1000;

//=========================================================================
//  Core Implementation
//=========================================================================

Status
RateLimiter::Create(
    const bool ignore_resources_and_priority,
    const RateLimiter::ResourceMap& resource_map,
    std::unique_ptr<RateLimiter>* rate_limiter)
{
  std::unique_ptr<RateLimiter> local_rate_limiter(
      new RateLimiter(ignore_resources_and_priority, resource_map));
  *rate_limiter = std::move(local_rate_limiter);

  return Status::Success;
}

Status
RateLimiter::RegisterModelInstance(
    TritonModelInstance* triton_model_instance,
    const RateLimiterConfig& rate_limiter_config)
{
  {
    std::lock_guard<std::mutex> lk1(model_ctx_mtx_);
    std::lock_guard<std::mutex> lk2(model_instance_ctx_mtx_);

    auto& model_context = model_contexts_[triton_model_instance->Model()];
    auto& model_instances =
        model_instance_ctxs_[triton_model_instance->Model()];

    auto pair_it = model_instances.emplace(
        triton_model_instance,
        std::unique_ptr<ModelInstanceContext>(new ModelInstanceContext(
            triton_model_instance, &model_context, rate_limiter_config,
            [this](ModelInstanceContext* instance) { OnStage(instance); },
            [this](ModelInstanceContext* instance) { OnRelease(instance); })));
    model_context.AddAvailableInstance(pair_it.first->second.get());
    model_context.AddSpecificRequestQueue(pair_it.first->second.get());

    if (!ignore_resources_and_priority_) {
      // As there can be multiple models being loaded concurrently, need
      // to hold a lock to protect the resource counts.
      // Without this serialization instances of other models might fail
      // to load because of the resource constraints in this instance.
      std::lock_guard<std::mutex> lk(resource_manager_mtx_);
      const auto& status =
          resource_manager_->AddModelInstance(pair_it.first->second.get());
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            resource_manager_->RemoveModelInstance(pair_it.first->second.get()),
            "Cannot remove instance from resource manager");
        return status;
      }
    }
  }

  InitializePayloadQueues(triton_model_instance);

  return Status::Success;
}

void
RateLimiter::UnregisterModelInstance(TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::mutex> lk1(model_ctx_mtx_);
  std::lock_guard<std::mutex> lk2(model_instance_ctx_mtx_);

  const TritonModel* model = triton_model_instance->Model();

  auto& model_context = model_contexts_[model];
  auto& model_instances = model_instance_ctxs_[model];
  auto i_it = model_instances.find(triton_model_instance);
  if (i_it != model_instances.end()) {
    if (!ignore_resources_and_priority_) {
      LOG_STATUS_ERROR(
          resource_manager_->RemoveModelInstance(i_it->second.get()),
          "Cannot remove instance from resource manager");
    }
    model_context.RemoveInstance(i_it->second.get());
    model_instances.erase(i_it);
  }

  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    auto p_it = payload_queues_.find(model);
    if (p_it != payload_queues_.end()) {
      auto s_it = p_it->second->specific_queues_.find(triton_model_instance);
      if (s_it != p_it->second->specific_queues_.end()) {
        p_it->second->specific_queues_.erase(s_it);
      }
    }
  }
}

void
RateLimiter::UnregisterModel(const TritonModel* model)
{
  {
    std::lock_guard<std::mutex> lk1(model_ctx_mtx_);
    std::lock_guard<std::mutex> lk2(model_instance_ctx_mtx_);

    auto& model_context = model_contexts_[model];

    model_context.RequestRemoval();
    for (const auto& instance : model_instance_ctxs_[model]) {
      if (!ignore_resources_and_priority_) {
        LOG_STATUS_ERROR(
            resource_manager_->RemoveModelInstance(instance.second.get()),
            "Cannot remove instance from resource manager");
      }
    }

    model_instance_ctxs_.erase(model);
    model_contexts_.erase(model);
  }

  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    if (payload_queues_.find(model) != payload_queues_.end()) {
      payload_queues_.erase(model);
    }
  }
}

void
RateLimiter::WaitForConsumer(
    const TritonModel* model, const TritonModelInstance* model_instance)
{
  PayloadQueue* payload_queue = nullptr;
  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    if (payload_queues_.find(model) == payload_queues_.end()) {
      LOG_ERROR << "Unable to find the payload queue for the model "
                << model->Name();
      return;
    }
    payload_queue = payload_queues_[model].get();
  }

  if (model_instance == nullptr) {
    payload_queue->queue_->WaitForConsumer();
  } else {
    payload_queue->specific_queues_[model_instance]->WaitForConsumer();
  }
}


int
RateLimiter::WaitingConsumerCount(
    const TritonModel* model, const TritonModelInstance* model_instance)
{
  PayloadQueue* payload_queue = nullptr;
  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    if (payload_queues_.find(model) == payload_queues_.end()) {
      LOG_ERROR << "Unable to find the payload queue for the model "
                << model->Name();
      return 0;
    }
    payload_queue = payload_queues_[model].get();
  }

  if (model_instance == nullptr) {
    return payload_queue->queue_->WaitingConsumerCount();
  } else {
    return payload_queue->specific_queues_[model_instance]
        ->WaitingConsumerCount();
  }
}

bool
RateLimiter::PayloadSlotAvailable(
    const TritonModel* model, const TritonModelInstance* model_instance,
    const bool support_prefetching, const bool force_non_blocking)
{
  bool result;
  if (support_prefetching) {
    PayloadQueue* payload_queue = nullptr;
    {
      std::lock_guard<std::mutex> lk(payload_queues_mu_);
      payload_queue = payload_queues_[model].get();
    }
    {
      std::lock_guard<std::mutex> lk(payload_queue->mu_);
      // The logic below sets cap on the number of payloads that
      // can be pre-fetched. For per-model batcher the cap is
      // twice the number of model instances. For per-instance
      // batcher the cap is 2.
      size_t multiplier = (model_instance == nullptr)
                              ? payload_queue->specific_queues_.size()
                              : 1;
      result = payload_queue->queue_->Size() < (2 * multiplier);
    }
  } else {
    result = true;
    if (force_non_blocking) {
      result = (WaitingConsumerCount(model, model_instance) > 0);
    } else {
      WaitForConsumer(model, model_instance);
    }
  }
  return result;
}

Status
RateLimiter::EnqueuePayload(
    const TritonModel* model, std::shared_ptr<Payload> payload)
{
  auto pinstance = payload->GetInstance();
  PayloadQueue* payload_queue = nullptr;
  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    if (payload_queues_.find(model) == payload_queues_.end()) {
      return Status(
          Status::Code::INTERNAL,
          "Unable to find the payload queue for the model " + model->Name());
    }
    payload_queue = payload_queues_[model].get();
  }

  // Update the pending consumer counts to prevent additional
  // requests from getting enqueued.
  if (pinstance != nullptr) {
    payload_queue->specific_queues_[pinstance]->DecrementConsumerCount();
  }
  payload_queue->queue_->DecrementConsumerCount();

  {
    std::lock_guard<std::mutex> lk(payload_queue->mu_);
    payload->SetState(Payload::State::REQUESTED);
    if (ignore_resources_and_priority_) {
      SchedulePayload(pinstance, payload_queue, payload);
    }
  }
  if (ignore_resources_and_priority_) {
    if (pinstance == nullptr) {
      payload_queue->cv_.notify_one();
    } else {
      payload_queue->cv_.notify_all();
    }
  } else {
    StandardScheduleFunc sched_func = [this, payload_queue,
                                       payload](ModelInstanceContext* mi) {
      {
        std::lock_guard<std::mutex> lk(payload_queue->mu_);
        auto cb = [mi]() { mi->Release(); };
        payload->AddInternalReleaseCallback(cb);
        this->SchedulePayload(mi->RawInstance(), payload_queue, payload);
      }
      if (mi->RawInstance() == nullptr) {
        payload_queue->cv_.notify_one();
      } else {
        payload_queue->cv_.notify_all();
      }
    };
    DeferPayloadSchedule(sched_func, model, payload->GetInstance());
  }
  return Status::Success;
}

void
RateLimiter::DequeuePayload(
    std::deque<TritonModelInstance*>& instances,
    std::shared_ptr<Payload>* payload)
{
  payload->reset();
  PayloadQueue* payload_queue = nullptr;
  auto model = instances[0]->Model();
  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    if (payload_queues_.find(model) == payload_queues_.end()) {
      LOG_ERROR << "Unable to find the payload queue for the model "
                << model->Name();
      return;
    }
    payload_queue = payload_queues_[model].get();
  }

  // Update the queue to reflect availability of a waiting
  // consumer.
  payload_queue->queue_->IncrementConsumerCount();
  for (const auto instance : instances) {
    payload_queue->specific_queues_[instance]->IncrementConsumerCount();
  }

  std::vector<std::shared_ptr<Payload>> merged_payloads;
  size_t instance_index = std::numeric_limits<std::size_t>::max();
  {
    std::unique_lock<std::mutex> lk(payload_queue->mu_);
    payload_queue->cv_.wait(lk, [&instances, &instance_index, payload_queue]() {
      bool empty = payload_queue->queue_->Empty();
      if (empty) {
        instance_index = 0;
        for (const auto instance : instances) {
          empty = payload_queue->specific_queues_[instance]->Empty();
          if (empty) {
            instance_index++;
          } else {
            break;
          }
        }
      }
      return !empty;
    });
    if (instance_index < instances.size()) {
      TritonModelInstance* instance = instances[instance_index];
      if (!payload_queue->specific_queues_[instance]->Empty()) {
        payload_queue->specific_queues_[instance]->Dequeue(
            payload, &merged_payloads);
      }
    } else {
      payload_queue->queue_->Dequeue(payload, &merged_payloads);
    }
  }
  for (auto& merge_payload : merged_payloads) {
    PayloadRelease(merge_payload);
  }
  // Call specified callback, notifying that payloads have been dequeued/merged.
  (*payload)->Callback();
  if ((*payload)->GetInstance() == nullptr) {
    (*payload)->SetInstance(instances.front());
    // Enqueue did not specify the specific instance to
    // run with the payload. Hence, need to explicitly
    // decrement the consumer count for the instance
    // which got allocated.
    payload_queue->specific_queues_[instances.front()]
        ->DecrementConsumerCount();
    instances.pop_front();
  } else {
    instances.erase(instances.begin() + instance_index);
  }

  // Decrement the counts from the remaining specific
  // instance handling as there will be no consumer for
  // these queues.
  // FIXME: DLIS-5238 For more accurate handling, the
  // consumer count for the instances that were not
  // requested should be decremented upon the
  // EnqueuePayload too. This will need instance
  // association to be derived via instances fed into
  // DequeuePayload call.
  // However, as multiple instances are provided to
  // DequeuePayload call only when using device-blocking
  // and a single consumer thread, we are decrementing the
  // specific instance consumer count as an approximation.
  for (const auto instance : instances) {
    payload_queue->specific_queues_[instance]->DecrementConsumerCount();
  }
}

std::shared_ptr<Payload>
RateLimiter::GetPayload(
    const Payload::Operation op_type, TritonModelInstance* instance)
{
  std::shared_ptr<Payload> payload;

  if (max_payload_bucket_count_ > 0) {
    std::lock_guard<std::mutex> lock(payload_mu_);

    if (!payload_bucket_.empty()) {
      payload = payload_bucket_.back();
      payload_bucket_.pop_back();
    }
    if (payload.get() == nullptr && (!payloads_in_use_.empty())) {
      // Just checking the front of the queue instead the entire queue for
      // an available payload to save time.
      if (payloads_in_use_.front().use_count() == 1) {
        payload = payloads_in_use_.front();
        payloads_in_use_.pop_front();
      }
    }
  }

  if (payload.get() == nullptr) {
    payload.reset(new Payload());
  }

  payload->Reset(op_type, instance);
  return payload;
}

void
RateLimiter::PayloadRelease(std::shared_ptr<Payload>& payload)
{
  // If this is an exit payload, the instance must not be staged once marked as
  // available, so mark the instance as removing.
  if (payload->GetOpType() == Payload::Operation::EXIT) {
    std::lock_guard<std::mutex> lk(model_instance_ctx_mtx_);
    auto it_model = model_instance_ctxs_.find(payload->GetInstance()->Model());
    if (it_model == model_instance_ctxs_.end()) {
      LOG_ERROR << "Should not print this! Releasing payload containing an "
                   "instance of an unknown model.";
      return;
    }
    auto it_instance = it_model->second.find(payload->GetInstance());
    if (it_instance == it_model->second.end()) {
      LOG_ERROR << "Should not print this! Releasing payload containing an "
                   "unknown instance.";
      return;
    }
    it_instance->second->RequestRemoval();  // mark the instance as removing
  }

  payload->OnRelease();
  if (max_payload_bucket_count_ > 0) {
    std::lock_guard<std::mutex> lock(payload_mu_);

    if (payloads_in_use_.size() + payload_bucket_.size() <
        max_payload_bucket_count_) {
      // Release iff the payload shared_ptr is uniquely held.
      if (payload.use_count() == 1) {
        payload->Release();
        payload_bucket_.push_back(std::move(payload));
        return;
      } else {
        payloads_in_use_.push_back(std::move(payload));
      }
    }
  }
}

RateLimiter::RateLimiter(
    const bool ignore_resources_and_priority, const ResourceMap& resource_map)
    : ignore_resources_and_priority_(ignore_resources_and_priority),
      max_payload_bucket_count_(MAX_PAYLOAD_BUCKET_COUNT)
{
  ResourceManager::Create(resource_map, &resource_manager_);
}

void
RateLimiter::InitializePayloadQueues(const TritonModelInstance* instance)
{
  auto& config = instance->Model()->Config();
  uint64_t max_queue_delay_microseconds;
  if (config.has_sequence_batching()) {
    const auto& batcher_config = config.sequence_batching();
    if (batcher_config.has_oldest()) {
      max_queue_delay_microseconds =
          batcher_config.oldest().max_queue_delay_microseconds();
    } else {
      max_queue_delay_microseconds = 0;
    }
  } else if (config.has_dynamic_batching()) {
    max_queue_delay_microseconds =
        config.dynamic_batching().max_queue_delay_microseconds();
  } else {
    max_queue_delay_microseconds = 0;
  }
  PayloadQueue* payload_queue = nullptr;
  {
    std::lock_guard<std::mutex> lk(payload_queues_mu_);
    if (payload_queues_.find(instance->Model()) == payload_queues_.end()) {
      payload_queues_.emplace(
          instance->Model(),
          new PayloadQueue(
              config.max_batch_size(), max_queue_delay_microseconds * 1000));
    }
    payload_queue = payload_queues_[instance->Model()].get();
  }
  {
    // NOTE: payload_queue can have a data race because instance->Model()
    // is the same for multiple instances of same model, so protect it when
    // creating model instances in parallel.
    std::lock_guard<std::mutex> lk(payload_queue->mu_);
    if (payload_queue->specific_queues_.find(instance) ==
        payload_queue->specific_queues_.end()) {
      payload_queue->specific_queues_.emplace(
          instance,
          new InstanceQueue(
              config.max_batch_size(), max_queue_delay_microseconds * 1000));
    }
  }
}

Status
RateLimiter::DeferPayloadSchedule(
    const StandardScheduleFunc& OnSchedule, const TritonModel* model,
    TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::mutex> lk(model_ctx_mtx_);

  auto itr = model_contexts_.find(model);
  if (itr == model_contexts_.end()) {
    return Status(
        Status::Code::INTERNAL,
        "Requested model is not yet registered with rate limiter");
  }

  if (itr->second.IsRemovalInProgress()) {
    return Status(
        Status::Code::INTERNAL,
        "New model requests can not be made to a model that is being "
        "removed");
  }

  itr->second.EnqueueModelInstanceRequest(OnSchedule, triton_model_instance);
  itr->second.StageInstanceIfAvailable(triton_model_instance);

  return Status::Success;
}

void
RateLimiter::SchedulePayload(
    TritonModelInstance* tmi, PayloadQueue* payload_queue,
    const std::shared_ptr<Payload>& payload)
{
  if (tmi == nullptr) {
    payload_queue->queue_->Enqueue(payload);
  } else {
    payload_queue->specific_queues_[tmi]->Enqueue(payload);
  }
  payload->SetState(Payload::State::SCHEDULED);
}

void
RateLimiter::OnStage(ModelInstanceContext* instance)
{
  {
    std::lock_guard<std::recursive_mutex> lk(staged_instances_mtx_);
    staged_instances_.push(instance);
  }
  AttemptAllocation();
}

void
RateLimiter::OnRelease(ModelInstanceContext* instance)
{
  {
    std::lock_guard<std::mutex> lk(model_ctx_mtx_);
    auto& model_context = model_contexts_[instance->RawInstance()->Model()];
    model_context.AddAvailableInstance(instance);
    resource_manager_->ReleaseResources(instance);
    if (model_context.ContainsPendingRequests(instance)) {
      model_context.StageInstanceIfAvailable(instance->RawInstance());
    }
  }
  AttemptAllocation();
}

void
RateLimiter::AttemptAllocation()
{
  std::lock_guard<std::recursive_mutex> lk(staged_instances_mtx_);
  if (!staged_instances_.empty()) {
    ModelInstanceContext* instance = staged_instances_.top();
    if (resource_manager_->AllocateResources(instance)) {
      staged_instances_.pop();
      instance->Allocate();
    }
  }
}

//=========================================================================
//  ModelContext Implementation
//=========================================================================

Status
RateLimiter::ModelContext::EnqueueModelInstanceRequest(
    const StandardScheduleFunc& OnSchedule,
    TritonModelInstance* triton_model_instance)
{
  std::lock_guard<std::recursive_mutex> lk(sched_request_queue_mtx_);

  if (triton_model_instance == nullptr) {
    generic_sched_request_queue_.push(OnSchedule);
  } else {
    auto it = specific_sched_request_queues_.find(triton_model_instance);
    if (it != specific_sched_request_queues_.end()) {
      it->second.push(OnSchedule);
    } else {
      return Status(
          Status::Code::INTERNAL,
          "instance not added to specific request queue");
    }
  }

  return Status::Success;
}

void
RateLimiter::ModelContext::AddAvailableInstance(ModelInstanceContext* instance)
{
  std::lock_guard<std::recursive_mutex> lk(avbl_instances_mtx_);
  avbl_instances_.push(instance);
  instance->MarkAvailable();
}


void
RateLimiter::ModelContext::StageInstanceIfAvailable(
    TritonModelInstance* req_instance)
{
  std::lock_guard<std::recursive_mutex> lk1(sched_request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);
  PriorityQueue backup_queue;

  while (!avbl_instances_.empty()) {
    ModelInstanceContext* instance = avbl_instances_.top();
    if (instance->IsRemovalInProgress() ||
        (req_instance != nullptr && instance->RawInstance() != req_instance)) {
      // Skip staging the available instance if either it is being removed or it
      // is not the requested instance (if specified).
      backup_queue.push(instance);
      avbl_instances_.pop();
      continue;
    }
    if (!specific_sched_request_queues_[instance->RawInstance()].empty()) {
      // Prioritize the specific requests for the available model
      // instance highest priority.
      const StandardScheduleFunc func =
          specific_sched_request_queues_[instance->RawInstance()].front();
      specific_sched_request_queues_[instance->RawInstance()].pop();
      instance->Stage(func);
    } else if (!generic_sched_request_queue_.empty()) {
      // If request is for generic model instance then use the
      // instance with the highest priority.
      const StandardScheduleFunc func = generic_sched_request_queue_.front();
      generic_sched_request_queue_.pop();
      instance->Stage(func);
    } else {
      // If there are requests for a specific model instance then backup
      // the model instance and keep searching through the available
      // model instances. The prioritization will be taken care of in the
      // staging priority queue.
      backup_queue.push(instance);
    }
    avbl_instances_.pop();
  }
  // Restore the backup queue
  if (!backup_queue.empty()) {
    avbl_instances_.swap(backup_queue);
  }
}

void
RateLimiter::ModelContext::AllocateInstanceIfAvailable()
{
  std::lock_guard<std::recursive_mutex> lk1(sched_request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);
  PriorityQueue backup_queue;
  while (!avbl_instances_.empty()) {
    ModelInstanceContext* instance = avbl_instances_.top();
    if (!specific_sched_request_queues_[instance->RawInstance()].empty()) {
      // Prioritize the specific requests for the available model
      // instance highest priority.
      const StandardScheduleFunc func =
          specific_sched_request_queues_[instance->RawInstance()].front();
      specific_sched_request_queues_[instance->RawInstance()].pop();
      instance->DirectAllocate(func);
    } else if (!generic_sched_request_queue_.empty()) {
      // If request is for generic model instance then use the
      // instance with the highest priority.
      const StandardScheduleFunc func = generic_sched_request_queue_.front();
      generic_sched_request_queue_.pop();
      instance->DirectAllocate(func);
    } else {
      // If there are requests for a specific model instance then backup
      // the model instance and keep searching through the available
      // model instances. The prioritization will be taken care of in the
      // staging priority queue.
      backup_queue.push(instance);
    }
    avbl_instances_.pop();
  }
  // Restore the backup queue
  if (!backup_queue.empty()) {
    avbl_instances_.swap(backup_queue);
  }
}

void
RateLimiter::ModelContext::AddSpecificRequestQueue(
    ModelInstanceContext* instance)
{
  std::lock_guard<std::recursive_mutex> lk(sched_request_queue_mtx_);
  specific_sched_request_queues_[instance->RawInstance()];
}

bool
RateLimiter::ModelContext::ContainsPendingRequests(
    ModelInstanceContext* instance)
{
  std::lock_guard<std::recursive_mutex> lk(sched_request_queue_mtx_);
  return (generic_sched_request_queue_.size() != 0) ||
         (specific_sched_request_queues_[instance->RawInstance()].size() != 0);
}

void
RateLimiter::ModelContext::RemoveInstance(ModelInstanceContext* instance)
{
  std::lock_guard<std::recursive_mutex> lk1(sched_request_queue_mtx_);
  std::lock_guard<std::recursive_mutex> lk2(avbl_instances_mtx_);

  PriorityQueue new_avbl_instances;
  while (!avbl_instances_.empty()) {
    ModelInstanceContext* curr_instance = avbl_instances_.top();
    if (curr_instance != instance) {
      new_avbl_instances.push(curr_instance);
    }
    avbl_instances_.pop();
  }
  avbl_instances_.swap(new_avbl_instances);

  specific_sched_request_queues_.erase(instance->RawInstance());
}


//=========================================================================
//  ModelInstanceContext Implementation
//=========================================================================

RateLimiter::ModelInstanceContext::ModelInstanceContext(
    TritonModelInstance* triton_model_instance,
    RateLimiter::ModelContext* model_context,
    const RateLimiter::RateLimiterConfig& rate_limiter_config,
    RateLimiter::StandardStageFunc OnStage,
    RateLimiter::StandardReleaseFunc OnRelease)
    : triton_model_instance_(triton_model_instance),
      model_context_(model_context), rate_limiter_config_(rate_limiter_config),
      OnStage_(OnStage), OnRelease_(OnRelease), exec_count_(0),
      state_(AVAILABLE), removal_in_progress_(false)
{
}

void
RateLimiter::ModelInstanceContext::MarkAvailable()
{
  std::lock_guard<std::mutex> lk(state_mtx_);
  state_ = AVAILABLE;
}

Status
RateLimiter::ModelInstanceContext::Stage(StandardScheduleFunc OnSchedule)
{
  {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (state_ != AVAILABLE) {
      return Status(
          Status::Code::INTERNAL,
          "Can not stage a model instance that is not yet available");
    }

    state_ = STAGED;
    OnSchedule_ = OnSchedule;
  }

  OnStage_(this);

  return Status::Success;
}

Status
RateLimiter::ModelInstanceContext::Allocate()
{
  {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (state_ != STAGED) {
      return Status(
          Status::Code::INTERNAL,
          "Can not allocate a model instance that is not yet staged");
    }

    state_ = ALLOCATED;
  }

  OnSchedule_(this);

  return Status::Success;
}

Status
RateLimiter::ModelInstanceContext::DirectAllocate(
    StandardScheduleFunc OnSchedule)
{
  {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (state_ != AVAILABLE) {
      return Status(
          Status::Code::INTERNAL,
          "Can not allocate a model instance that is not yet available");
    }

    state_ = ALLOCATED;
  }

  OnSchedule(this);

  return Status::Success;
}

void
RateLimiter::ModelInstanceContext::Release()
{
  exec_count_++;
  OnRelease_(this);
}

double
RateLimiter::ModelInstanceContext::ScaledPriority()
{
  // TODO: Different schemes for the prioritization of
  // model instance can be added here.
  // The priority of instance is 1 by default. If specified
  // as 0, the priority is still treated as 1.
  auto priority = std::max(rate_limiter_config_.priority(), 1u);
  return (exec_count_ * priority);
}

void
RateLimiter::ModelInstanceContext::RequestRemoval()
{
  std::unique_lock<std::mutex> lk(state_mtx_);
  removal_in_progress_ = true;
}

bool
RateLimiter::ModelInstanceContext::IsRemovalInProgress()
{
  std::unique_lock<std::mutex> lk(state_mtx_);
  return removal_in_progress_;
}


//=========================================================================
//  ResourceManager Implementation
//=========================================================================

Status
RateLimiter::ResourceManager::Create(
    const ResourceMap& resource_map,
    std::unique_ptr<ResourceManager>* resource_manager)
{
  std::unique_ptr<ResourceManager> local_resource_manager(
      new ResourceManager(resource_map));
  *resource_manager = std::move(local_resource_manager);
  return Status::Success;
}

Status
RateLimiter::ResourceManager::AddModelInstance(
    const ModelInstanceContext* instance)
{
  // Add instance into model resources.
  std::lock_guard<std::mutex> lk1(model_resources_mtx_);
  auto pr = model_resources_.emplace(std::make_pair(instance, ResourceMap()));
  for (const auto& resource : instance->GetRateLimiterConfig()->resources()) {
    if (resource.global()) {
      (pr.first->second[GLOBAL_RESOURCE_KEY])[resource.name()] =
          resource.count();
    } else {
      (pr.first->second[instance->RawInstance()->DeviceId()])[resource.name()] =
          resource.count();
    }
  }
  // Increase max resource if needed.
  std::lock_guard<std::mutex> lk2(max_resources_mtx_);
  UpdateMaxResource(pr.first->second);
  return ParseAndValidateResources();
}

Status
RateLimiter::ResourceManager::RemoveModelInstance(
    const ModelInstanceContext* instance)
{
  // Find instance from model resources.
  std::lock_guard<std::mutex> lk1(model_resources_mtx_);
  const auto& itr = model_resources_.find(instance);
  if (itr == model_resources_.end()) {
    return Status(Status::Code::INTERNAL, "Cannot find the instance to remove");
  }
  // Check if max resources need to be updated.
  bool update_needed = false;
  std::lock_guard<std::mutex> lk2(max_resources_mtx_);
  for (const auto& resource_device_map : itr->second) {
    auto ditr = max_resources_.find(resource_device_map.first);
    if (ditr != max_resources_.end()) {
      for (const auto& resource : resource_device_map.second) {
        auto ritr = ditr->second.find(resource.first);
        if (ritr != ditr->second.end() && ritr->second <= resource.second) {
          update_needed = true;
          if (ritr->second < resource.second) {
            LOG_ERROR << "Should not print this! Removing an instance with "
                         "resource above max resource.";
          }
          break;
        }
      }
    }
    if (update_needed) {
      break;
    }
  }
  // Remove instance from model resources.
  model_resources_.erase(instance);
  // Re-compute max resource if needed.
  if (update_needed) {
    ComputeResourceLimits();
  }
  return ParseAndValidateResources();
}

void
RateLimiter::ResourceManager::ComputeResourceLimits()
{
  // Obtain the maximum resource across all the instances and use it as the
  // default available.
  max_resources_.clear();
  for (const auto& instance_resources : model_resources_) {
    UpdateMaxResource(instance_resources.second);
  }
}

void
RateLimiter::ResourceManager::UpdateMaxResource(
    const ResourceMap& instance_resource_map)
{
  for (const auto& resource_device_map : instance_resource_map) {
    auto ditr = max_resources_.find(resource_device_map.first);
    if (ditr == max_resources_.end()) {
      ditr = max_resources_
                 .emplace(resource_device_map.first, resource_device_map.second)
                 .first;
    } else {
      for (const auto& resource : resource_device_map.second) {
        auto ritr = ditr->second.find(resource.first);
        if (ritr == ditr->second.end()) {
          ritr = ditr->second.emplace(resource.first, resource.second).first;
        } else {
          if (ritr->second < resource.second) {
            ritr->second = resource.second;
          }
        }
      }
    }
  }
}

Status
RateLimiter::ResourceManager::ParseAndValidateResources()
{
  if (!explicit_max_resources_.empty()) {
    RETURN_IF_ERROR(ParseAndValidateExplicitResources());
  }
  RETURN_IF_ERROR(ValidateMaxResources());

  if (LOG_VERBOSE_IS_ON(1)) {
    std::string resource_map_str{"\nMax Resource Map===>\n"};
    for (const auto& ditr : max_resources_) {
      if (!ditr.second.empty()) {
        std::string device_str{
            (ditr.first == GLOBAL_RESOURCE_KEY) ? "GLOBAL"
                                                : std::to_string(ditr.first)};
        resource_map_str += "\tDevice: " + device_str + "\n";
        for (const auto& ritr : ditr.second) {
          resource_map_str += "\t\tResource: " + ritr.first +
                              "\t Count: " + std::to_string(ritr.second) + "\n";
        }
      }
    }
    LOG_VERBOSE(1) << resource_map_str;
  }

  return Status::Success;
}

Status
RateLimiter::ResourceManager::ValidateMaxResources()
{
  for (const auto& global_resource : max_resources_[GLOBAL_RESOURCE_KEY]) {
    for (const auto& ditr : max_resources_) {
      if (ditr.first != GLOBAL_RESOURCE_KEY) {
        for (const auto& ritr : ditr.second) {
          if (global_resource.first.compare(ritr.first) == 0) {
            return Status(
                Status::Code::INVALID_ARG,
                (std::string("Resource \"") + ritr.first +
                 "\" is present as both global and device-specific resource in "
                 "the model configuration.")
                    .c_str());
          }
        }
      }
    }
  }
  return Status::Success;
}

Status
RateLimiter::ResourceManager::ParseAndValidateExplicitResources()
{
  for (auto& ditr : max_resources_) {
    for (auto& ritr : ditr.second) {
      // If not specified explicitly, consider the resource to be unavailable.
      size_t resource_count = 0;
      if (ditr.first == GLOBAL_RESOURCE_KEY) {
        // Ignore the device specification... will search for all resources in
        // the map...
        for (const auto& exp_ditr : explicit_max_resources_) {
          for (const auto& exp_ritr : exp_ditr.second) {
            if (ritr.first.compare(exp_ritr.first) == 0) {
              if (resource_count < exp_ritr.second) {
                resource_count = exp_ritr.second;
              }
            }
          }
        }
      } else {
        // Search only for the device specific or per-device resources...
        // device-specific
        for (const auto& exp_ritr : explicit_max_resources_[ditr.first]) {
          if (ritr.first.compare(exp_ritr.first) == 0) {
            if (resource_count < exp_ritr.second) {
              resource_count = exp_ritr.second;
            }
          }
        }
        // per-device
        for (const auto& exp_ritr :
             explicit_max_resources_[PER_DEVICE_RESOURCE_KEY]) {
          if (ritr.first.compare(exp_ritr.first) == 0) {
            if (resource_count < exp_ritr.second) {
              resource_count = exp_ritr.second;
            }
          }
        }
      }
      if (resource_count < ritr.second) {
        return Status(
            Status::Code::INVALID_ARG,
            (std::string("Resource count for \"") + ritr.first +
             "\" is limited to " + std::to_string(resource_count) +
             " which will prevent scheduling of one or more model "
             "instances, the minimum required count is " +
             std::to_string(ritr.second))
                .c_str());
      } else {
        ritr.second = resource_count;
      }
    }
  }

  return Status::Success;
}

bool
RateLimiter::ResourceManager::AllocateResources(
    const ModelInstanceContext* instance)
{
  std::lock_guard<std::mutex> lk1(model_resources_mtx_);
  std::lock_guard<std::mutex> lk2(allocated_resources_mtx_);
  const auto& itr = model_resources_.find(instance);
  if (itr == model_resources_.end()) {
    return false;
  } else {
    // First pass to verify if resources are available
    {
      std::lock_guard<std::mutex> lk3(max_resources_mtx_);
      for (const auto& ditr : itr->second) {
        auto allocated_ditr = allocated_resources_.find(ditr.first);
        if (allocated_ditr == allocated_resources_.end()) {
          allocated_ditr =
              allocated_resources_
                  .emplace(ditr.first, std::map<std::string, size_t>())
                  .first;
        }
        for (const auto& ritr : ditr.second) {
          auto allocated_ritr = allocated_ditr->second.find(ritr.first);
          if (allocated_ritr == allocated_ditr->second.end()) {
            allocated_ritr =
                allocated_ditr->second.emplace(ritr.first, 0).first;
          }
          if ((allocated_ritr->second + ritr.second) >
              (max_resources_[ditr.first])[ritr.first]) {
            return false;
          }
        }
      }
    }

    // Second pass to actually allocate the resources
    for (const auto& ditr : itr->second) {
      for (const auto& ritr : ditr.second) {
        (allocated_resources_[ditr.first])[ritr.first] += ritr.second;
      }
    }
  }

  return true;
}

Status
RateLimiter::ResourceManager::ReleaseResources(
    const ModelInstanceContext* instance)
{
  std::lock_guard<std::mutex> lk1(model_resources_mtx_);
  std::lock_guard<std::mutex> lk2(allocated_resources_mtx_);
  const auto& itr = model_resources_.find(instance);
  if (itr == model_resources_.end()) {
    return Status(
        Status::Code::INTERNAL,
        "Unable find the instance resources to release");
  } else {
    for (const auto& ditr : itr->second) {
      for (const auto& ritr : ditr.second) {
        (allocated_resources_[ditr.first])[ritr.first] -= ritr.second;
      }
    }
  }

  return Status::Success;
}

RateLimiter::ResourceManager::ResourceManager(const ResourceMap& resource_map)
    : explicit_max_resources_(resource_map)
{
}

}}  // namespace triton::core
