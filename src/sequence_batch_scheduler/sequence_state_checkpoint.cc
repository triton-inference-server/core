// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sequence_state_checkpoint.h"

#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>

#include "model_config_utils.h"
#include "sequence_batch_scheduler.h"
#include "server.h"

namespace triton { namespace core {
namespace {
constexpr std::array<char, 8> kMagic{'T', 'R', 'I', 'T', 'O', 'N', 'S', 'C'};
constexpr size_t kDigestSize = 32;
constexpr size_t kHeaderSize = kMagic.size() + kDigestSize + sizeof(uint64_t) +
                               sizeof(uint64_t) + kDigestSize;

Status
Invalid(const std::string& text)
{
  return Status(
      Status::Code::INVALID_ARG, "sequence state checkpoint: " + text);
}

Status
IOError(const std::string& text)
{
  return Status(Status::Code::INTERNAL, "sequence state checkpoint: " + text);
}

struct Fd {
  int fd{-1};
  explicit Fd(int f) : fd(f) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd()
  {
    if (fd >= 0)
      close(fd);
  }
};

bool
Sync(int fd)
{
  while (fsync(fd) != 0) {
    if (errno != EINTR)
      return false;
  }
  return true;
}

bool
Remove(const std::string& path)
{
  while (unlink(path.c_str()) != 0) {
    if (errno != EINTR)
      return false;
  }
  return true;
}

bool
Publish(const std::string& temporary_path, const std::string& final_path)
{
  while (link(temporary_path.c_str(), final_path.c_str()) != 0) {
    if (errno != EINTR)
      return false;
  }
  return true;
}

Status
SyncParentDirectory(const std::string& path)
{
  const auto parent = std::filesystem::path(path).parent_path();
  Fd dir(open(
      (parent.empty() ? "." : parent.c_str()),
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if ((dir.fd < 0) || !Sync(dir.fd))
    return IOError("failed to synchronize parent directory");
  return Status::Success;
}

struct Hash {
  EVP_MD_CTX* ctx{nullptr};
  Hash()
  {
    ctx = EVP_MD_CTX_new();
    if (ctx == nullptr)
      throw std::runtime_error("SHA256 init failed");
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
      EVP_MD_CTX_free(ctx);
      ctx = nullptr;
      throw std::runtime_error("SHA256 init failed");
    }
  }
  Hash(const Hash&) = delete;
  Hash& operator=(const Hash&) = delete;
  ~Hash() { EVP_MD_CTX_free(ctx); }
  void Add(const void* p, size_t n)
  {
    if ((n != 0) && (EVP_DigestUpdate(ctx, p, n) != 1))
      throw std::runtime_error("SHA256 update failed");
  }
  SequenceStateFingerprint Finish()
  {
    SequenceStateFingerprint d{};
    unsigned n;
    if (EVP_DigestFinal_ex(ctx, d.data(), &n) != 1 || n != d.size())
      throw std::runtime_error("SHA256 final failed");
    return d;
  }
};
void
AppendU64(std::vector<char>* data, uint64_t value)
{
  for (size_t byte = 0; byte < sizeof(value); ++byte) {
    data->push_back(static_cast<char>(value >> (8 * byte)));
  }
}

uint64_t
ReadU64(const char* data)
{
  uint64_t value = 0;
  for (size_t byte = 0; byte < sizeof(value); ++byte) {
    value |= uint64_t(static_cast<unsigned char>(data[byte])) << (8 * byte);
  }
  return value;
}

bool
WriteAll(int fd, const char* data, size_t byte_size)
{
  while (byte_size != 0) {
    const ssize_t written = write(fd, data, byte_size);
    if ((written < 0) && (errno == EINTR)) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    data += written;
    byte_size -= written;
  }
  return true;
}

bool
ReadAll(int fd, char* data, size_t byte_size)
{
  while (byte_size != 0) {
    const ssize_t read_size = read(fd, data, byte_size);
    if ((read_size < 0) && (errno == EINTR)) {
      continue;
    }
    if (read_size <= 0) {
      return false;
    }
    data += read_size;
    byte_size -= read_size;
  }
  return true;
}

Status
CollectInputState(
    const std::shared_ptr<SequenceStates>& states, size_t expected_byte_size,
    std::vector<char>* payload)
{
  if (!states)
    return Invalid("state is not initialized");
  if (states.use_count() != 1)
    return Invalid("state is still referenced by a completed request");
  std::map<const Memory*, size_t> owners;
  for (const auto& entry : states->InputStates()) {
    if (!entry.second->Data())
      return Invalid("state buffer is unavailable");
    ++owners[entry.second->Data().get()];
  }
  for (const auto& entry : states->OutputStates()) {
    if (!entry.second->Data())
      return Invalid("state buffer is unavailable");
    ++owners[entry.second->Data().get()];
  }
  for (const auto& entry : states->InputStates()) {
    const auto& memory = entry.second->Data();
    if (static_cast<size_t>(memory.use_count()) != owners[memory.get()])
      return Invalid("state buffer is still referenced by a completed request");
  }
  for (const auto& entry : states->OutputStates()) {
    const auto& memory = entry.second->Data();
    if (static_cast<size_t>(memory.use_count()) != owners[memory.get()])
      return Invalid("state buffer is still referenced by a completed request");
  }
  size_t byte_size = 0;
  for (const auto& [name, s] : states->InputStates()) {
    const size_t state_size = s->Data()->TotalByteSize();
    if (state_size > (expected_byte_size - byte_size))
      return Invalid("state layout changed");
    byte_size += state_size;
  }
  if (byte_size != expected_byte_size)
    return Invalid("state layout changed");

  payload->clear();
  payload->reserve(expected_byte_size);
  for (const auto& [name, s] : states->InputStates()) {
    const auto& memory = s->Data();
    for (size_t index = 0; index < memory->BufferCount(); ++index) {
      size_t buffer_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t device;
      const char* buffer =
          memory->BufferAt(index, &buffer_size, &memory_type, &device);
      if ((buffer == nullptr) || (memory_type != TRITONSERVER_MEMORY_CPU) ||
          (buffer_size > (expected_byte_size - payload->size())))
        return Invalid("state is not fixed CPU memory");
      payload->insert(payload->end(), buffer, buffer + buffer_size);
    }
  }
  if (payload->size() != expected_byte_size)
    return Invalid("state layout changed");
  return Status::Success;
}
}  // namespace

Status
SequenceStateModelFingerprint(
    const std::string& model_name, int64_t version,
    const std::string& state_layout, SequenceStateFingerprint* result)
{
  try {
    Hash h;
    constexpr char format[] = "triton-sequence-state-checkpoint-v1";
    h.Add(format, sizeof(format));
    h.Add(model_name.c_str(), model_name.size() + 1);
    h.Add(&version, sizeof(version));
    h.Add(state_layout.c_str(), state_layout.size() + 1);
    *result = h.Finish();
    return Status::Success;
  }
  catch (const std::exception& e) {
    return IOError(e.what());
  }
}

Status
CaptureSequenceState(
    const std::shared_ptr<SequenceStates>& states, size_t expected_byte_size,
    std::vector<char>* payload)
{
  return CollectInputState(states, expected_byte_size, payload);
}

Status
WriteSequenceStateCheckpoint(
    const std::string& path, uint64_t id,
    const SequenceStateFingerprint& fingerprint,
    const std::vector<char>& payload)
{
  std::vector<char> header(kMagic.begin(), kMagic.end());
  header.insert(header.end(), fingerprint.begin(), fingerprint.end());
  AppendU64(&header, id);
  AppendU64(&header, payload.size());
  Hash h;
  h.Add(header.data(), header.size());
  h.Add(payload.data(), payload.size());
  const auto digest = h.Finish();
  header.insert(header.end(), digest.begin(), digest.end());

  std::string tmp = path + ".tmp.XXXXXX";
  Fd fd(mkostemp(tmp.data(), O_CLOEXEC));
  if (fd.fd < 0)
    return IOError("failed to create temporary file");
  auto fail = [&]() {
    Remove(tmp);
    return IOError("failed to persist checkpoint; session remains active");
  };
  if (!WriteAll(fd.fd, header.data(), header.size()) ||
      !WriteAll(fd.fd, payload.data(), payload.size()) || !Sync(fd.fd))
    return fail();
  if (!Publish(tmp, path))
    return fail();
  if (!Remove(tmp) || !SyncParentDirectory(path).IsOk()) {
    Remove(path);
    SyncParentDirectory(path);
    return IOError("failed to publish checkpoint; session remains active");
  }
  return Status::Success;
}

Status
ReadSequenceStateCheckpoint(
    const std::string& path, uint64_t id,
    const SequenceStateFingerprint& fingerprint, size_t expected_byte_size,
    std::vector<char>* payload)
{
  Fd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
  struct stat st {};
  if (fd.fd < 0 || fstat(fd.fd, &st) != 0 || !S_ISREG(st.st_mode) ||
      st.st_size != static_cast<off_t>(kHeaderSize + expected_byte_size))
    return Invalid("invalid checkpoint file or size");

  std::array<char, kHeaderSize> header;
  if (!ReadAll(fd.fd, header.data(), header.size()))
    return Invalid("checkpoint is truncated");
  payload->resize(expected_byte_size);
  if (!ReadAll(fd.fd, payload->data(), payload->size()))
    return Invalid("checkpoint is truncated");
  char extra;
  ssize_t extra_bytes;
  do {
    extra_bytes = read(fd.fd, &extra, 1);
  } while ((extra_bytes < 0) && (errno == EINTR));
  if (extra_bytes != 0)
    return Invalid("checkpoint changed while reading");
  if (memcmp(header.data(), kMagic.data(), kMagic.size()) ||
      memcmp(header.data() + kMagic.size(), fingerprint.data(), kDigestSize) ||
      ReadU64(header.data() + kMagic.size() + kDigestSize) != id ||
      ReadU64(header.data() + kMagic.size() + kDigestSize + sizeof(uint64_t)) !=
          expected_byte_size)
    return Invalid("checkpoint does not match the model or sequence");
  Hash h;
  h.Add(header.data(), kHeaderSize - kDigestSize);
  h.Add(payload->data(), payload->size());
  const auto digest = h.Finish();
  if (memcmp(
          digest.data(), header.data() + kHeaderSize - kDigestSize,
          kDigestSize))
    return Invalid("checkpoint checksum mismatch");
  return Status::Success;
}

Status
DiscardSequenceStateCheckpoint(
    const std::string& path, uint64_t id,
    const SequenceStateFingerprint& fingerprint, size_t expected_byte_size)
{
  std::vector<char> payload;
  RETURN_IF_ERROR(ReadSequenceStateCheckpoint(
      path, id, fingerprint, expected_byte_size, &payload));
  if (!Remove(path))
    return IOError("failed to remove checkpoint");
  return SyncParentDirectory(path);
}

Status
RestoreSequenceStateCheckpoint(
    const std::vector<char>& payload,
    const std::shared_ptr<SequenceStates>& states)
{
  size_t expected = 0;
  for (const auto& [name, s] : states->InputStates())
    expected += s->Data()->TotalByteSize();
  if (expected != payload.size())
    return Invalid("checkpoint tensor layout mismatch");
  size_t offset = 0;
  for (const auto& [name, s] : states->InputStates()) {
    // Initialize() always creates AllocatedMemory in this CPU-only scope.
    auto* data = dynamic_cast<AllocatedMemory*>(s->Data().get());
    TRITONSERVER_MemoryType t;
    int64_t device;
    if (data == nullptr)
      return Invalid("restore allocation type is unsupported");
    char* p = data->MutableBuffer(&t, &device);
    size_t n = data->TotalByteSize();
    if (!p || t != TRITONSERVER_MEMORY_CPU)
      return Invalid("restore allocation failed");
    memcpy(p, payload.data() + offset, n);
    offset += n;
  }
  return Status::Success;
}

Status
SequenceBatchScheduler::ConfigureCheckpointing(
    const inference::ModelConfig& config,
    const std::vector<std::shared_ptr<TritonModelInstance>>& instances)
{
  const auto parameter =
      config.parameters().find("sequence_state_checkpointing");
  if (parameter == config.parameters().end())
    return Status::Success;
  RETURN_IF_ERROR(ParseBoolParameter(
      "sequence_state_checkpointing", parameter->second.string_value(),
      &checkpoint_enabled_));
  if (!checkpoint_enabled_)
    return Status::Success;

  if ((instances.size() != 1) ||
      (instances.front()->Kind() != TRITONSERVER_INSTANCEGROUPKIND_CPU) ||
      config.sequence_batching().has_oldest() ||
      config.sequence_batching().iterative_sequence() ||
      config.model_transaction_policy().decoupled() ||
      config.sequence_batching().state().empty())
    return Status(
        Status::Code::INVALID_ARG,
        "sequence state checkpointing requires one CPU model instance, "
        "direct non-iterative non-decoupled batching and implicit state");

  std::vector<const inference::ModelSequenceBatching_State*> states;
  for (const auto& state : config.sequence_batching().state()) {
    if ((triton::common::GetDataTypeByteSize(state.data_type()) <= 0) ||
        state.use_growable_memory())
      return Status(
          Status::Code::INVALID_ARG,
          "sequence state checkpointing requires fixed numeric state");

    int64_t byte_size = 0;
    RETURN_IF_ERROR(GetByteSize(
        state.data_type(), state.dims(), state.input_name(), &byte_size));
    if ((byte_size < 0) ||
        (static_cast<uint64_t>(byte_size) >
         std::numeric_limits<size_t>::max() - checkpoint_state_byte_size_))
      return Status(
          Status::Code::INVALID_ARG,
          "sequence state checkpoint size exceeds the host address space");
    checkpoint_state_byte_size_ += byte_size;
    states.push_back(&state);
  }

  std::sort(states.begin(), states.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->input_name() < rhs->input_name();
  });
  for (const auto* state : states) {
    const std::string serialized = state->SerializeAsString();
    const size_t byte_size = serialized.size();
    checkpoint_state_layout_.append(
        reinterpret_cast<const char*>(&byte_size), sizeof(byte_size));
    checkpoint_state_layout_.append(serialized);
  }
  return SequenceStateModelFingerprint(
      model_->Name(), model_->Version(), checkpoint_state_layout_,
      &checkpoint_fingerprint_);
}

Status
DirectSequenceBatch::RunCheckpointCommand(
    int32_t drain_slot, std::function<Status()> run)
{
  auto command = std::make_shared<CheckpointCommand>();
  command->drain_slot = drain_slot;
  command->run = std::move(run);
  auto result = command->result.get_future();
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (checkpoint_command_)
      return Status(
          Status::Code::ALREADY_EXISTS,
          "sequence state checkpoint operation already pending");
    checkpoint_command_ = std::move(command);
  }
  cv_.notify_one();
  return result.get();
}

void
DirectSequenceBatch::CheckpointBoundary()
{
  for (const int32_t slot : clear_state_after_execution_)
    sequence_states_[slot].reset();
  clear_state_after_execution_.clear();

  if (!checkpoint_command_)
    return;
  if ((checkpoint_command_->drain_slot >= 0) &&
      !queues_[checkpoint_command_->drain_slot].empty())
    return;

  auto command = std::move(checkpoint_command_);
  try {
    command->result.set_value(command->run());
  }
  catch (const std::exception& e) {
    command->result.set_value(Status(Status::Code::INTERNAL, e.what()));
  }
}

Status
SequenceBatchScheduler::SequenceStateCheckpoint(
    TRITONSERVER_SequenceStateCheckpointOperation operation, uint64_t id,
    const std::string& path, uint64_t* occupied, uint64_t* state_bytes)
{
  if ((operation < TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_SUSPEND) ||
      (operation > TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_STATUS))
    return Status(Status::Code::INVALID_ARG, "invalid checkpoint operation");
  if (!checkpoint_enabled_)
    return Status(
        Status::Code::UNSUPPORTED,
        "sequence state checkpointing is not enabled for this model");
  if ((operation != TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_STATUS) &&
      ((id == 0) || path.empty()))
    return Status(
        Status::Code::INVALID_ARG,
        "checkpoint operations require a sequence ID and path");
  if ((batchers_.size() != 1) ||
      model_->Config().sequence_batching().has_oldest())
    return Status(
        Status::Code::UNSUPPORTED,
        "sequence state checkpointing requires one direct batcher");

  std::lock_guard<std::mutex> control(checkpoint_control_mu_);
  *occupied = 0;
  *state_bytes = 0;

  auto* batch =
      dynamic_cast<DirectSequenceBatch*>(batchers_.begin()->second.get());
  if (batch == nullptr)
    return Status(
        Status::Code::INTERNAL, "direct sequence batcher unavailable");

  auto collect = [&]() {
    *occupied = 0;
    *state_bytes = 0;
    std::set<const Memory*> allocations;
    for (size_t slot = 0; slot < batch->sequence_states_.size(); ++slot) {
      if (batch->seq_slot_correlation_ids_[slot].InSequence())
        ++*occupied;
      const auto& states = batch->sequence_states_[slot];
      if (!states)
        continue;
      for (auto* state_map :
           {&states->InputStates(),
            static_cast<
                const std::map<std::string, std::shared_ptr<SequenceState>>*>(
                &states->OutputStates())}) {
        for (const auto& [name, state] : *state_map) {
          const auto& data = state->Data();
          if (data && allocations.insert(data.get()).second)
            *state_bytes += data->TotalByteSize();
        }
      }
    }
    return Status::Success;
  };
  auto collect_at_boundary = [&]() {
    return batch->RunCheckpointCommand(-1, collect);
  };

  if (operation == TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_STATUS)
    return collect_at_boundary();

  const InferenceRequest::SequenceId correlation_id(id);

  if (operation == TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_DISCARD) {
    RETURN_IF_ERROR(DiscardSequenceStateCheckpoint(
        path, id, checkpoint_fingerprint_, checkpoint_state_byte_size_));
    {
      std::lock_guard<std::mutex> lock(mu_);
      checkpoint_suspended_.erase(correlation_id);
    }
    return collect_at_boundary();
  }

  std::vector<char> payload;
  if (operation == TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_RESUME) {
    RETURN_IF_ERROR(ReadSequenceStateCheckpoint(
        path, id, checkpoint_fingerprint_, checkpoint_state_byte_size_,
        &payload));
  }

  int32_t slot = -1;
  {
    std::lock_guard<std::mutex> admission(checkpoint_admission_mu_);
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_)
      return Status(Status::Code::UNAVAILABLE, "model is stopping");

    const auto active = sequence_to_batcherseqslot_map_.find(correlation_id);
    if (operation == TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_SUSPEND) {
      if (active == sequence_to_batcherseqslot_map_.end())
        return Status(
            Status::Code::INVALID_ARG,
            "suspend requires a sequence with an assigned slot");
      slot = active->second.seq_slot_;
    } else {
      if ((active != sequence_to_batcherseqslot_map_.end()) ||
          sequence_to_backlog_map_.count(correlation_id))
        return Status(
            Status::Code::ALREADY_EXISTS,
            "sequence is already active or backlogged");
      if (ready_batcher_seq_slots_.empty())
        return Status(
            Status::Code::UNAVAILABLE,
            "no sequence slot is available for restore");
      const auto target = ready_batcher_seq_slots_.top();
      ready_batcher_seq_slots_.pop();
      slot = target.seq_slot_;
      sequence_to_batcherseqslot_map_[correlation_id] = target;
    }
    checkpoint_closed_.insert(correlation_id);
  }

  auto open_admission = [&]() {
    std::lock_guard<std::mutex> lock(mu_);
    checkpoint_closed_.erase(correlation_id);
  };

  if (operation == TRITONSERVER_SEQUENCE_STATE_CHECKPOINT_SUSPEND) {
    Status status = batch->RunCheckpointCommand(slot, [&, slot]() {
      if (batch->seq_slot_correlation_ids_[slot] != correlation_id)
        return Status(
            Status::Code::INVALID_ARG,
            "sequence ended while suspend was draining");
      return CaptureSequenceState(
          batch->sequence_states_[slot], checkpoint_state_byte_size_, &payload);
    });
    if (!status.IsOk()) {
      open_admission();
      return status;
    }

    status = WriteSequenceStateCheckpoint(
        path, id, checkpoint_fingerprint_, payload);
    if (!status.IsOk()) {
      open_admission();
      return status;
    }

    status = batch->RunCheckpointCommand(slot, [&, slot]() {
      if (batch->seq_slot_correlation_ids_[slot] != correlation_id)
        return Status(
            Status::Code::INVALID_ARG,
            "sequence ended before suspend completed");
      batch->sequence_states_[slot].reset();
      {
        std::lock_guard<std::mutex> lock(mu_);
        sequence_to_batcherseqslot_map_.erase(correlation_id);
        correlation_id_timestamps_.erase(correlation_id);
        checkpoint_restored_.erase(correlation_id);
        checkpoint_suspended_.insert(correlation_id);
        checkpoint_closed_.erase(correlation_id);
      }
      batch->seq_slot_correlation_ids_[slot] = ReleaseSequenceSlot(
          BatcherSequenceSlot(batch->model_instance_, slot),
          &batch->queues_[slot]);
      return collect();
    });
    if (!status.IsOk())
      open_admission();
    return status;
  }

  Status status = batch->RunCheckpointCommand(-1, [&, slot]() {
    auto states = std::make_shared<SequenceStates>();
    Status restore_status = states->Initialize(
        StateOutputConfigMap(), MaxBatchSize(), InitialState(),
        batch->model_instance_->Kind(), batch->model_instance_->DeviceId(),
        batch->model_instance_->Model()
            ->Server()
            ->CudaVirtualAddressSpaceSize());
    if (restore_status.IsOk())
      restore_status = RestoreSequenceStateCheckpoint(payload, states);
    if (restore_status.IsOk()) {
      batch->sequence_states_[slot] = std::move(states);
      batch->seq_slot_correlation_ids_[slot] = correlation_id;
      batch->max_active_seq_slot_ = std::max(slot, batch->max_active_seq_slot_);
      const uint64_t now =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      std::lock_guard<std::mutex> lock(mu_);
      correlation_id_timestamps_[correlation_id] = now;
      checkpoint_restored_.insert(correlation_id);
      checkpoint_suspended_.erase(correlation_id);
      checkpoint_closed_.erase(correlation_id);
    } else {
      {
        std::lock_guard<std::mutex> lock(mu_);
        sequence_to_batcherseqslot_map_.erase(correlation_id);
        checkpoint_closed_.erase(correlation_id);
      }
      batch->seq_slot_correlation_ids_[slot] = ReleaseSequenceSlot(
          BatcherSequenceSlot(batch->model_instance_, slot),
          &batch->queues_[slot]);
    }
    collect();
    return restore_status;
  });
  if (!status.IsOk())
    open_admission();
  return status;
}

}}  // namespace triton::core
