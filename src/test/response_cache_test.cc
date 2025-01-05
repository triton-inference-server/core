// Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <numeric>
#include <thread>

#include "cache_manager.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "memory.h"
#include "triton/common/logging.h"

namespace tc = triton::core;

/* Mock classes for Unit Testing */
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

  return Status::Success;
}

//
// InferenceRequest
//
InferenceRequest::InferenceRequest(
    Model* model, const int64_t requested_model_version)
    : needs_normalization_(true), model_raw_(model),
      requested_model_version_(requested_model_version), flags_(0),
      correlation_id_(0), batch_size_(0), timeout_us_(0), collect_stats_(true)
{
  // Unit test doesn't need actual response factory logic
  // or other priority/request_counting logic, it just needs
  // a non-null response factory object.
  response_factory_.reset(new InferenceResponseFactory());
}

InferenceRequest::Input::Input(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count)
    : name_(name), datatype_(datatype),
      original_shape_(shape, shape + dim_count),
      tensor_type_(TensorType::TENSOR), data_(new MemoryReference),
      has_host_policy_specific_data_(false)
{
}

// Use const global var as locals can't be returned in ModelName(),
// and we don't care about the model for the unit test
const std::string MODEL = "model";

const std::string&
InferenceRequest::ModelName() const
{
  return MODEL;
}

int64_t
InferenceRequest::ActualModelVersion() const
{
  // Not using model in unit test mock
  return requested_model_version_;
}

Status
InferenceRequest::PrepareForInference()
{
  // Remove override inputs as those are added during any previous
  // inference execution.
  inputs_.clear();
  override_inputs_.clear();

  // Initially show the actual inputs to be only the original
  // inputs. If overrides are added later they will be added to
  // 'inputs_'.
  for (auto& pr : original_inputs_) {
    inputs_.emplace(std::make_pair(pr.first, std::addressof(pr.second)));
  }

  // Clear the timestamps
  queue_start_ns_ = 0;
#ifdef TRITON_ENABLE_STATS
  request_start_ns_ = 0;
#endif  // TRITON_ENABLE_STATS

  return Status::Success;
}

Status
InferenceRequest::Input::DataBuffer(
    const size_t idx, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id) const
{
  *base = data_->BufferAt(idx, byte_size, memory_type, memory_type_id);

  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count,
    InferenceRequest::Input** input)
{
  const auto& pr = original_inputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple(name, datatype, shape, dim_count));
  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' already exists in request");
  }

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, InferenceRequest::Input** input)
{
  return AddOriginalInput(name, datatype, &shape[0], shape.size(), input);
}

Status
InferenceRequest::Input::AppendData(
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBuffer(
        static_cast<const char*>(base), byte_size, memory_type, memory_type_id);
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
    uint64_t responses_sent, uint64_t infer_start_ns
#endif  // TRITON_ENABLE_METRICS
    )
    : model_(model), id_(id), allocator_(allocator), alloc_userp_(alloc_userp),
      response_fn_(response_fn), response_userp_(response_userp),
      response_delegator_(delegator),
#ifdef TRITON_ENABLE_METRICS
      responses_sent_(responses_sent), infer_start_ns_(infer_start_ns),
#endif  // TRITON_ENABLE_METRICS
      null_response_(false)
{
  // Skip allocator logic / references in unit test
}

std::ostream&
operator<<(std::ostream& out, const InferenceResponse& response)
{
  out << "[0x" << std::addressof(response) << "] "
      << "response id: " << response.Id() << std::endl;

  out << "status:" << response.ResponseStatus().AsString() << std::endl;

  return out;
}

InferenceResponse::Output::~Output()
{
  Status status = ReleaseDataBuffer();
  if (!status.IsOk()) {
    std::cerr << "[ERROR] failed to release buffer for output '" << name_
              << "': " << status.AsString();
  }
}

Status
InferenceResponse::Output::ReleaseDataBuffer()
{
  if (allocated_buffer_ != nullptr) {
    free(allocated_buffer_);
  }

  allocated_buffer_ = nullptr;
  buffer_attributes_.SetByteSize(0);
  buffer_attributes_.SetMemoryType(TRITONSERVER_MEMORY_CPU);
  buffer_attributes_.SetMemoryTypeId(0);
  allocated_userp_ = nullptr;

  return Status::Success;
}

// Same as defined in infer_response.cc
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

// Simplified version of AllocateDataBuffer for CPU memory only
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

  // Simplifications - CPU memory only for now
  if (*memory_type != TRITONSERVER_MEMORY_CPU || *memory_type_id != 0) {
    return Status(
        Status::Code::INTERNAL, "Only standard CPU memory supported for now");
  }

  // Allocate buffer to copy to
  *buffer = malloc(buffer_byte_size);
  if (buffer == nullptr || *buffer == nullptr) {
    return Status(
        Status::Code::INTERNAL, "buffer was nullptr in AllocateDataBuffer");
  }

  // Set relevant member variables for DataBuffer() to return
  allocated_buffer_ = *buffer;
  buffer_attributes_.SetByteSize(buffer_byte_size);
  buffer_attributes_.SetMemoryType(*memory_type);
  buffer_attributes_.SetMemoryTypeId(*memory_type_id);
  allocated_userp_ = nullptr;
  return Status::Success;
}

Status
InferenceResponse::AddOutput(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, InferenceResponse::Output** output)
{
  outputs_.emplace_back(name, datatype, shape, allocator_, alloc_userp_);

  if (output != nullptr) {
    *output = std::addressof(outputs_.back());
  }

  return Status::Success;
}

InferenceRequest::SequenceId::SequenceId()
    : sequence_label_(""), sequence_index_(0),
      id_type_(InferenceRequest::SequenceId::DataType::UINT64)
{
}

InferenceRequest::SequenceId::SequenceId(const std::string& sequence_label)
    : sequence_label_(sequence_label), sequence_index_(0),
      id_type_(InferenceRequest::SequenceId::DataType::STRING)
{
}

InferenceRequest::SequenceId::SequenceId(uint64_t sequence_index)
    : sequence_label_(""), sequence_index_(sequence_index),
      id_type_(InferenceRequest::SequenceId::DataType::UINT64)
{
}

}}  // namespace triton::core


namespace helpers {

// Helpers
void
CheckStatus(tc::Status status)
{
  ASSERT_TRUE(status.IsOk()) << "ERROR: " << status.Message();
}

void
InsertWrapper(
    std::shared_ptr<tc::TritonCache> cache, tc::InferenceResponse* r,
    std::string key)
{
  CheckStatus(cache->Insert(r, key));
}

void
LookupWrapper(
    std::shared_ptr<tc::TritonCache> cache, tc::InferenceResponse* r,
    std::string key)
{
  CheckStatus(cache->Lookup(r, key));
}

void
LookupWrapperMaybeMiss(
    std::shared_ptr<tc::TritonCache> cache, tc::InferenceResponse* r,
    std::string key)
{
  auto status = cache->Lookup(r, key);
  // Success and Cache Miss OK
  auto ok =
      (status.IsOk() || status.StatusCode() == tc::Status::Code::NOT_FOUND);
  ASSERT_TRUE(ok) << "ERROR: " << status.Message();
}

void
reset_response(
    std::unique_ptr<tc::InferenceResponse>* response,
    tc::InferenceRequest* request)
{
  helpers::CheckStatus(request->ResponseFactory()->CreateResponse(response));
}

// Only support 1-Dimensional data to keep it simple
struct Tensor {
  std::string name;
  std::vector<int> data;
};

// Only support 1-Dimensional data to keep it simple
std::unique_ptr<tc::InferenceResponse>
GenerateResponse(
    const tc::InferenceRequest* request, inference::DataType dtype,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    const std::vector<helpers::Tensor>& outputs)
{
  std::cout << "Create response object" << std::endl;
  std::unique_ptr<tc::InferenceResponse> response;
  helpers::CheckStatus(request->ResponseFactory()->CreateResponse(&response));

  std::cout << "Add output metadata to response object" << std::endl;
  for (const auto& tensor : outputs) {
    if (tensor.data.size() == 0) {
      std::cout << "[ERROR] Can't generate a request with no output data"
                << std::endl;
      return nullptr;
    }

    tc::InferenceResponse::Output* response_output = nullptr;
    std::vector<int64_t> shape{1, -1};
    shape[1] = tensor.data.size();
    uint64_t output_size = sizeof(tensor.data[0]) * tensor.data.size();
    helpers::CheckStatus(
        response->AddOutput(tensor.name, dtype, shape, &response_output));

    std::cout << "Allocate output data buffer for response object of size: "
              << output_size << std::endl;
    void* buffer;
    helpers::CheckStatus(response_output->AllocateDataBuffer(
        &buffer, output_size, &memory_type, &memory_type_id));
    if (buffer == nullptr) {
      std::cout << "[ERROR] buffer was nullptr;" << std::endl;
      return nullptr;
    }
    // Copy data from output to response buffer
    std::memcpy(buffer, tensor.data.data(), output_size);
  }

  return response;
}

// Only support 1-Dimensional data to keep it simple
tc::InferenceRequest*
GenerateRequest(
    tc::Model* model, uint64_t model_version, inference::DataType dtype,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    const std::vector<helpers::Tensor>& inputs, const std::string& request_id)
{
  auto request = new tc::InferenceRequest(model, model_version);
  for (const auto& tensor : inputs) {
    if (tensor.data.size() == 0) {
      std::cout << "[ERROR] Can't generate a request with no input data"
                << std::endl;
      return nullptr;
    }

    tc::InferenceRequest::Input* request_input = nullptr;
    std::vector<int64_t> shape{1, -1};
    shape[1] = tensor.data.size();
    request->AddOriginalInput(tensor.name, dtype, shape, &request_input);
    if (request_input == nullptr) {
      std::cout << "[ERROR] request_input was nullptr" << std::endl;
      return nullptr;
    }

    uint64_t input_size = sizeof(tensor.data[0]) * tensor.data.size();
    request_input->AppendData(
        tensor.data.data(), input_size, memory_type, memory_type_id);
  }
  // PrepareForInference for use of ImmutableInputs()
  helpers::CheckStatus(request->PrepareForInference());
  request->SetId(request_id);  // for debugging purposes
  return request;
}

tc::Status
InsertLookupCompare(
    std::shared_ptr<tc::TritonCache> cache,
    std::vector<boost::span<tc::Byte>> expected_buffers, const std::string& key)
{
  if (!cache) {
    return tc::Status(tc::Status::Code::INTERNAL, "cache was nullptr");
  } else if (expected_buffers.empty()) {
    return tc::Status(tc::Status::Code::INTERNAL, "entry was empty");
  }

  helpers::CheckStatus(cache->Insert(expected_buffers, key));
  auto lookup_entry = std::make_unique<tc::CacheEntry>();
  auto status = cache->Lookup(key, lookup_entry.get());
  if (!status.IsOk()) {
    return tc::Status(
        tc::Status::Code::INTERNAL, "Lookup failed: " + status.Message());
  }

  auto lookup_buffers = lookup_entry->Buffers();
  if (lookup_buffers.size() != expected_buffers.size()) {
    return tc::Status(
        tc::Status::Code::INTERNAL,
        "Expected " + std::to_string(expected_buffers.size()) + " got " +
            std::to_string(lookup_buffers.size()));
  }

  for (size_t b = 0; b < expected_buffers.size(); b++) {
    boost::span<tc::Byte> lookup = {
        static_cast<tc::Byte*>(lookup_buffers[b].first),
        lookup_buffers[b].second};
    boost::span<tc::Byte> expected = expected_buffers[b];
    if (!std::equal(
            lookup.begin(), lookup.end(), expected.begin(), expected.end())) {
      return tc::Status(
          tc::Status::Code::INTERNAL,
          "Buffer bytes didn't match for test input");
    }
  }
  return tc::Status::Success;
}

std::shared_ptr<tc::TritonCache>
CreateLocalCache(uint64_t cache_size)
{
  // Create TritonCacheManager
  std::shared_ptr<tc::TritonCacheManager> cache_manager;
  auto cache_dir = "/opt/tritonserver/caches";
  helpers::CheckStatus(
      tc::TritonCacheManager::Create(&cache_manager, cache_dir));

  // Create TritonCache
  std::shared_ptr<tc::TritonCache> cache;
  auto cache_config = R"({"size": )" + std::to_string(cache_size) + "}";
  std::cout << "Creating local cache with config: " << cache_config
            << std::endl;
  auto cache_name = "local";
  helpers::CheckStatus(
      cache_manager->CreateCache(cache_name, cache_config, &cache));

  return cache;
}

std::shared_ptr<tc::TritonCache>
CreateRedisCache(std::string host, std::string port)
{
  // Create TritonCacheManager
  std::shared_ptr<tc::TritonCacheManager> cache_manager;
  auto cache_dir = "/opt/tritonserver/caches";
  helpers::CheckStatus(
      tc::TritonCacheManager::Create(&cache_manager, cache_dir));

  // Create TritonCache
  std::shared_ptr<tc::TritonCache> cache;
  std::ostringstream cache_config_json;
  auto cache_config =
      R"({"host": ")" + host + R"(", "port": ")" + port + R"("})";
  std::cout << "Creating redis cache with config: " << cache_config
            << std::endl;
  auto cache_name = "redis";
  helpers::CheckStatus(
      cache_manager->CreateCache(cache_name, cache_config, &cache));

  return cache;
}

void
CreateCacheExpectFail(
    const std::string& cache_name, const std::string& cache_config)
{
  // Create TritonCacheManager
  std::shared_ptr<tc::TritonCacheManager> cache_manager;
  auto cache_dir = "/opt/tritonserver/caches";
  helpers::CheckStatus(
      tc::TritonCacheManager::Create(&cache_manager, cache_dir));

  // Create TritonCache
  std::shared_ptr<tc::TritonCache> cache;
  auto status = cache_manager->CreateCache(cache_name, cache_config, &cache);

  ASSERT_FALSE(status.IsOk()) << "Creating cache with config: '" << cache_config
                              << "' succeeded when it should fail.";
  ASSERT_EQ(cache, nullptr);
}

}  // namespace helpers

namespace {

// Test Fixture
class RequestResponseCacheTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Sample input data
    data0 = {1, 2, 3, 4};
    data1 = {5, 6, 7, 8};

    // Sample input vectors
    inputs0 = std::vector<helpers::Tensor>{{"input", data0}};
    inputs1 = std::vector<helpers::Tensor>{{"input", data1}};
    inputs2 = std::vector<helpers::Tensor>{{"input", data1}};
    inputs3 =
        std::vector<helpers::Tensor>{{"input0", data0}, {"input1", data1}};
    inputs4 =
        std::vector<helpers::Tensor>{{"input1", data1}, {"input0", data0}};

    // Create three requests with same input name, two with same data, one with
    // different data
    request0 = helpers::GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs0,
        "request0");
    request1 = helpers::GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs1,
        "request1");
    request2 = helpers::GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs2,
        "request2");
    // Create two requests with the same two inputs but inserted in different
    // order
    request3 = helpers::GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs3,
        "request3");
    request4 = helpers::GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs4,
        "request4");
    // Verify requests were created correctly
    ASSERT_NE(request0, nullptr);
    ASSERT_NE(request1, nullptr);
    ASSERT_NE(request2, nullptr);
    ASSERT_NE(request3, nullptr);
    ASSERT_NE(request4, nullptr);

    // Generate a set of unique requests to use for parallelism tests
    for (size_t idx = 0; idx < thread_count; idx++) {
      std::vector<int> data(thread_count, static_cast<int>(idx));
      std::vector<helpers::Tensor> inputs{
          helpers::Tensor{"input" + std::to_string(idx), data}};

      std::string request_id = "unique" + std::to_string(idx);
      auto request = helpers::GenerateRequest(
          model, model_version, dtype, memory_type, memory_type_id, inputs,
          request_id);
      ASSERT_NE(request, nullptr);
      unique_requests.emplace_back(request);
    }
    ASSERT_EQ(unique_requests.size(), thread_count);

    // Sample outputs
    helpers::Tensor output_tensor0 = {"output", data0};
    outputs0 = std::vector<helpers::Tensor>{output_tensor0};
    // Response of 100 ints, taking ~400 bytes at a time
    data100 = std::vector<int>(100, 0);
    std::iota(data100.begin(), data100.end(), 1);
    output100_size = sizeof(int) * data100.size();
    helpers::Tensor output_tensor100 = {"output", data100};
    outputs100 = std::vector<helpers::Tensor>{output_tensor100};

    // Sample responses
    response0 = helpers::GenerateResponse(
        request0, dtype, memory_type, memory_type_id, outputs0);
    ASSERT_NE(response0, nullptr);
    response_400bytes = helpers::GenerateResponse(
        request0, dtype, memory_type, memory_type_id, outputs100);
    ASSERT_NE(response_400bytes, nullptr);

    // Redis cache config
    auto rh = std::getenv("TRITON_REDIS_HOST");
    if (rh) {
      redis_host = rh;
    }
    auto rp = std::getenv("TRITON_REDIS_PORT");
    if (rp) {
      redis_port = rp;
    }
  }

  void TearDown() override
  {
    delete request0;
    delete request1;
    delete request2;
    delete request3;
    delete request4;
    for (auto r : unique_requests) {
      delete r;
    }
  }

 public:
  tc::Model* model = nullptr;
  uint64_t model_version = 1;
  inference::DataType dtype = inference::DataType::TYPE_INT32;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;
  size_t thread_count = 100;
  uint64_t output100_size;
  std::string redis_host = "localhost";
  std::string redis_port = "6379";

  std::vector<int> data0, data1, data100;
  std::vector<helpers::Tensor> inputs0, inputs1, inputs2, inputs3, inputs4,
      inputs100;
  std::vector<helpers::Tensor> outputs0, outputs100;
  tc::InferenceRequest *request0, *request1, *request2, *request3, *request4;
  std::vector<tc::InferenceRequest*> unique_requests;
  std::unique_ptr<tc::InferenceResponse> response0, response_400bytes;
};

// Group common cache tests into namespace for testing multiple implementations
namespace tests {

void
InsertLookupCompareBytes(std::shared_ptr<tc::TritonCache> cache)
{
  // Setup byte buffers
  std::vector<tc::Byte> buffer1{1, tc::Byte{1}};
  std::vector<tc::Byte> buffer2{2, tc::Byte{2}};
  std::vector<tc::Byte> buffer3{4, tc::Byte{4}};
  std::vector<tc::Byte> buffer4{8, tc::Byte{8}};
  std::vector<tc::Byte> buffer5{16, tc::Byte{16}};
  // Setup entry
  std::vector<boost::span<tc::Byte>> entry;
  // Add buffers to entry
  entry.push_back(buffer1);
  entry.push_back(buffer2);
  entry.push_back(buffer3);
  entry.push_back(buffer4);
  entry.push_back(buffer5);

  helpers::CheckStatus(
      helpers::InsertLookupCompare(cache, entry, "TestCacheEntry"));
}

// Hash a collection of unique requests and assert no collisions occurred
void
HashUnique(
    std::shared_ptr<tc::TritonCache> cache,
    std::vector<tc::InferenceRequest*>& unique_requests)
{
  ASSERT_NE(unique_requests.size(), 0);
  std::vector<std::string> hashes;
  for (const auto& request : unique_requests) {
    std::string hash = "";
    helpers::CheckStatus(cache->Hash(*request, &hash));
    ASSERT_NE(hash, "");
    hashes.push_back(hash);
  }
  ASSERT_NE(hashes.size(), 0);

  // Verify no two hashes from the unique requests are the same
  for (size_t i = 0; i < hashes.size(); i++) {
    for (size_t j = 0; j < hashes.size(); j++) {
      if (i == j) {
        continue;
      }
      ASSERT_NE(hashes[i], hashes[j]);
    }
  }
}

// Hash specifically crafted requests to verify their hashes are as expected
void
HashLogic(
    std::shared_ptr<tc::TritonCache> cache, tc::InferenceRequest* request0,
    tc::InferenceRequest* request1, tc::InferenceRequest* request2,
    tc::InferenceRequest* request3, tc::InferenceRequest* request4)
{
  std::string hash0, hash1, hash2, hash3, hash4;
  helpers::CheckStatus(cache->Hash(*request0, &hash0));
  helpers::CheckStatus(cache->Hash(*request1, &hash1));
  helpers::CheckStatus(cache->Hash(*request2, &hash2));
  helpers::CheckStatus(cache->Hash(*request3, &hash3));
  helpers::CheckStatus(cache->Hash(*request4, &hash4));
  // Different input data should have different hashes
  ASSERT_NE(hash0, hash1);
  // Same input data should have same hashes
  ASSERT_EQ(hash1, hash2);
  // Two requests with same two inputs but added in different orders
  ASSERT_EQ(hash3, hash4);
}


void
ParallelInsert(
    std::shared_ptr<tc::TritonCache> cache, size_t thread_count,
    std::unique_ptr<tc::InferenceResponse>& insert_response,
    size_t expected_cache_hits)
{
  // Create threads
  std::vector<std::thread> threads;
  std::cout << "Insert responses into cache with [" << thread_count
            << "] threads in parallel" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    auto key = std::to_string(idx);
    threads.emplace_back(std::thread(
        &helpers::InsertWrapper, cache, insert_response.get(), key));
  }

  // Join threads
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads[idx].join();
  }

  // Lookup each inserted key to verify that expected number remain in cache
  size_t cache_hits = 0;
  size_t cache_misses = 0;
  for (size_t idx = 0; idx < thread_count; idx++) {
    auto key = std::to_string(idx);
    auto entry = std::make_unique<tc::CacheEntry>();
    auto status = cache->Lookup(key, entry.get());
    if (status.IsOk()) {
      cache_hits++;
    } else {
      cache_misses++;
    }
  }
  ASSERT_EQ(cache_hits, expected_cache_hits);
  ASSERT_EQ(cache_hits + cache_misses, thread_count);
}

void
ParallelLookup(
    std::shared_ptr<tc::TritonCache> cache, size_t thread_count,
    std::unique_ptr<tc::InferenceResponse>& insert_response,
    std::vector<tc::InferenceRequest*>& unique_requests,
    std::vector<int> expected_outputs)
{
  const size_t expected_cache_hits = thread_count;
  constexpr size_t expected_cache_misses = 0;

  // Create threads
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<tc::InferenceResponse>> responses;

  // Insert [thread_count] entries into cache sequentially
  for (size_t idx = 0; idx < thread_count; idx++) {
    // Create response for each thread to fill from cache
    std::unique_ptr<tc::InferenceResponse> response;
    helpers::CheckStatus(
        unique_requests[idx]->ResponseFactory()->CreateResponse(&response));
    responses.push_back(std::move(response));
    // Insert response for each thread
    auto key = std::to_string(idx);
    cache->Insert(insert_response.get(), key);
  }

  // Assert all entries were put into cache and no evictions occurred yet
  size_t cache_hits = 0;
  size_t cache_misses = 0;
  for (size_t idx = 0; idx < thread_count; idx++) {
    auto key = std::to_string(idx);
    auto entry = std::make_unique<tc::CacheEntry>();
    auto status = cache->Lookup(key, entry.get());

    if (status.IsOk()) {
      cache_hits++;
    } else {
      std::cout << "ERROR: " << status.Message() << std::endl;
      cache_misses++;
    }
  }
  ASSERT_EQ(cache_hits, expected_cache_hits);
  ASSERT_EQ(cache_misses, expected_cache_misses);
  ASSERT_EQ(cache_hits + cache_misses, thread_count);

  std::cout << "Lookup from cache with [" << thread_count
            << "] threads in parallel" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    auto key = std::to_string(idx);
    threads.emplace_back(
        std::thread(&helpers::LookupWrapper, cache, responses[idx].get(), key));
  }

  // Join threads
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads[idx].join();
  }

  // Grab output from sample response for comparison
  const auto& response0_output = insert_response->Outputs()[0];

  // Verify output results from cache
  for (size_t idx = 0; idx < thread_count; idx++) {
    // Fetch output buffer details
    const void* response_buffer = nullptr;
    size_t response_byte_size = 0;
    TRITONSERVER_MemoryType response_memory_type;
    int64_t response_memory_type_id;
    void* userp;

    // TODO: Handle multiple outputs more generically
    const auto& response_test = responses[idx];
    for (const auto& response_test_output : response_test->Outputs()) {
      ASSERT_EQ(response_test_output.Name(), response0_output.Name());
      ASSERT_EQ(response_test_output.DType(), response0_output.DType());
      ASSERT_EQ(response_test_output.Shape(), response0_output.Shape());
      helpers::CheckStatus(response_test_output.DataBuffer(
          &response_buffer, &response_byte_size, &response_memory_type,
          &response_memory_type_id, &userp));

      // TODO: Use Triton DType to cast buffer and compare outputs generically
      const int* cache_output = static_cast<const int*>(response_buffer);
      for (size_t i = 0; i < response_byte_size / sizeof(int); i++) {
        ASSERT_EQ(cache_output[i], expected_outputs[i]);
      }
    }
  }
}

// Run Inserts/Lookups in parallel to check for race conditions, deadlocks, etc
void
ParallelLookupInsert(
    std::shared_ptr<tc::TritonCache> cache, size_t thread_count,
    std::unique_ptr<tc::InferenceResponse>& insert_response,
    std::vector<tc::InferenceRequest*>& unique_requests)
{
  // Create threads
  std::vector<std::thread> insert_threads;
  std::vector<std::thread> lookup_threads;
  std::vector<std::unique_ptr<tc::InferenceResponse>> responses;

  std::cout << "Create responses" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    // Create response for each thread to fill from cache
    std::unique_ptr<tc::InferenceResponse> response;
    helpers::CheckStatus(
        unique_requests[idx]->ResponseFactory()->CreateResponse(&response));
    responses.push_back(std::move(response));
  }

  // Insert then Lookup [thread_count] entries from cache in parallel
  std::cout << "Insert and Lookup responses into cache with [" << thread_count
            << "] threads in parallel" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    auto key = std::to_string(idx);
    insert_threads.emplace_back(std::thread(
        &helpers::InsertWrapper, cache, insert_response.get(), key));
    lookup_threads.emplace_back(std::thread(
        &helpers::LookupWrapperMaybeMiss, cache, responses[idx].get(), key));
  }

  // Join threads
  for (size_t idx = 0; idx < thread_count; idx++) {
    insert_threads[idx].join();
    lookup_threads[idx].join();
  }
}

void
EndToEnd(
    std::shared_ptr<tc::TritonCache> cache, tc::InferenceRequest* request,
    std::unique_ptr<tc::InferenceResponse>& response,
    const std::vector<helpers::Tensor>& expected_outputs)
{
  std::string key = "";
  helpers::CheckStatus(cache->Hash(*request, &key));
  ASSERT_NE(key, "");

  std::cout << "Lookup request in empty cache" << std::endl;
  auto status = cache->Lookup(nullptr, key);
  // This hash not in cache yet
  ASSERT_FALSE(status.IsOk()) << "hash [" + key + "] should not be in cache";
  // Insertion should succeed
  helpers::CheckStatus(cache->Insert(response.get(), key));

  // Duplicate insertion should fail since request already exists in cache
  status = cache->Insert(response.get(), key);
  // Cache implementations may choose behavior for duplicate insertion
  if (cache->Name() == "redis") {
    ASSERT_TRUE(status.IsOk())
        << "Inserting duplicate item in cache should succeed for redis cache";
  } else {
    ASSERT_FALSE(status.IsOk())
        << "Inserting duplicate item in cache should fail unless "
           "implementation "
        << "explicitly allows it and is specified here.";
  }

  // Create response to test cache lookup
  std::cout << "Create response object into fill from cache" << std::endl;
  std::unique_ptr<tc::InferenceResponse> response_test;
  helpers::CheckStatus(
      request->ResponseFactory()->CreateResponse(&response_test));

  // Lookup should now succeed
  std::cout << "Lookup request in cache after insertion" << std::endl;
  helpers::CheckStatus(cache->Lookup(response_test.get(), key));
  // Grab output from sample response for comparison
  const auto& response0_output = response->Outputs()[0];

  // Fetch output buffer details
  const void* response_buffer = nullptr;
  size_t response_byte_size = 0;
  TRITONSERVER_MemoryType response_memory_type;
  int64_t response_memory_type_id;
  void* userp;
  // TODO: Handle multiple outputs and memory types more generically
  for (const auto& response_test_output : response_test->Outputs()) {
    ASSERT_EQ(response_test_output.Name(), response0_output.Name());
    ASSERT_EQ(response_test_output.DType(), response0_output.DType());
    ASSERT_EQ(response_test_output.Shape(), response0_output.Shape());
    helpers::CheckStatus(response_test_output.DataBuffer(
        &response_buffer, &response_byte_size, &response_memory_type,
        &response_memory_type_id, &userp));
  }

  // TODO: Use Triton DType to cast buffer and compare outputs generically
  const int* cache_output = static_cast<const int*>(response_buffer);
  for (size_t i = 0; i < response_byte_size / sizeof(int); i++) {
    ASSERT_EQ(cache_output[i], expected_outputs[0].data[i]);
  }
}

}  // namespace tests

//
// Local Cache Testing
//
// Currently, cache size and eviction related tests are specific to the local
// cache implementation.
//
// Other tests related to hashing, insertion, and lookups are fairly agnostic
// to the cache implementation.
//

// Test cache size too small to initialize.
TEST_F(RequestResponseCacheTest, TestLocalCacheSizeTooSmall)
{
  // Pick intentionally small cache size, expecting failure
  constexpr uint64_t cache_size = 1;
  auto cache_config = R"({"size": )" + std::to_string(cache_size) + "}";
  std::cout << "Create cache of size: " << cache_size << std::endl;
  helpers::CreateCacheExpectFail("local", cache_config);
}

// Test cache size too large to initialize.
TEST_F(RequestResponseCacheTest, TestLocalCacheSizeTooLarge)
{
  // Pick intentionally large cache size, expecting failure
  constexpr uint64_t cache_size = ULLONG_MAX;
  auto cache_config = R"({"size": )" + std::to_string(cache_size) + "}";
  std::cout << "Create cache of size: " << cache_size << std::endl;
  helpers::CreateCacheExpectFail("local", cache_config);
}

TEST_F(RequestResponseCacheTest, TestLocalCacheSizeSmallerThanEntryBytes)
{
  constexpr uint64_t cache_size = 4 * 1024 * 1024;  // 4 MB, arbitrary
  auto cache = helpers::CreateLocalCache(cache_size);
  ASSERT_NE(cache, nullptr);

  // Setup byte buffer larger than cache size
  std::vector<tc::Byte> large_data(cache_size + 1);
  // Setup entry
  std::vector<boost::span<tc::Byte>> entry;
  entry.push_back(large_data);

  auto status = cache->Insert(entry, "large_bytes");
  // We expect insertion to fail here since cache is too small
  ASSERT_FALSE(status.IsOk())
      << "Inserting item larger than cache succeeded when it should fail";
}

TEST_F(RequestResponseCacheTest, TestLocalCacheSizeSmallerThanEntryResponse)
{
  constexpr uint64_t cache_size = 4 * 1024 * 1024;  // 4 MB, arbitrary
  auto cache = helpers::CreateLocalCache(cache_size);
  ASSERT_NE(cache, nullptr);

  // Set output data to be larger than cache size
  // NOTE: This is not 1 byte larger than cache_size, the cache_size + 1 is to
  // be clear it will always be larger than cache even if the dtype is changed.
  std::vector<int> large_data(cache_size + 1, 0);
  std::cout << "Create large_response (larger than cache) of size: "
            << large_data.size() << std::endl;
  std::vector<helpers::Tensor> large_outputs{
      helpers::Tensor{"output", large_data}};
  auto large_response = helpers::GenerateResponse(
      request0, dtype, memory_type, memory_type_id, large_outputs);

  std::cout << "Insert large_response into cache" << std::endl;
  auto status = cache->Insert(large_response.get(), "large_response");
  // We expect insertion to fail here since cache is too small
  ASSERT_FALSE(status.IsOk())
      << "Inserting item larger than cache succeeded when it should fail";
}

TEST_F(RequestResponseCacheTest, TestLocalCacheEvictionLRU)
{
  // Set size 1200 to hold exactly 2x (400byte + metadata) responses, not 3x
  auto cache = helpers::CreateLocalCache(1200);
  ASSERT_NE(cache, nullptr);
  // Insert 2 responses, expecting both to fit in cache
  helpers::CheckStatus(cache->Insert(response_400bytes.get(), "request0"));
  helpers::CheckStatus(cache->Insert(response_400bytes.get(), "request1"));
  // Validate both responses fit in cache by looking them up
  tc::CacheEntry entry0, entry1, entry2, entry3, entry4, entry5, entry6, entry7;
  auto status = cache->Lookup("request0", &entry0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(cache->Lookup("request1", &entry1).IsOk());
  // Insert a 3rd response, expecting the 1st response to be evicted
  // in LRU order
  helpers::CheckStatus(cache->Insert(response_400bytes.get(), "request2"));
  ASSERT_TRUE(cache->Lookup("request2", &entry2).IsOk());
  ASSERT_FALSE(cache->Lookup("request0", &entry3).IsOk());
  // Lookup 2nd request to bump its LRU order over 3rd
  ASSERT_TRUE(cache->Lookup("request1", &entry4).IsOk());
  // Insert a 4th response, expecting the 3rd to get evicted by LRU order
  // after looking up the 2nd
  helpers::CheckStatus(cache->Insert(response_400bytes.get(), "request3"));
  ASSERT_TRUE(cache->Lookup("request3", &entry5).IsOk());
  ASSERT_TRUE(cache->Lookup("request1", &entry6).IsOk());
  ASSERT_FALSE(cache->Lookup("request2", &entry7).IsOk());
}

TEST_F(RequestResponseCacheTest, TestLocalCacheInsertLookupCompareBytes)
{
  auto cache = helpers::CreateLocalCache(1024);
  ASSERT_NE(cache, nullptr);
  tests::InsertLookupCompareBytes(cache);
}

// This test isn't cache implementation specific since hashing is done
// in Triton core internally for now, but hashing may be exposed to
// implementations in the future.
TEST_F(RequestResponseCacheTest, TestLocalCacheHashing)
{
  auto cache = helpers::CreateLocalCache(1024);
  ASSERT_NE(cache, nullptr);
  tests::HashLogic(cache, request0, request1, request2, request3, request4);
  tests::HashUnique(cache, unique_requests);
}

TEST_F(RequestResponseCacheTest, TestLocalCacheParallelInsert)
{
  // Set size 1200 to hold exactly 2x (400byte + metadata) responses, not 3x
  auto cache = helpers::CreateLocalCache(1200);
  ASSERT_NE(cache, nullptr);
  const size_t expected_cache_hits = 2;
  tests::ParallelInsert(
      cache, thread_count, response_400bytes, expected_cache_hits);
}

TEST_F(RequestResponseCacheTest, TestLocalCacheParallelLookup)
{
  // Set size large enough to hold all responses
  auto cache = helpers::CreateLocalCache(2 * thread_count * output100_size);
  ASSERT_NE(cache, nullptr);
  tests::ParallelLookup(
      cache, thread_count, response_400bytes, unique_requests, data100);
}
TEST_F(RequestResponseCacheTest, TestLocalCacheParallelLookupInsert)
{
  // Set size that can hold a few responses but will certainly
  // run into evictions
  auto cache = helpers::CreateLocalCache(1024);
  ASSERT_NE(cache, nullptr);
  tests::ParallelLookupInsert(
      cache, thread_count, response_400bytes, unique_requests);
}

TEST_F(RequestResponseCacheTest, TestLocalCacheEndToEnd)
{
  auto cache = helpers::CreateLocalCache(8 * 1024 * 1024);
  ASSERT_NE(cache, nullptr);
  tests::EndToEnd(cache, request0, response0, outputs0);
}


//
// Redis Cache Testing
//
// The following tests are fairly agnostic to cache implementation,
// there are no tests around specific Redis settings or eviction
// policies at this time, and instead tests Redis's default settings.
//
// NOTE: These tests require a Redis server to already be running and
// accessible. There is an assumed host:port of localhost:6379 for testing
// purposes, but these can be configured via TRITON_REDIS_HOST and
// TRITON_REDIS_PORT env vars.
//

TEST_F(RequestResponseCacheTest, TestRedisCacheInsertLookupCompareBytes)
{
  auto cache = helpers::CreateRedisCache(redis_host, redis_port);
  ASSERT_NE(cache, nullptr);
  tests::InsertLookupCompareBytes(cache);
}
// This test isn't cache implementation specific since hashing is done
// in Triton core internally for now, but hashing may be exposed to
// implementations in the future.
TEST_F(RequestResponseCacheTest, TestRedisCacheHashing)
{
  auto cache = helpers::CreateRedisCache(redis_host, redis_port);
  ASSERT_NE(cache, nullptr);
  tests::HashLogic(cache, request0, request1, request2, request3, request4);
  tests::HashUnique(cache, unique_requests);
}


TEST_F(RequestResponseCacheTest, TestRedisCacheParallelInsert)
{
  auto cache = helpers::CreateRedisCache(redis_host, redis_port);
  ASSERT_NE(cache, nullptr);
  // Don't expect any cache misses from Redis by default.
  // Future tests can set a fixed size and eviction policy on Redis.
  // For now, no eviction policy testing is done on Redis cache.
  const size_t expected_cache_hits = thread_count;
  tests::ParallelInsert(
      cache, thread_count, response_400bytes, expected_cache_hits);
}

TEST_F(RequestResponseCacheTest, TestRedisCacheParallelLookup)
{
  auto cache = helpers::CreateRedisCache(redis_host, redis_port);
  ASSERT_NE(cache, nullptr);
  tests::ParallelLookup(
      cache, thread_count, response_400bytes, unique_requests, data100);
}

TEST_F(RequestResponseCacheTest, TestRedisCacheParallelLookupInsert)
{
  auto cache = helpers::CreateRedisCache(redis_host, redis_port);
  ASSERT_NE(cache, nullptr);
  tests::ParallelLookupInsert(
      cache, thread_count, response_400bytes, unique_requests);
}

TEST_F(RequestResponseCacheTest, TestRedisCacheEndToEnd)
{
  auto cache = helpers::CreateRedisCache(redis_host, redis_port);
  ASSERT_NE(cache, nullptr);
  tests::EndToEnd(cache, request0, response0, outputs0);
}

}  // namespace

int
main(int argc, char** argv)
{
#ifdef TRITON_ENABLE_LOGGING
  LOG_SET_VERBOSE(2);
#endif  // TRITON_ENABLE_LOGGING

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
