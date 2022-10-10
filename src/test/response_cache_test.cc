// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gtest/gtest.h"

#include <thread>
#include "memory.h"
#include "response_cache.h"
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
      response_delegator_));

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
  // a non-null reponse factory object.
  response_factory_.reset(new InferenceResponseFactory());
}

InferenceRequest::Input::Input(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count)
    : name_(name), datatype_(datatype),
      original_shape_(shape, shape + dim_count), is_shape_tensor_(false),
      data_(new MemoryReference), has_host_policy_specific_data_(false)
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
        void(std::unique_ptr<InferenceResponse>&&, const uint32_t)>& delegator)
    : model_(model), id_(id), allocator_(allocator), alloc_userp_(alloc_userp),
      response_fn_(response_fn), response_userp_(response_userp),
      response_delegator_(delegator), null_response_(false)
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


namespace {

// Helpers
void
check_status(tc::Status status)
{
  ASSERT_TRUE(status.IsOk()) << "ERROR: " << status.Message();
}

void
cache_stats(std::unique_ptr<tc::RequestResponseCache>& cache)
{
  std::cout << "Cache entries: " << cache->NumEntries() << std::endl;
  std::cout << "Cache evictions: " << cache->NumEvictions() << std::endl;
  std::cout << "Cache free bytes: " << cache->FreeBytes() << std::endl;
  std::cout << "Cache alloc'd bytes: " << cache->AllocatedBytes() << std::endl;
  std::cout << "Cache total bytes: " << cache->TotalBytes() << std::endl;
}

void
reset_response(
    std::unique_ptr<tc::InferenceResponse>* response,
    tc::InferenceRequest* request)
{
  check_status(request->ResponseFactory()->CreateResponse(response));
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
    const std::vector<Tensor>& outputs)
{
  std::cout << "Create response object" << std::endl;
  std::unique_ptr<tc::InferenceResponse> response;
  check_status(request->ResponseFactory()->CreateResponse(&response));

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
    std::cout << "Output size bytes: " << output_size << std::endl;
    check_status(
        response->AddOutput(tensor.name, dtype, shape, &response_output));

    std::cout << "Allocate output data buffer for response object" << std::endl;
    void* buffer;
    check_status(response_output->AllocateDataBuffer(
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
    const std::vector<Tensor>& inputs, const std::string& request_id)
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
  check_status(request->PrepareForInference());
  request->SetId(request_id);  // for debugging purposes
  return request;
}

// Test Fixture
class RequestResponseCacheTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Sample input data
    data0 = {1, 2, 3, 4};
    data1 = {5, 6, 7, 8};

    // Sample input vectors
    inputs0 = std::vector<Tensor>{{"input", data0}};
    inputs1 = std::vector<Tensor>{{"input", data1}};
    inputs2 = std::vector<Tensor>{{"input", data1}};
    inputs3 = std::vector<Tensor>{{"input0", data0}, {"input1", data1}};
    inputs4 = std::vector<Tensor>{{"input1", data1}, {"input0", data0}};

    // Create three requests with same input name, two with same data, one with
    // different data
    request0 = GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs0,
        "request0");
    request1 = GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs1,
        "request1");
    request2 = GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs2,
        "request2");
    // Create two requests with the same two inputs but inserted in different
    // order
    request3 = GenerateRequest(
        model, model_version, dtype, memory_type, memory_type_id, inputs3,
        "request3");
    request4 = GenerateRequest(
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
      std::vector<Tensor> inputs{Tensor{"input" + std::to_string(idx), data}};

      std::string request_id = "unique" + std::to_string(idx);
      std::cout << "Generating request: " << request_id << std::endl;
      auto request = GenerateRequest(
          model, model_version, dtype, memory_type, memory_type_id, inputs,
          request_id);
      ASSERT_NE(request, nullptr);
      unique_requests.emplace_back(request);
    }
    ASSERT_EQ(unique_requests.size(), thread_count);

    // Sample outputs
    Tensor output_tensor0 = {"output", data0};
    output0_size = sizeof(int) * data0.size();
    outputs0 = std::vector<Tensor>{output_tensor0};
    // Response of 100 ints, taking ~400 bytes at a time
    data100 = std::vector<int>(100, 0);
    Tensor output_tensor100 = {"output", data100};
    outputs100 = std::vector<Tensor>{output_tensor100};

    // Sample responses
    response0 = GenerateResponse(
        request0, dtype, memory_type, memory_type_id, outputs0);
    ASSERT_NE(response0, nullptr);
    response_400bytes = GenerateResponse(
        request0, dtype, memory_type, memory_type_id, outputs100);
    ASSERT_NE(response_400bytes, nullptr);
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
  size_t thread_count = 10;
  uint64_t output0_size;

  std::vector<int> data0, data1, data100;
  std::vector<Tensor> inputs0, inputs1, inputs2, inputs3, inputs4, inputs100;
  std::vector<Tensor> outputs0, outputs100;
  tc::InferenceRequest *request0, *request1, *request2, *request3, *request4;
  std::vector<tc::InferenceRequest*> unique_requests;
  std::unique_ptr<tc::InferenceResponse> response0, response_400bytes;
};

// Test hashing for consistency on same request
TEST_F(RequestResponseCacheTest, TestHashing)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 4 * 1024 * 1024;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);

  // Compare hashes
  std::cout << "Compare hashes" << std::endl;
  check_status(cache->HashAndSet(request0));
  check_status(cache->HashAndSet(request1));
  check_status(cache->HashAndSet(request2));
  check_status(cache->HashAndSet(request3));
  check_status(cache->HashAndSet(request4));

  std::cout << "request0->CacheKey(): " << request0->CacheKey() << std::endl;
  std::cout << "request1->CacheKey(): " << request1->CacheKey() << std::endl;
  std::cout << "request2->CacheKey(): " << request2->CacheKey() << std::endl;
  std::cout << "request3->CacheKey(): " << request3->CacheKey() << std::endl;
  std::cout << "request4->CacheKey(): " << request4->CacheKey() << std::endl;
  // Different input data should have different hashes
  ASSERT_NE(request0->CacheKey(), request1->CacheKey());
  // Same input data should have same hashes
  ASSERT_EQ(request1->CacheKey(), request2->CacheKey());
  // Two requests with same two inputs but added in different orders
  ASSERT_EQ(request3->CacheKey(), request4->CacheKey());
}


// Test cache size too large to initialize.
TEST_F(RequestResponseCacheTest, TestCacheSizeTooLarge)
{
  // Pick intentionally large cache size, expecting failure
  constexpr uint64_t cache_size = ULLONG_MAX;
  std::cout << "Create cache of size: " << cache_size << std::endl;
  std::unique_ptr<tc::RequestResponseCache> cache;
  auto status = tc::RequestResponseCache::Create(cache_size, &cache);
  ASSERT_FALSE(status.IsOk()) << "Creating cache of size " << cache_size
                              << " succeeded when it should fail.";
}

// Test cache size too small to initialize.
// See following boost code for reference:
// -
// https://github.com/boostorg/interprocess/blob/41018201d6b7a34f38a0303a1ad591d978989cb8/include/boost/interprocess/managed_external_buffer.hpp#L75-L77
// -
// https://github.com/boostorg/interprocess/blob/41018201d6b7a34f38a0303a1ad591d978989cb8/include/boost/interprocess/detail/managed_memory_impl.hpp#L172-L174
TEST_F(RequestResponseCacheTest, TestCacheSizeTooSmall)
{
  // Pick intentionally small cache size, expecting failure
  constexpr uint64_t cache_size = 1;
  std::cout << "Create cache of size: " << cache_size << std::endl;
  std::unique_ptr<tc::RequestResponseCache> cache;
  auto status = tc::RequestResponseCache::Create(cache_size, &cache);
  ASSERT_FALSE(status.IsOk()) << "Creating cache of size " << cache_size
                              << " succeeded when it should fail.";
}

// Test cache too small for entry
TEST_F(RequestResponseCacheTest, TestCacheSizeSmallerThanEntry)
{
  // Create cache
  constexpr uint64_t cache_size = 1024;
  std::cout << "Create cache of size: " << cache_size << std::endl;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);

  // Set output data to be larger than cache size
  // NOTE: This is not 1 byte larger than cache_size, the cache_size + 1 is to
  // be clear it will always be larger than cache even if the dtype is changed.
  std::vector<int> large_data(cache_size + 1, 0);
  std::cout << "Create large_response (larger than cache) of size: "
            << large_data.size() << std::endl;
  std::vector<Tensor> large_outputs{Tensor{"output", large_data}};
  auto large_response = GenerateResponse(
      request0, dtype, memory_type, memory_type_id, large_outputs);

  std::cout << "Insert large_response into cache" << std::endl;
  auto status = cache->Insert(*large_response, request0);
  // We expect insertion to fail here since cache is too small
  std::cout << status.Message() << std::endl;
  ASSERT_FALSE(status.IsOk())
      << "Inserting item larger than cache succeeded when it should fail";
}

// Test hashing for consistency on same request
TEST_F(RequestResponseCacheTest, TestEviction)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 1024;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);
  cache_stats(cache);

  std::cout << "Lookup unique_requests[0] in empty cache" << std::endl;
  auto status = cache->Lookup(nullptr, unique_requests[0]);
  // This hash not in cache yet
  ASSERT_FALSE(status.IsOk())
      << "hash [" + std::to_string(unique_requests[0]->CacheKey()) +
             "] should not be in cache";
  std::cout << "Insert response into cache" << std::endl;
  check_status(cache->Insert(*response_400bytes, unique_requests[0]));
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), 1u);
  ASSERT_EQ(cache->NumEvictions(), 0u);

  check_status(cache->Insert(*response_400bytes, unique_requests[1]));
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), 2u);
  ASSERT_EQ(cache->NumEvictions(), 0u);

  check_status(cache->Insert(*response_400bytes, unique_requests[2]));
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), 2u);
  ASSERT_EQ(cache->NumEvictions(), 1u);

  check_status(cache->Insert(*response_400bytes, unique_requests[3]));
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), 2u);
  ASSERT_EQ(cache->NumEvictions(), 2u);
}

// Test inserting into cache with multiple threads in parallel
// and asserting that the correct number of entries and evictions
// occurred based on cache and entry sizes
TEST_F(RequestResponseCacheTest, TestParallelInsertion)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 1024;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);
  cache_stats(cache);

  // Create threads
  std::vector<std::thread> threads;
  std::cout << "Insert responses into cache with [" << thread_count
            << "] threads in parallel" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads.emplace_back(std::thread(
        &tc::RequestResponseCache::Insert, cache.get(),
        std::ref(*response_400bytes), unique_requests[idx]));
  }

  // Join threads
  for (size_t idx = 0; idx < thread_count; idx++) {
    std::cout << "Joining idx: " << idx << std::endl;
    threads[idx].join();
  }

  // Cache size only has room for 2 entries of 100 ints, so we expect 2 entries
  // and N-2 evictions for N threads
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), 2u) << "NumEntries: " << cache->NumEntries();
  ASSERT_EQ(cache->NumEvictions(), (uint64_t)(thread_count - 2u))
      << "NumEvictions: " << cache->NumEvictions();
}

// Test evicting from cache with multiple threads in parallel
// and asserting that the correct number of entries and evictions
// occurred
TEST_F(RequestResponseCacheTest, TestParallelEviction)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 1024;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);
  cache_stats(cache);

  // Create threads
  std::vector<std::thread> threads;

  // Insert [thread_count] entries into cache sequentially
  for (size_t idx = 0; idx < thread_count; idx++) {
    cache->Insert(*response0, unique_requests[idx]);
  }

  // Assert all entries were put into cache and no evictions occurred yet
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), (uint64_t)thread_count)
      << "NumEntries: " << cache->NumEntries();
  ASSERT_EQ(cache->NumEvictions(), 0u)
      << "NumEvictions: " << cache->NumEvictions();

  // Evict [thread_count] entries from cache in parallel
  std::cout << "Evict from cache with [" << thread_count
            << "] threads in parallel" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads.emplace_back(
        std::thread(&tc::RequestResponseCache::Evict, cache.get()));
  }

  // Join threads
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads[idx].join();
  }

  // Assert all entries were evicted from cache and exactly [thread_count]
  // evictions occurred
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), 0u) << "NumEntries: " << cache->NumEntries();
  ASSERT_EQ(cache->NumEvictions(), (uint64_t)thread_count)
      << "NumEvictions: " << cache->NumEvictions();
}

// Test LRU ordering of cache
TEST_F(RequestResponseCacheTest, TestLRU)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 1024;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);
  cache_stats(cache);

  // Insert 3 items into cache: 0, 1, 2
  check_status(cache->Insert(*response0, unique_requests[0]));
  check_status(cache->Insert(*response0, unique_requests[1]));
  check_status(cache->Insert(*response0, unique_requests[2]));

  // Verify items 0, 1, 2, in cache
  reset_response(&response0, unique_requests[0]);
  check_status(cache->Lookup(response0.get(), unique_requests[0]));
  reset_response(&response0, unique_requests[1]);
  check_status(cache->Lookup(response0.get(), unique_requests[1]));
  reset_response(&response0, unique_requests[2]);
  check_status(cache->Lookup(response0.get(), unique_requests[2]));

  // Evict item from cache, should be item 0 since it was looked up last
  cache->Evict();
  // Assert Lookup for item 0 fails but items 1, 2 succeed
  tc::Status status;
  reset_response(&response0, unique_requests[0]);
  status = cache->Lookup(response0.get(), unique_requests[0]);
  ASSERT_FALSE(status.IsOk());
  reset_response(&response0, unique_requests[1]);
  check_status(cache->Lookup(response0.get(), unique_requests[1]));
  reset_response(&response0, unique_requests[2]);
  check_status(cache->Lookup(response0.get(), unique_requests[2]));

  // Insert item 3, 4
  check_status(cache->Insert(*response0, unique_requests[3]));
  check_status(cache->Insert(*response0, unique_requests[4]));

  // Evict twice, assert items 1 and 2 were evicted
  cache->Evict();
  cache->Evict();
  reset_response(&response0, unique_requests[1]);
  status = cache->Lookup(response0.get(), unique_requests[1]);
  ASSERT_FALSE(status.IsOk());
  reset_response(&response0, unique_requests[2]);
  status = cache->Lookup(response0.get(), unique_requests[2]);
  ASSERT_FALSE(status.IsOk());

  // Lookup items 3 and 4
  reset_response(&response0, unique_requests[3]);
  check_status(cache->Lookup(response0.get(), unique_requests[3]));
  reset_response(&response0, unique_requests[4]);
  check_status(cache->Lookup(response0.get(), unique_requests[4]));

  // Evict, assert item 3 was evicted
  cache->Evict();
  reset_response(&response0, unique_requests[3]);
  status = cache->Lookup(response0.get(), unique_requests[3]);
  ASSERT_FALSE(status.IsOk());
  reset_response(&response0, unique_requests[4]);
  check_status(cache->Lookup(response0.get(), unique_requests[4]));
}

// Test looking up from cache with multiple threads in parallel
// and asserting the responses were populated correctly
TEST_F(RequestResponseCacheTest, TestParallelLookup)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 1024;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);
  cache_stats(cache);

  // Create threads
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<tc::InferenceResponse>> responses;

  // Insert [thread_count] entries into cache sequentially
  for (size_t idx = 0; idx < thread_count; idx++) {
    // Create response for each thread to fill from cache
    std::unique_ptr<tc::InferenceResponse> response;
    check_status(
        unique_requests[idx]->ResponseFactory()->CreateResponse(&response));
    responses.push_back(std::move(response));
    // Insert response for each thread
    cache->Insert(*response0, unique_requests[idx]);
  }

  // Assert all entries were put into cache and no evictions occurred yet
  cache_stats(cache);
  ASSERT_EQ(cache->NumEntries(), (uint64_t)thread_count)
      << "NumEntries: " << cache->NumEntries();
  ASSERT_EQ(cache->NumEvictions(), 0u)
      << "NumEvictions: " << cache->NumEvictions();

  // Lookup [thread_count] entries from cache in parallel
  std::cout << "Lookup from cache with [" << thread_count
            << "] threads in parallel" << std::endl;
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads.emplace_back(std::thread(
        &tc::RequestResponseCache::Lookup, cache.get(), responses[idx].get(),
        unique_requests[idx]));
  }

  // Join threads
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads[idx].join();
  }

  // Grab output from sample response for comparison
  const auto& response0_output = response0->Outputs()[0];

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
      check_status(response_test_output.DataBuffer(
          &response_buffer, &response_byte_size, &response_memory_type,
          &response_memory_type_id, &userp));

      // TODO: Use Triton DType to cast buffer and compare outputs generically
      int* cache_output = (int*)response_buffer;
      std::cout << "Check output buffer data from cache entry for thread ["
                << idx << "]:" << std::endl;
      for (size_t i = 0; i < response_byte_size / sizeof(int); i++) {
        std::cout << cache_output[i] << " == " << data0[i] << std::endl;
        ASSERT_EQ(cache_output[i], data0[i]);
      }
    }
  }
}

// Test end-to-end flow of cache
TEST_F(RequestResponseCacheTest, TestEndToEnd)
{
  // Create cache
  std::cout << "Create cache" << std::endl;
  uint64_t cache_size = 256;
  std::unique_ptr<tc::RequestResponseCache> cache;
  tc::RequestResponseCache::Create(cache_size, &cache);
  cache_stats(cache);

  std::cout << "Lookup request0 in empty cache" << std::endl;
  auto status = cache->Lookup(nullptr, request0);
  // This hash not in cache yet
  ASSERT_FALSE(status.IsOk()) << "hash [" +
                                     std::to_string(request0->CacheKey()) +
                                     "] should not be in cache";
  std::cout << "Insert response into cache with request0" << std::endl;
  // Insertion should succeed
  check_status(cache->Insert(*response0, request0));
  cache_stats(cache);

  // Check cache stats
  auto total_lookup_latency = cache->TotalLookupLatencyNs();
  auto total_insertion_latency = cache->TotalInsertionLatencyNs();
  std::cout << "Total lookup latency: " << total_lookup_latency << std::endl;
  std::cout << "Total insertion latency: " << total_insertion_latency
            << std::endl;
  ASSERT_TRUE(total_lookup_latency > 0)
      << "ERROR: Total lookup latency should be non-zero";
  ASSERT_TRUE(total_insertion_latency > 0)
      << "ERROR: Total insertion latency should be non-zero";

  // Duplicate insertion should fail since request0 already exists in cache
  status = cache->Insert(*response0, request0);
  ASSERT_FALSE(status.IsOk())
      << "Inserting duplicate item in cache should fail";

  // Create response to test cache lookup
  std::cout << "Create response object into fill from cache" << std::endl;
  std::unique_ptr<tc::InferenceResponse> response_test;
  check_status(request0->ResponseFactory()->CreateResponse(&response_test));

  // Lookup should now succeed
  std::cout << "Lookup request0 in cache after insertion" << std::endl;
  check_status(cache->Lookup(response_test.get(), request0));

  // Check cache stats again
  auto total_lookup_latency2 = cache->TotalLookupLatencyNs();
  auto total_insertion_latency2 = cache->TotalInsertionLatencyNs();
  std::cout << "Total lookup latency2: " << total_lookup_latency2 << std::endl;
  std::cout << "Total insertion latency2: " << total_insertion_latency2
            << std::endl;
  ASSERT_TRUE(total_lookup_latency2 > total_lookup_latency)
      << "ERROR: Total lookup latency should increase";
  ASSERT_TRUE(total_insertion_latency2 > total_insertion_latency)
      << "ERROR: Total insertion latency should increase";

  // Grab output from sample response for comparison
  const auto& response0_output = response0->Outputs()[0];

  // Fetch output buffer details
  const void* response_buffer = nullptr;
  size_t response_byte_size = 0;
  TRITONSERVER_MemoryType response_memory_type;
  int64_t response_memory_type_id;
  void* userp;
  // TODO: How to handle different memory types? GPU vs CPU vs Pinned, etc.
  // TODO: Handle multiple outputs more generically
  for (const auto& response_test_output : response_test->Outputs()) {
    ASSERT_EQ(response_test_output.Name(), response0_output.Name());
    ASSERT_EQ(response_test_output.DType(), response0_output.DType());
    ASSERT_EQ(response_test_output.Shape(), response0_output.Shape());
    check_status(response_test_output.DataBuffer(
        &response_buffer, &response_byte_size, &response_memory_type,
        &response_memory_type_id, &userp));
  }

  // TODO: Use Triton DType to cast buffer and compare outputs generically
  int* cache_output = (int*)response_buffer;
  std::cout << "Check output buffer data from cache entry:" << std::endl;
  for (size_t i = 0; i < response_byte_size / sizeof(int); i++) {
    std::cout << cache_output[i] << " == " << outputs0[0].data[i] << std::endl;
    ASSERT_EQ(cache_output[i], outputs0[0].data[i]);
  }

  // Simple Evict() test
  ASSERT_EQ(cache->NumEntries(), 1u);
  ASSERT_EQ(cache->NumEvictions(), 0u);
  cache->Evict();
  ASSERT_EQ(cache->NumEntries(), 0u);
  ASSERT_EQ(cache->NumEvictions(), 1u);
  std::cout << "Done!" << std::endl;
}

}  // namespace

int
main(int argc, char** argv)
{
#ifdef TRITON_ENABLE_LOGGING
  LOG_SET_VERBOSE(1);
#endif  // TRITON_ENABLE_LOGGING

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
