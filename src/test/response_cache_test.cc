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
#include "cache_manager.h"
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
tc::Status
TestCacheImpl(std::shared_ptr<tc::TritonCache> cache)
{
  std::cout << "======================================" << std::endl;
  std::cout << "==== Testing Cache Implementation ====" << std::endl;
  std::cout << "======================================" << std::endl;
  if (!cache) {
    return tc::Status(tc::Status::Code::INTERNAL, "cache was nullptr");
  }

  auto status = tc::Status::Success;
  std::cout << "=============== Insert Bytes ===============" << std::endl;
  // Setup byte buffers
  std::vector<std::byte> buffer1{1, std::byte{0x01}};
  std::vector<std::byte> buffer2{2, std::byte{0x02}};
  std::vector<std::byte> buffer3{4, std::byte{0x03}};
  std::vector<std::byte> buffer4{8, std::byte{0x04}};
  std::vector<std::byte> buffer5{16, std::byte{0xFF}};
  // Setup items
  std::vector<std::shared_ptr<tc::CacheEntryItem>> items;
  items.emplace_back(new tc::CacheEntryItem());
  items.emplace_back(new tc::CacheEntryItem());
  // Add buffers to items
  items[0]->AddBuffer(buffer1);
  items[0]->AddBuffer(buffer2);
  items[1]->AddBuffer(buffer3);
  items[1]->AddBuffer(buffer4);
  items[1]->AddBuffer(buffer5);
  status = cache->Insert(items, "test_bytes_123_key");
  std::cout << "=============== Lookup Bytes ===============" << std::endl;
  const auto responses = cache->Lookup("test_bytes_123_key");
  if (!responses.has_value()) {
    return tc::Status(tc::Status::Code::INTERNAL, "Lookup failed");
  }
  const auto lookup_items = responses.value();
  if (lookup_items.size() != items.size()) {
    return tc::Status(
        tc::Status::Code::INTERNAL, "Expected " + std::to_string(items.size()) +
                                        " got " +
                                        std::to_string(lookup_items.size()));
  }

  for (size_t i = 0; i < items.size(); i++) {
    auto expected_buffers = items[i]->Buffers();
    auto lookup_buffers = lookup_items[i]->Buffers();
    if (lookup_buffers.size() != expected_buffers.size()) {
      return tc::Status(
          tc::Status::Code::INTERNAL,
          "Expected " + std::to_string(expected_buffers.size()) + " got " +
              std::to_string(lookup_buffers.size()));
    }


    for (size_t b = 0; b < expected_buffers.size(); b++) {
      if (lookup_buffers[b] != expected_buffers[b]) {
        return tc::Status(
            tc::Status::Code::INTERNAL,
            "Buffer bytes didn't match for test input");
      }
    }
  }
  std::cout << "======================================" << std::endl;
  std::cout << "============ Done Testing ============" << std::endl;
  std::cout << "======================================" << std::endl;
  return tc::Status::Success;
}

void
check_status(tc::Status status)
{
  ASSERT_TRUE(status.IsOk()) << "ERROR: " << status.Message();
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


// Test end-to-end flow of cache
TEST_F(RequestResponseCacheTest, TestEndToEnd)
{
  std::cout << "Create cache" << std::endl;

  // Create CacheManager
  std::shared_ptr<tc::TritonCacheManager> cache_manager;
  auto cache_dir = "/opt/tritonserver/caches";
  check_status(tc::TritonCacheManager::Create(&cache_manager, cache_dir));

  // Create Cache
  std::shared_ptr<tc::TritonCache> cache;
  auto cache_config = R"({"size": 256})";
  check_status(cache_manager->CreateCache(
      "response_cache" /* name */, cache_config, &cache));
  ASSERT_NE(cache, nullptr);

  // TODO: Flesh out test more
  check_status(TestCacheImpl(cache));

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
