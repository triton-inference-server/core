// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//

#include "cuda_block_manager.h"

#include "cuda_utils.h"


namespace triton { namespace core {
std::unique_ptr<CudaBlockManager> CudaBlockManager::instance_ = nullptr;

void
Allocation::Merge(std::unique_ptr<Allocation>&& allocation)
{
  for (auto block : allocation->Blocks()) {
    blocks_.push_back(block);
  }
  allocation->Blocks().clear();
}

Status
CudaBlockManager::Create(double min_supported_compute_capability)
{
  if (instance_ != nullptr) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("CUDA Block manager has already been created.")));
  }

  std::set<int> supported_gpus;
  RETURN_IF_ERROR(
      GetSupportedGPUs(&supported_gpus, min_supported_compute_capability));

  size_t block_size;
  RETURN_IF_ERROR(GetAllocationGranularity(block_size));
  instance_.reset(new CudaBlockManager());

  instance_->block_size_ = block_size;
  for (auto gpu : supported_gpus) {
    instance_->free_blocks_[gpu] = std::vector<CUmemGenericAllocationHandle>();
  }

  return Status::Success;
}

Status
CudaBlockManager::Allocate(
    size_t size, std::unique_ptr<Allocation>& allocation, int device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("CUDA Block manager has not been created.")));
  }

  std::lock_guard<std::mutex> lock(instance_->mu_);
  if (instance_->free_blocks_.count(device_id) == 0) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("Invalid device id '") + std::to_string(device_id) +
         "' to allocate memory from."));
  }

  // Align the byte size to number of blocks. The allocated bytes must be a
  // multiple of the cudaAllocationGranularity. In the current implementation
  // the block size is the cuda minimum allocation granularity.
  size_t num_blocks =
      (size + instance_->block_size_ - 1) / (instance_->block_size_);

  size_t free_blocks_size = instance_->free_blocks_[device_id].size();
  size_t block_index = 0;

  for (; block_index < num_blocks && block_index < free_blocks_size;
       ++block_index) {
    allocation->AddBlock(instance_->free_blocks_[device_id].back());
    instance_->free_blocks_[device_id].pop_back();
  }

  for (; block_index < num_blocks; ++block_index) {
    CUmemGenericAllocationHandle block{};
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    RETURN_IF_ERROR(CudaDriverHelper::GetInstance().CuMemCreate(
        &block, instance_->block_size_, &prop, 0 /* flags */));
    allocation->AddBlock(block);
  }

  return Status::Success;
}

Status
CudaBlockManager::Free(Allocation* allocation, int device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("CUDA Block manager has not been created.")));
  }

  std::lock_guard<std::mutex> lock(instance_->mu_);

  if (instance_->free_blocks_.count(device_id) == 0) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("Invalid device id '") + std::to_string(device_id) +
         "' to allocate memory from."));
  }

  auto& free_blocks = instance_->free_blocks_[device_id];
  free_blocks.insert(
      free_blocks.end(), allocation->Blocks().begin(),
      allocation->Blocks().end());
  allocation->Blocks().clear();

  return Status::Success;
}

CudaBlockManager::~CudaBlockManager()
{
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& free_block_pair : free_blocks_) {
    auto free_blocks = free_block_pair.second;
    for (auto& free_block : free_blocks) {
      CudaDriverHelper::GetInstance().CuMemRelease(free_block);
    }
  }
}

}};  // namespace triton::core
