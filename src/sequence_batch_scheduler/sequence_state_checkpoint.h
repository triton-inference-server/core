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

#pragma once

#include <array>
#include <string>
#include <vector>

#include "sequence_state.h"

namespace triton { namespace core {
using SequenceStateFingerprint = std::array<unsigned char, 32>;
Status SequenceStateModelFingerprint(
    const std::string& model_name, int64_t version,
    const std::string& state_layout, SequenceStateFingerprint* result);
Status CaptureSequenceState(
    const std::shared_ptr<SequenceStates>& states, size_t expected_byte_size,
    std::vector<char>* payload);
Status WriteSequenceStateCheckpoint(
    const std::string& path, uint64_t id,
    const SequenceStateFingerprint& fingerprint,
    const std::vector<char>& payload);
Status ReadSequenceStateCheckpoint(
    const std::string& path, uint64_t id,
    const SequenceStateFingerprint& fingerprint, size_t expected_byte_size,
    std::vector<char>* payload);
Status DiscardSequenceStateCheckpoint(
    const std::string& path, uint64_t id,
    const SequenceStateFingerprint& fingerprint, size_t expected_byte_size);
Status RestoreSequenceStateCheckpoint(
    const std::vector<char>& payload,
    const std::shared_ptr<SequenceStates>& states);
}}  // namespace triton::core
