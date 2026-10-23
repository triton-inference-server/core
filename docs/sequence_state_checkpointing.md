<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->
# Sequence state checkpointing

Sequence state checkpointing persists Triton-managed implicit state and frees
the sequence slot. The sequence can later resume without replaying requests.

Enable the feature in the model configuration:

```protobuf
parameters {
  key: "sequence_state_checkpointing"
  value { string_value: "true" }
}
```

The C API supports suspend, resume, discard, and status operations. Resume
continues the sequence without the START flag.

## Scope

The current implementation supports one CPU model instance with direct,
non-iterative, non-decoupled sequence batching and fixed numeric implicit
state. GPU state, growable state, multiple instances, and oldest scheduling
are not supported.

## Operation

Suspend closes admission for one sequence and waits for accepted requests to
finish. The batcher copies the next-input state at its execution boundary.
File I/O runs on the calling thread, so other sequence slots can continue.
The slot is released only after the checkpoint is durably published.

Resume validates the checkpoint before reserving a slot. Validation covers the
format, checksum, sequence ID, model identity, and state layout.

Checkpoint files are written to a temporary file, synchronized with `fsync`,
published without overwriting an existing file, and followed by a parent
directory `fsync`. Reads reject symlinks, non-regular files, truncation,
unexpected sizes, and checksum mismatches.

The checkpoint size is derived from the configured state tensors. Control
operations are serialized per model.

## Validation

Run the Linux integration harness described in
`qa/L0_sequence_state_checkpointing/README.md`.
