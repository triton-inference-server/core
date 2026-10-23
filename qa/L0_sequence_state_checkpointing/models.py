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
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Generate deterministic CPU ONNX models. All mutable state is implicit."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

p = argparse.ArgumentParser()
p.add_argument("repository", type=Path)
args = p.parse_args()
for name, neural, single in [
    ("acc", False, False),
    ("acc_single", False, True),
    ("rnn", True, False),
    ("acc_mismatch", False, False),
]:
    d = 4 if neural else 1
    dtype = T.FLOAT if neural else T.INT32
    dtname = "FP32" if neural else "INT32"
    shape = ["batch", d]
    inputs = [
        h.make_tensor_value_info("X", dtype, shape),
        h.make_tensor_value_info("S_IN", dtype, shape),
        h.make_tensor_value_info("START", T.BOOL, ["batch", 1]),
    ]
    outputs = [
        h.make_tensor_value_info("Y", dtype, shape),
        h.make_tensor_value_info("S_OUT", dtype, shape),
    ]
    zero = np.zeros((1, d), dtype=np.float32 if neural else np.int32)
    initializers = [nh.from_array(zero, "zero")]
    nodes = [h.make_node("Where", ["START", "zero", "S_IN"], ["state"])]
    if neural:
        # A genuine recurrent neural cell h(t)=tanh(x(t)Wx+h(t-1)Wh+b).
        wx = np.array(
            [
                [0.3, 0.1, -0.2, 0.05],
                [0.1, 0.4, 0.05, -0.1],
                [0.2, -0.1, 0.3, 0.1],
                [-0.1, 0.2, 0.1, 0.25],
            ],
            np.float32,
        )
        wh = np.array(
            [[0.4, 0.1, 0, 0], [0, 0.3, 0.1, 0], [0, 0, 0.2, 0.1], [0.1, 0, 0, 0.35]],
            np.float32,
        )
        initializers += [
            nh.from_array(wx, "wx"),
            nh.from_array(wh, "wh"),
            nh.from_array(np.array([0.01, -0.02, 0.03, -0.04], np.float32), "bias"),
        ]
        nodes += [
            h.make_node("MatMul", ["X", "wx"], ["xw"]),
            h.make_node("MatMul", ["state", "wh"], ["hw"]),
            h.make_node("Add", ["xw", "hw"], ["sum"]),
            h.make_node("Add", ["sum", "bias"], ["pre"]),
            h.make_node("Tanh", ["pre"], ["S_OUT"]),
        ]
    else:
        nodes += [h.make_node("Add", ["state", "X"], ["S_OUT"])]
    nodes += [h.make_node("Identity", ["S_OUT"], ["Y"])]
    model = h.make_model(
        h.make_graph(nodes, name, inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    folder = args.repository / name
    (folder / "1").mkdir(parents=True, exist_ok=True)
    onnx.save(model, folder / "1/model.onnx")
    config = f"""name: "{name}"
backend: "onnxruntime"
max_batch_size: 2
input [ {{ name: "X" data_type: TYPE_{dtname} dims: [ {d} ] }} ]
output [ {{ name: "Y" data_type: TYPE_{dtname} dims: [ {d} ] }} ]
instance_group [ {{ count: 1 kind: KIND_CPU }} ]
parameters {{ key: "intra_op_thread_count" value {{ string_value: "1" }} }}
parameters {{ key: "inter_op_thread_count" value {{ string_value: "1" }} }}
sequence_batching {{
  max_sequence_idle_microseconds: 600000000
  direct {{ }}
  control_input [ {{ name: "START" control [ {{ kind: CONTROL_SEQUENCE_START bool_false_true: [ false, true ] }} ] }} ]
  state [ {{ input_name: "S_IN" output_name: "S_OUT" data_type: TYPE_{dtname} dims: [ {d} ]
    use_same_buffer_for_input_output: {'true' if single else 'false'}
    initial_state {{ name: "zero" data_type: TYPE_{dtname} dims: [ {d} ] zero_data: true }}
  }} ]
}}
"""
    config += 'parameters { key: "sequence_state_checkpointing" value { string_value: "true" } }\n'
    (folder / "config.pbtxt").write_text(config)
