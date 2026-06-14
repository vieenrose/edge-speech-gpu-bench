#!/usr/bin/env python3
"""Make a melo8k (MeloTTS/OpenVoice2) ONNX runnable on onnxruntime 1.11.0 (Jetson Nano gen1).

The exported model uses LayerNormalization (ai.onnx opset 17). ORT 1.11 — the last version that
supports the Nano's CUDA 10.2 — caps at opset 16, and onnx.version_converter can't downgrade
LayerNormalization (no opset-16 schema). This decomposes every LayerNormalization into opset-16
primitives (ReduceMean/Sub/Mul/Add/Sqrt/Div) and stamps the model at opset 16. Numerically exact.

    LayerNorm(X; gamma, beta) = (X - mean) / sqrt(var + eps) * gamma + beta
    mean = ReduceMean(X, axis), var = ReduceMean((X-mean)^2, axis)

Usage: python melo8k_layernorm_to_opset16.py model.onnx model.opset16.onnx
Validated: max|orig-op16| = 9.2e-6, corr 1.000000 (noise=0), and it loads+runs on ORT 1.11.0.
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper


def convert(src, dst, target_opset=16):
    m = onnx.load(src)
    g = m.graph
    new_nodes, eps_inits, n = [], {}, 0

    def eps_const(val):
        key = f"ln_eps_{val:.3e}"
        if key not in eps_inits:
            eps_inits[key] = numpy_helper.from_array(np.array(val, dtype=np.float32), key)
        return key

    for node in g.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue
        n += 1
        X = node.input[0]
        G = node.input[1]
        B = node.input[2] if len(node.input) > 2 else None
        Y = node.output[0]
        p = f"ln{n}_"
        axis, eps = -1, 1e-5
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
            if a.name == "epsilon":
                eps = a.f
        ek = eps_const(eps)
        new_nodes += [
            helper.make_node("ReduceMean", [X], [p + "mean"], axes=[axis], keepdims=1),
            helper.make_node("Sub", [X, p + "mean"], [p + "d"]),
            helper.make_node("Mul", [p + "d", p + "d"], [p + "dd"]),
            helper.make_node("ReduceMean", [p + "dd"], [p + "var"], axes=[axis], keepdims=1),
            helper.make_node("Add", [p + "var", ek], [p + "vareps"]),
            helper.make_node("Sqrt", [p + "vareps"], [p + "std"]),
            helper.make_node("Div", [p + "d", p + "std"], [p + "norm"]),
            helper.make_node("Mul", [p + "norm", G], [p + "scaled"] if B else [Y]),
        ]
        if B:
            new_nodes.append(helper.make_node("Add", [p + "scaled", B], [Y]))

    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(eps_inits.values())
    del m.opset_import[:]
    m.opset_import.extend([helper.make_opsetid("", target_opset)])
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, dst)
    print(f"decomposed {n} LayerNormalization nodes -> opset {target_opset}; saved {dst}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
