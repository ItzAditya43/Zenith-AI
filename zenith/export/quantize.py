"""int8 quantisation with onnxruntime, then re-tie the embedding.

Two kinds of int8:
  dynamic      weights *and* activations int8 (MatMulInteger) — fast, used for
               the output head, wo, w1/w3.
  weight-only  int8 weights + DequantizeLinear, fp32 activations — for layers
               whose inputs have too wide a range for 8 bits. On the trained
               model, dynamic-quantising the SwiGLU down-projections (w2, input
               max/median ~450x) alone dropped logit correlation to 0.45 and
               top-1 agreement to 19%; wq/wk/wv cost ~12% agreement. Same file
               size either way.

The exporter writes the tied embedding twice (the Gather table and a transposed
copy for the output MatMul), and quantisation keeps both: 2 x 10.2MB. The
output MatMul's weight is quantised per output channel, i.e. one scale per
vocab entry, which is exactly a per-row quantised embedding table. So after
quantisation the embedding lookup is rebuilt as a Gather on that int8 matrix
(+ its per-token scale) and the duplicate table is dropped.
"""
import argparse
import os
import re

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import QuantType, quantize_dynamic


WEIGHT_ONLY = re.compile(r"/layers\.\d+/|/(w[0-9qkvo])(_\d+)?/")


def weight_only_int8(path, name_pattern):
    """Replace fp32 MatMul weights matching name_pattern with per-output-channel
    symmetric int8 + DequantizeLinear (activations stay fp32)."""
    m = onnx.load(path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    n_done = 0
    for node in list(g.node):
        if node.op_type != "MatMul" or not name_pattern.search(node.name) or node.input[1] not in inits:
            continue
        w = onnx.numpy_helper.to_array(inits[node.input[1]])  # (in, out)
        scale = (np.abs(w).max(axis=0) / 127.0).astype(np.float32)
        scale[scale == 0] = 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        base = node.input[1]
        g.initializer.remove(inits[base])
        g.initializer.extend([
            onnx.numpy_helper.from_array(q, base + "_wq"),
            onnx.numpy_helper.from_array(scale, base + "_wscale"),
            onnx.numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), base + "_wzp"),
        ])
        g.node.insert(list(g.node).index(node), helper.make_node(
            "DequantizeLinear", [base + "_wq", base + "_wscale", base + "_wzp"], [base + "_deq"],
            axis=1, name=node.name + "_weight_dequant"))
        node.input[1] = base + "_deq"
        n_done += 1
    onnx.save(m, path)
    return n_done


def share_tied_embedding(path, unsmooth=None):
    m = onnx.load(path)
    g = m.graph
    gather = next(n for n in g.node if n.op_type == "Gather" and n.input[0].endswith("tok_embeddings.weight_quantized"))
    deq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == gather.output[0])
    mmi = next(n for n in g.node if n.op_type == "MatMulInteger" and n.output[0].startswith("logits"))
    w_q, w_scale, w_zp = mmi.input[1], mmi.input[1].replace("_quantized", "_scale"), mmi.input[3]
    inits = {i.name: i for i in g.initializer}
    zp = onnx.numpy_helper.to_array(inits[w_zp])
    assert not zp.any(), "expected symmetric int8 weights (zero point 0)"

    tokens, out = gather.input[1], deq.output[0]
    axes = helper.make_tensor("emb_unsqueeze_axes", TensorProto.INT64, [1], [-1])
    g.initializer.append(axes)
    new_nodes = [
        helper.make_node("Gather", [w_q, tokens], ["emb_q"], axis=1, name="emb_gather"),         # (D, B, T) int8
        helper.make_node("Cast", ["emb_q"], ["emb_f"], to=TensorProto.FLOAT, name="emb_cast"),
        helper.make_node("Transpose", ["emb_f"], ["emb_t"], perm=[1, 2, 0], name="emb_transpose"),  # (B, T, D)
        helper.make_node("Gather", [w_scale, tokens], ["emb_s"], axis=0, name="emb_scale_gather"),  # (B, T)
        helper.make_node("Unsqueeze", ["emb_s", "emb_unsqueeze_axes"], ["emb_s3"], name="emb_scale_unsqueeze"),
        helper.make_node("Mul", ["emb_t", "emb_s3"], ["emb_deq" if unsmooth is not None else out], name="emb_dequant"),
    ]
    if unsmooth is not None:
        # head weight is E*s (SmoothQuant, see export_onnx.smooth_output_head); undo s
        g.initializer.append(onnx.numpy_helper.from_array(unsmooth, "zenith_unsmooth"))
        new_nodes.append(helper.make_node("Mul", ["emb_deq", "zenith_unsmooth"], [out], name="emb_unsmooth"))
    idx = list(g.node).index(gather)
    g.node.remove(gather)
    g.node.remove(deq)
    for k, n in enumerate(new_nodes):
        g.node.insert(idx + k, n)
    for name in list(gather.input[:1]) + list(deq.input[1:]):
        if name in inits:
            g.initializer.remove(inits[name])
    onnx.checker.check_model(m)
    onnx.save(m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="export/zenith_kb_fp32.onnx")
    ap.add_argument("--output", default="export/zenith_kb.onnx")
    args = ap.parse_args()

    src = onnx.load(args.input)
    inits = {i.name for i in src.graph.initializer}
    # SmoothQuant's 1/s rides along as an unused initializer, which the
    # quantizer prunes, so grab it from the fp32 model first
    unsmooth = next((onnx.numpy_helper.to_array(i) for i in src.graph.initializer if i.name == "zenith_unsmooth"), None)
    dynamic_nodes = [n.name for n in src.graph.node
                     if (n.op_type == "MatMul" and n.input[1] in inits and not WEIGHT_ONLY.search(n.name))
                     or (n.op_type == "Gather" and n.input[0] in inits)]
    quantize_dynamic(
        args.input, args.output,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gather"],
        nodes_to_quantize=dynamic_nodes,
        per_channel=True,
        extra_options={"EnableSubgraph": False},
    )
    n_wo = weight_only_int8(args.output, WEIGHT_ONLY)
    share_tied_embedding(args.output, unsmooth)
    print(f"dynamic int8: {len(dynamic_nodes)} nodes, weight-only int8: {n_wo} MatMuls")
    print(f"{args.input} ({os.path.getsize(args.input) / 1e6:.1f} MB) -> "
          f"{args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
