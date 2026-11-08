"""Cross-family core suite for surgery.py (Gemma 4 31B, OLMo 3.1 32B Think): the same checks as test_surgery.py's core
phase, with the op list written for each architecture.  One process = one vLLM engine.

  python test_xf.py --model <dir> --tp 2 [--out DIR]

Checks: determinism; HF->vLLM layout (bitwise vs the checkpoint, coverage incl. replica slots, gaps only at vocab padding,
aliasing only between declared tied parameters); derived-tensor audit; every op class against the independent CPU
reference of test_surgery.expected_after (selected and sibling regions, logprobs move - or, for the edits the
architecture makes invisible on the probes ('inert'), stay at bf16 noise -, bit-exact revert, full fingerprint); replica slots (Gemma 4 k_eq_v: the V slot of qkv_proj) equal to the primary after apply and revert;
residual-space directions (space='residual': the edited write is orthogonal to u through the RMSNorm gain, resp. the
read ignores u); exact-zero edits; semantic equivalences; chat template and thinking mode; score() vs generation
logprobs.  Writes <out>/core_xf_<family>_tp<T>_<stamp>.json.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_surgery as TS  # noqa: E402  (helpers only; its __main__ block does not run)
from test_surgery import Ckpt, cmp, diffstats, log, op_test, probe, summarize_layout, unit  # noqa: E402

STAMP = time.strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------------------------------------------ op test wrapper
def op_test_xf(S, ck, name, edit, base, fp_ref, expect, full_fp):
    """test_surgery.op_test, plus (a) tied parameters count as touched (Gemma 4 lm_head = embed_tokens), (b) replica
    slots read back equal to the primary after apply and after revert."""
    prims = S.expand(edit)
    reps = sorted({p["hf"] for p in prims if p["spec"].get("rsegs")})
    rep_ok = {}
    orig_apply = S.apply

    def apply_and_check(e, verify=True):
        h = orig_apply(e, verify)
        for hf in reps:
            rep_ok.setdefault(hf, {})["after_apply"] = TS.bits_equal(S.read_replica(hf), S.read(hf))
        return h
    S.apply = apply_and_check
    try:
        rec = op_test(S, ck, name, edit, base, fp_ref, expect, full_fp)
    finally:
        S.apply = orig_apply
    for hf in reps:
        rep_ok[hf]["after_revert"] = TS.bits_equal(S.read_replica(hf), S.read(hf))
        rep_ok[hf]["replica_equals_checkpoint"] = TS.bits_equal(S.read_replica(hf), ck.get(hf).to(S.read(hf).dtype))
    if reps:
        rec["replicas"] = rep_ok
    if full_fp and "fingerprint_changed_params" in rec:
        vnames = {p["vname"] for p in prims}
        tied = {a for a, t in S.alias_params.items() if t in vnames}
        rec["tied_also_changed"] = sorted(tied & set(rec["fingerprint_changed_params"]))
        rec["only_touched_changed"] = set(rec["fingerprint_changed_params"]) <= vnames | tied
    # verdict, same rule as op_test plus the replica checks
    wchk = rec["weights_vs_reference"]
    ok = rec["revert"]["blocks_bit_exact"] and rec["revert_reads_bit_exact"] and rec["logprobs_after_revert"]["identical"]
    ok = ok and rec["siblings_untouched"] and all(v["exact"] or v.get("within_bound", False) for v in wchk.values())
    if full_fp:
        ok = ok and rec["only_touched_changed"] and rec["revert_full_fingerprint_equal"]
    lpc = rec["logprobs_vs_base"]
    if expect == "change":
        ok = ok and rec["changed_elems"] > 0 and not lpc["identical"] and (lpc["d_prefill"] > 0 or lpc["n_diverged"] > 0)
    elif expect == "inert":     # weights edited exactly, outputs move only at bf16 noise level (architectural invariance)
        ok = ok and rec["changed_elems"] > 0 and lpc["d_prefill"] <= 1e-3 and lpc["n_diverged"] == 0 and lpc["d_decode"] <= 1e-2
    else:
        ok = ok and lpc["identical"] and rec["changed_elems"] == 0
    ok = ok and all(all(v.values()) for v in rep_ok.values())
    if rec["PASS"] != bool(ok):
        log(f"  verdict for {name} revised by the xf checks: {rec['PASS']} -> {bool(ok)}")
    rec["PASS"] = bool(ok)
    return rec


# ------------------------------------------------------------------------------------------------ per-family op lists
def rms(ck, hf):
    return float(ck.get(hf).float().pow(2).mean().sqrt())


def tests_gemma4(S, rng, ck):
    c, H, I = S.cfg, S.cfg.H, S.cfg.I
    P = S.layout.hf_prefix + "layers."
    full = [l for l, t in enumerate(c.layer_types) if t == "full_attention"]          # 5, 11, ..., 59
    sl = lambda L: L not in full
    assert sl(7) and sl(22) and 11 in full and 23 in full and 35 in full
    Qs, Qf = c.nq * c.hd, c.nq * c.hd_full
    toks = sorted(set(S.tokenizer.encode(" Paris France prime German blue", add_special_tokens=False)))
    t = []
    # attention: q/k/v rows per head (q_norm / k_norm / v_norm normalise per head, so scaling them is ~a no-op: use set
    # and directional ops), o_proj columns per head; heads >= nq/2 and kv heads >= nkv/2 live on rank 1 at TP2
    t.append(("set q_proj head rows (sliding L0, heads 3, 20)", {"op": "set", "param": "layers.0.self_attn.q_proj", "head": [3, 20], "value": 0.0}))
    t.append(("set q_proj head rows (full L5, head 30)", {"op": "set", "param": "layers.5.self_attn.q_proj", "head": 30, "value": 0.0}))
    t.append(("project_out in-side k_proj (sliding L2)", {"op": "project_out", "param": "layers.2.self_attn.k_proj", "side": "in", "v": unit(rng, H)}))
    t.append(("set v_proj kv-head (sliding L1, kv 9)", {"op": "set", "param": "layers.1.self_attn.v_proj", "head": 9, "value": 0.0}))
    t.append(("set k_proj kv-heads, k_eq_v full L11 (K and V slot), kv 1, 3",
              {"op": "set", "param": "layers.11.self_attn.k_proj", "head": [1, 3], "value": 0.0}))
    t.append(("amplify out-side k_proj, k_eq_v full L17 (sum over sharded rows, replica)",
              {"op": "amplify", "param": "layers.17.self_attn.k_proj", "side": "out", "u": unit(rng, c.nkv_full * c.hd_full), "alpha": 2.0}))
    t.append(("scale o_proj head cols (sliding L7, heads 5, 25)", {"op": "scale", "param": "layers.7.self_attn.o_proj", "head": [5, 25], "alpha": 3.0}))
    t.append(("scale o_proj head cols (full L23, heads 0, 31)", {"op": "scale", "param": "layers.23.self_attn.o_proj", "head": [0, 31], "alpha": 0.0}))
    # MLP neurons (I/2 = 10752 splits ranks at TP2)
    t.append(("set gate_proj neuron rows", {"op": "set", "param": "layers.10.mlp.gate_proj", "neuron": [123, 4567, 15000], "value": 0.0}))
    t.append(("set down_proj neuron cols", {"op": "set", "param": "layers.12.mlp.down_proj", "neuron": [456, 20000], "value": 0.0}))
    t.append(("scale up_proj neuron rows", {"op": "scale", "param": "layers.13.mlp.up_proj", "neuron": [789, 11000], "alpha": 4.0}))
    # rank-1 and directional ops in parameter space
    hf = P + "20.mlp.down_proj.weight"
    t.append(("rank1 down_proj", {"op": "rank1", "param": hf, "u": unit(rng, H), "v": unit(rng, I), "alpha": 0.5 * rms(ck, hf) * float(np.sqrt(H * I))}))
    t.append(("project_out out-side o_proj (param space)", {"op": "project_out", "param": "layers.22.self_attn.o_proj", "side": "out", "u": unit(rng, H)}))
    t.append(("project_out out-side gate_proj (sum over sharded rows)", {"op": "project_out", "param": "layers.24.mlp.gate_proj", "side": "out", "u": unit(rng, I)}))
    t.append(("amplify in-side down_proj (sum over sharded cols)", {"op": "amplify", "param": "layers.25.mlp.down_proj", "side": "in", "v": unit(rng, I), "alpha": 3.0}))
    t.append(("project_out in-side o_proj (sum over sharded cols, full L29)", {"op": "project_out", "param": "layers.29.self_attn.o_proj", "side": "in", "v": unit(rng, Qf)}))
    # residual-space directions (the loop's L3 / L3g edits): through post-norm / pre-norm gains
    t.append(("project_out residual out-side o_proj (post_attention_layernorm gain)",
              {"op": "project_out", "param": "layers.26.self_attn.o_proj", "side": "out", "u": unit(rng, H), "space": "residual"}))
    t.append(("amplify residual out-side down_proj (post_feedforward_layernorm gain)",
              {"op": "amplify", "param": "layers.27.mlp.down_proj", "side": "out", "u": unit(rng, H), "alpha": 2.0, "space": "residual"}))
    t.append(("project_out residual in-side gate_proj (pre_feedforward_layernorm gain)",
              {"op": "project_out", "param": "layers.28.mlp.gate_proj", "side": "in", "v": unit(rng, H), "space": "residual"}))
    t.append(("project_out residual in-side q_proj (input_layernorm gain, full L35)",
              {"op": "project_out", "param": "layers.35.self_attn.q_proj", "side": "in", "v": unit(rng, H), "space": "residual"}))
    # embeddings (tied with lm_head; vocab 262144 = 2 x 131072 at TP2)
    t.append(("scale embed_tokens rows (prompt tokens; tied lm_head)", {"op": "scale", "param": "embed_tokens", "rows": toks, "alpha": 1.5}))
    # rows 131000:131200 straddle the rank boundary (131072) but are no probe tokens, and as tied lm_head rows their
    # softcapped logits are far below the probes' tokens: the probes cannot see this edit ("inert"); the boundary
    # mapping is proven by weights_vs_reference on the rows of both ranks
    t.append(("scale embed_tokens rows across TP boundary (inert on the probes)",
              {"op": "scale", "param": "embed_tokens", "rows": "131000:131200", "alpha": 2.0}, "inert"))
    t.append(("project_out in-side embed_tokens rows 0:8192", {"op": "project_out", "param": "embed_tokens", "rows": "0:8192", "side": "in", "v": unit(rng, H)}))
    # norms (plain x*w at runtime), layer_scalar buffer
    t.append(("scale q_norm (sliding L7, attention temperature)", {"op": "scale", "param": "layers.7.self_attn.q_norm", "alpha": 1.5}))
    t.append(("scale k_norm (full L11)", {"op": "scale", "param": "layers.11.self_attn.k_norm", "alpha": 1.5}))
    # input_layernorm feeds only qkv_proj (gemma4.py:724) and q, k, v are each RMS-normalised per head right after
    # (q_norm :542, k_norm :546, v_norm :551): a uniform gain scale cancels up to eps -> "inert"
    t.append(("scale input_layernorm (inert: per-head q/k/v norms cancel it)",
              {"op": "scale", "param": "layers.9.input_layernorm", "alpha": 0.5}, "inert"))
    t.append(("scale post_attention_layernorm", {"op": "scale", "param": "layers.9.post_attention_layernorm", "alpha": 2.0}))
    t.append(("scale pre_feedforward_layernorm", {"op": "scale", "param": "layers.14.pre_feedforward_layernorm", "alpha": 0.5}))
    t.append(("scale post_feedforward_layernorm", {"op": "scale", "param": "layers.14.post_feedforward_layernorm", "alpha": 2.0}))
    t.append(("add final norm", {"op": "add", "param": "norm", "value": 0.2}))
    t.append(("scale layer_scalar (buffer, L30)", {"op": "scale", "param": "layers.30.layer_scalar", "alpha": 1.5}))
    t.append(("set layer_scalar (buffer, L3)", {"op": "set", "param": "layers.3.layer_scalar", "value": 0.5}))
    # block skipping
    t.append(("zero_block attn (sliding L30)", {"op": "zero_block", "layer": 30, "parts": ["attn"]}))
    t.append(("zero_block attn (full L41)", {"op": "zero_block", "layer": 41, "parts": ["attn"]}))
    t.append(("zero_block mlp (L33)", {"op": "zero_block", "layer": 33, "parts": ["mlp"]}))
    t.append(("composite edit list", [{"op": "scale", "param": "layers.40.self_attn.o_proj", "head": 1, "alpha": 0.7},
                                      {"op": "project_out", "param": "layers.43.mlp.up_proj", "side": "in", "v": unit(rng, H), "space": "residual"},
                                      {"op": "zero_block", "layer": 44, "parts": ["mlp"]}]))
    z = []
    z.append(("ZERO scale alpha=1 q_norm", {"op": "scale", "param": "layers.3.self_attn.q_norm", "alpha": 1.0}))
    z.append(("ZERO add 0 post_attention_layernorm", {"op": "add", "param": "layers.6.post_attention_layernorm", "value": 0.0}))
    z.append(("ZERO rank1 alpha=0 down_proj", {"op": "rank1", "param": "layers.20.mlp.down_proj", "u": unit(rng, H), "v": unit(rng, I), "alpha": 0.0}))
    z.append(("ZERO amplify alpha=0 o_proj in-side", {"op": "amplify", "param": "layers.7.self_attn.o_proj", "side": "in", "v": unit(rng, Qs), "alpha": 0.0}))
    z.append(("ZERO amplify alpha=0 k_proj out-side (replica)", {"op": "amplify", "param": "layers.17.self_attn.k_proj", "side": "out",
                                                                 "u": unit(rng, c.nkv_full * c.hd_full), "alpha": 0.0}))
    z.append(("ZERO scale 1 layer_scalar", {"op": "scale", "param": "layers.30.layer_scalar", "alpha": 1.0}))
    cur = S.read("layers.7.self_attn.o_proj", {"head": [5, 25]}).float().numpy()
    z.append(("ZERO set o_proj heads to current values", {"op": "set", "param": "layers.7.self_attn.o_proj", "head": [5, 25], "value": cur}))
    eq = [
        ("MLP neuron L15 n=1000: gate row == up row == down col",
         [{"op": "set", "param": "layers.15.mlp.gate_proj", "neuron": 1000, "value": 0.0},
          {"op": "set", "param": "layers.15.mlp.up_proj", "neuron": 1000, "value": 0.0},
          {"op": "set", "param": "layers.15.mlp.down_proj", "neuron": 1000, "value": 0.0}]),
        ("sliding L19 kv-heads 1, 9: v_proj rows == o_proj cols of their query heads (v_norm(0) = 0)",
         [{"op": "set", "param": "layers.19.self_attn.v_proj", "head": [1, 9], "value": 0.0},
          {"op": "set", "param": "layers.19.self_attn.o_proj", "head": [2, 3, 18, 19], "value": 0.0}]),
        ("k_eq_v full L47 kv-head 1, 3: k_proj rows (K and V slot) == o_proj cols of their 8+8 query heads",
         [{"op": "set", "param": "layers.47.self_attn.k_proj", "head": [1, 3], "value": 0.0},
          {"op": "set", "param": "layers.47.self_attn.o_proj", "head": list(range(8, 16)) + list(range(24, 32)), "value": 0.0}]),
    ]
    return t, z, eq


def tests_olmo3(S, rng, ck):
    c, H, I = S.cfg, S.cfg.H, S.cfg.I
    P = S.layout.hf_prefix + "layers."
    Q, K = c.nq * c.hd, c.nkv * c.hd
    m = S.layout.vshape[S.layout.specs[S.layout.canon("embed_tokens")]["vname"]][0]     # padded vocab per rank
    toks = sorted(set(S.tokenizer.encode(" Paris France prime German blue", add_special_tokens=False)))
    t = []
    t.append(("set q_proj head rows (L0, heads 3, 30)", {"op": "set", "param": "layers.0.self_attn.q_proj", "head": [3, 30], "value": 0.0}))
    t.append(("project_out in-side k_proj (L2)", {"op": "project_out", "param": "layers.2.self_attn.k_proj", "side": "in", "v": unit(rng, H)}))
    t.append(("set v_proj kv-head (full L3, kv 6)", {"op": "set", "param": "layers.3.self_attn.v_proj", "head": 6, "value": 0.0}))
    t.append(("amplify out-side k_proj (sum over sharded rows)", {"op": "amplify", "param": "layers.5.self_attn.k_proj", "side": "out", "u": unit(rng, K), "alpha": 2.0}))
    t.append(("scale o_proj head cols (heads 5, 25)", {"op": "scale", "param": "layers.7.self_attn.o_proj", "head": [5, 25], "alpha": 3.0}))
    t.append(("scale q_norm head block (OLMo q_norm spans all heads; head 25)", {"op": "scale", "param": "layers.8.self_attn.q_norm", "head": 25, "alpha": 2.0}))
    t.append(("scale k_norm kv-head block (kv 1, 6)", {"op": "scale", "param": "layers.9.self_attn.k_norm", "head": [1, 6], "alpha": 2.0}))
    t.append(("set gate_proj neuron rows", {"op": "set", "param": "layers.10.mlp.gate_proj", "neuron": [123, 4567, 20000], "value": 0.0}))
    t.append(("set down_proj neuron cols", {"op": "set", "param": "layers.12.mlp.down_proj", "neuron": [456, 27000], "value": 0.0}))
    t.append(("scale up_proj neuron rows", {"op": "scale", "param": "layers.13.mlp.up_proj", "neuron": [789, 14000], "alpha": 4.0}))
    hf = P + "20.mlp.down_proj.weight"
    t.append(("rank1 down_proj", {"op": "rank1", "param": hf, "u": unit(rng, H), "v": unit(rng, I), "alpha": 0.5 * rms(ck, hf) * float(np.sqrt(H * I))}))
    t.append(("project_out out-side o_proj (param space)", {"op": "project_out", "param": "layers.22.self_attn.o_proj", "side": "out", "u": unit(rng, H)}))
    t.append(("project_out out-side gate_proj (sum over sharded rows)", {"op": "project_out", "param": "layers.24.mlp.gate_proj", "side": "out", "u": unit(rng, I)}))
    t.append(("amplify in-side down_proj (sum over sharded cols)", {"op": "amplify", "param": "layers.25.mlp.down_proj", "side": "in", "v": unit(rng, I), "alpha": 3.0}))
    t.append(("project_out in-side o_proj (sum over sharded cols)", {"op": "project_out", "param": "layers.29.self_attn.o_proj", "side": "in", "v": unit(rng, Q)}))
    t.append(("project_out residual out-side o_proj (post_attention_layernorm gain)",
              {"op": "project_out", "param": "layers.26.self_attn.o_proj", "side": "out", "u": unit(rng, H), "space": "residual"}))
    t.append(("amplify residual out-side down_proj (post_feedforward_layernorm gain)",
              {"op": "amplify", "param": "layers.27.mlp.down_proj", "side": "out", "u": unit(rng, H), "alpha": 2.0, "space": "residual"}))
    t.append(("project_out residual in-side gate_proj (raw stream, no pre-norm)",
              {"op": "project_out", "param": "layers.28.mlp.gate_proj", "side": "in", "v": unit(rng, H), "space": "residual"}))
    t.append(("project_out in-side lm_head rows 0:8192", {"op": "project_out", "param": "lm_head", "rows": "0:8192", "side": "in", "v": unit(rng, H)}))
    t.append(("scale lm_head rows across TP boundary (padded vocab)", {"op": "scale", "param": "lm_head", "rows": f"{m - 100}:{m + 100}", "alpha": 2.0}))
    t.append(("scale embed_tokens rows (prompt tokens)", {"op": "scale", "param": "embed_tokens", "rows": toks, "alpha": 1.5}))
    t.append(("scale post_attention_layernorm", {"op": "scale", "param": "layers.9.post_attention_layernorm", "alpha": 2.0}))
    t.append(("scale post_feedforward_layernorm", {"op": "scale", "param": "layers.14.post_feedforward_layernorm", "alpha": 0.5}))
    t.append(("add final norm", {"op": "add", "param": "norm", "value": 0.2}))
    t.append(("zero_block attn (sliding L30)", {"op": "zero_block", "layer": 30, "parts": ["attn"]}))
    t.append(("zero_block attn (full L31)", {"op": "zero_block", "layer": 31, "parts": ["attn"]}))
    t.append(("zero_block mlp (L33)", {"op": "zero_block", "layer": 33, "parts": ["mlp"]}))
    t.append(("composite edit list", [{"op": "scale", "param": "layers.40.self_attn.o_proj", "head": 1, "alpha": 0.7},
                                      {"op": "project_out", "param": "layers.41.mlp.up_proj", "side": "in", "v": unit(rng, H), "space": "residual"},
                                      {"op": "zero_block", "layer": 42, "parts": ["mlp"]}]))
    z = []
    z.append(("ZERO scale alpha=1 q_norm", {"op": "scale", "param": "layers.3.self_attn.q_norm", "alpha": 1.0}))
    z.append(("ZERO add 0 post_attention_layernorm", {"op": "add", "param": "layers.6.post_attention_layernorm", "value": 0.0}))
    z.append(("ZERO rank1 alpha=0 down_proj", {"op": "rank1", "param": "layers.20.mlp.down_proj", "u": unit(rng, H), "v": unit(rng, I), "alpha": 0.0}))
    z.append(("ZERO amplify alpha=0 o_proj in-side", {"op": "amplify", "param": "layers.7.self_attn.o_proj", "side": "in", "v": unit(rng, Q), "alpha": 0.0}))
    z.append(("ZERO amplify alpha=0 gate_proj out-side", {"op": "amplify", "param": "layers.24.mlp.gate_proj", "side": "out", "u": unit(rng, I), "alpha": 0.0}))
    cur = S.read("layers.7.self_attn.o_proj", {"head": [5, 25]}).float().numpy()
    z.append(("ZERO set o_proj heads to current values", {"op": "set", "param": "layers.7.self_attn.o_proj", "head": [5, 25], "value": cur}))
    g = c.nq // c.nkv
    eq = [
        ("MLP neuron L15 n=1000: gate row == up row == down col",
         [{"op": "set", "param": "layers.15.mlp.gate_proj", "neuron": 1000, "value": 0.0},
          {"op": "set", "param": "layers.15.mlp.up_proj", "neuron": 1000, "value": 0.0},
          {"op": "set", "param": "layers.15.mlp.down_proj", "neuron": 1000, "value": 0.0}]),
        (f"L19 kv-heads 1, 6: v_proj rows == o_proj cols of their {g}+{g} query heads",
         [{"op": "set", "param": "layers.19.self_attn.v_proj", "head": [1, 6], "value": 0.0},
          {"op": "set", "param": "layers.19.self_attn.o_proj", "head": list(range(g, 2 * g)) + list(range(6 * g, 7 * g)), "value": 0.0}]),
    ]
    return t, z, eq


# ------------------------------------------------------------------------------------------------ semantic checks
def residual_semantics(S, rng):
    """CPU check on the real edited weights that space='residual' does what it claims.
    writer (out side): for random inputs a, the block's residual write r = g * rms_norm(W' a) has u^T r / |r| ~ 0;
    reader (in side): W' (g * u) ~ 0, i.e. the read no longer sees the stream's u-component (beyond the rms)."""
    import torch
    out = {}
    lay = S.layout
    L = 26
    for kind, nm, side in (("writer", f"layers.{L}.self_attn.o_proj", "out"), ("writer", f"layers.{L + 1}.mlp.down_proj", "out"),
                           ("reader", f"layers.{L + 2}.mlp.gate_proj", "in")):
        sp = S.spec(nm)
        gname, form = lay.resid_gain(sp, side)
        g = torch.ones(S.cfg.H, dtype=torch.float64) if gname is None else S.read(gname).double()
        u = torch.as_tensor(unit(rng, S.cfg.H)).double()
        W0 = S.read(nm).double()
        h = S.apply({"op": "project_out", "param": nm, "side": side, "u" if side == "out" else "v": u.numpy(), "space": "residual"})
        W1 = S.read(nm).double()
        S.revert(h)
        if kind == "writer":
            a = torch.randn(W0.shape[1], 64, dtype=torch.float64)

            def frac(W):
                z = W @ a
                r = g[:, None] * z / z.pow(2).mean(0, keepdim=True).sqrt()
                return float(((u @ r).abs() / r.norm(dim=0)).mean())
            out[nm] = dict(gain=gname, mean_abs_cos_u_write_before=frac(W0), after=frac(W1))
        else:
            gu = g * u
            out[nm] = dict(gain=gname, read_of_gu_rel_before=float((W0 @ gu).norm() / (W0.norm() * gu.norm() / W0.shape[1] ** 0.5)),
                           after=float((W1 @ gu).norm() / (W0.norm() * gu.norm() / W0.shape[1] ** 0.5)))
    ok = all(v["after"] < 0.02 * max(v.get("mean_abs_cos_u_write_before", v.get("read_of_gu_rel_before")), 1e-9) or v["after"] < 5e-3
             for v in out.values())
    return dict(checks=out, PASS=bool(ok))


def equiv_tests(S, base, eq):
    out = []
    for name, variants in eq:
        res = []
        for v in variants:
            h = S.apply(v)
            res.append(probe(S))
            S.revert(h)
        same = all(a["ids"] == b["ids"] and a["lp"] == b["lp"] for r in res[1:] for a, b in zip(res[0], r))
        rec = dict(name=name, variants_bit_identical=same, differs_from_base=not cmp(base, res[0])["identical"],
                   vs_base=cmp(base, res[0]), pairwise=[cmp(res[0], r) for r in res[1:]])
        rec["PASS"] = bool(same and rec["differs_from_base"])
        log(f"{'PASS' if rec['PASS'] else 'FAIL'} EQUIV {name}: {rec}")
        out.append(rec)
    return out


def layout_ok(S, rep):
    """Mismatches none; coverage gaps only the declared vocab padding rows; aliasing only between declared tied params."""
    bad = []
    for r in rep:
        if r["mismatches"]:
            bad.append((r["rank"], "mismatch", r["mismatches"][:3]))
        for vn, cb in r["coverage_bad"].items():
            if cb["overlaps"] or [tuple(g) for g in cb["gaps"]] != S.layout.expected_gaps(vn, r["rank"]):
                bad.append((r["rank"], vn, cb))
        canon = lambda n: S.alias_params.get(n, n)
        for a, b in r["aliasing"]:
            if canon(a) != canon(b):
                bad.append((r["rank"], "alias", a, b))
    return bad


def chat_checks(S, prompts):
    r = {}
    q = "What is 17 * 23?"
    for name, kw in [("default", {}), ("thinking_on", dict(enable_thinking=True)), ("thinking_off", dict(enable_thinking=False)),
                     ("effort_medium", dict(reasoning_effort="medium")), ("effort_none", dict(reasoning_effort="none"))]:
        r[f"render_{name}"] = S.render(q, **kw)
    g = S.generate(["What is 17 * 23? Reply with just the number."], max_tokens=2048, reasoning_effort="medium",
                   sampling={"temperature": 0.0})
    s = g[0]["samples"][0]
    ids = s["token_ids"]
    r["gen_medium_greedy"] = dict(num_tokens=s["num_tokens"], finish_reason=s["finish_reason"], think_closed=s["think_closed"],
                                  think_end=s.get("think_end"), reasoning=s["reasoning"][:400], content=s["content"],
                                  head_tokens=S.tokenizer.convert_ids_to_tokens(ids[:6]),
                                  around_end=S.tokenizer.convert_ids_to_tokens(ids[max(0, (s.get("think_end") or 0) - 3):(s.get("think_end") or 0) + 4]))
    g = S.generate(["What is 17 * 23? Reply with just the number."], max_tokens=256, enable_thinking=False, sampling={"temperature": 0.0})
    s = g[0]["samples"][0]
    r["gen_thinking_off_greedy"] = dict(num_tokens=s["num_tokens"], finish_reason=s["finish_reason"], think_closed=s["think_closed"],
                                        reasoning=s["reasoning"][:200], content=s["content"][:300],
                                        head_tokens=S.tokenizer.convert_ids_to_tokens(s["token_ids"][:6]))
    t0 = time.time()
    g = S.generate(prompts[:2], n=2, max_tokens=8192, reasoning_effort="medium")
    r["gen_math_medium_secs"] = round(time.time() - t0, 1)
    r["gen_math_medium"] = [dict(prompt_tokens=x["prompt_tokens"], samples=[dict(
        num_tokens=s["num_tokens"], finish_reason=s["finish_reason"], think_closed=s["think_closed"], think_end=s.get("think_end"),
        reasoning_head=s["reasoning"][:200], reasoning_tail=s["reasoning"][-200:], content_tail=s["content"][-300:]) for s in x["samples"]]) for x in g]
    return r


def score_checks(S, prompts):
    res = {}
    g = S.generate(prompts[:4], max_tokens=4096, logprobs=0, reasoning_effort="medium")
    flat = [(r, s) for r in g for s in r["samples"]]
    seqs = [r["prompt_token_ids"] + s["token_ids"] for r, s in flat]
    starts = [len(r["prompt_token_ids"]) for r, s in flat]
    glp = [s["logprobs"] for r, s in flat]
    res["gen"] = [dict(tokens=s["num_tokens"], finish=s["finish_reason"]) for r, s in flat]
    sc1 = S.score(seqs, starts)
    sc2 = S.score(seqs, starts)
    res["score_secs"] = round(S.last_score["seconds"], 2)
    res["repeat_bit_identical"] = all(np.array_equal(x["logprobs"], y["logprobs"]) for x, y in zip(sc1, sc2))
    res["n_tokens_match"] = all(x["n_tokens"] == len(l) for x, l in zip(sc1, glp))
    res["score_vs_generate_sampled"] = diffstats(glp, [x["logprobs"] for x in sc1])
    gb = S.generate(prompts[:4], max_tokens=1024, logprobs=0, sampling={"temperature": 0.0}, reasoning_effort="medium")
    gflat = [(r, s) for r in gb for s in r["samples"]]
    gsc = S.score([r["prompt_token_ids"] + s["token_ids"] for r, s in gflat], [len(r["prompt_token_ids"]) for r, s in gflat])
    res["score_vs_generate_greedy"] = diffstats([s["logprobs"] for r, s in gflat], [x["logprobs"] for x in gsc])
    fv = S.score(seqs, starts, final_via_sampler=True)
    res["final_via_sampler_last_abs_diff"] = [float(abs(x["logprobs"][-1] - y["logprobs"][-1])) for x, y in zip(sc1, fv)]
    # an edit moves score(), its revert restores it bit for bit
    e = {"op": "zero_block", "layer": S.cfg.L // 2, "parts": ["mlp"]}
    h = S.apply(e)
    sc3 = S.score(seqs, starts)
    S.revert(h)
    sc4 = S.score(seqs, starts)
    res["edit_moves_score"] = diffstats([x["logprobs"] for x in sc1], [x["logprobs"] for x in sc3])
    res["revert_restores_score_bit_identical"] = all(np.array_equal(x["logprobs"], y["logprobs"]) for x, y in zip(sc1, sc4))
    sv = res["score_vs_generate_sampled"]
    res["PASS"] = bool(res["repeat_bit_identical"] and res["n_tokens_match"] and sv["p50"] < 1e-3 and
                       res["revert_restores_score_bit_identical"] and not res["edit_moves_score"]["identical"])
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--pool", default=os.path.expanduser("~/selfedit/bundle/rsi/pool.jsonl"))
    ap.add_argument("--out", default=os.environ.get("SELFEDIT_ENGINE_OUT", "/mnt/ssd/test/selfedit/engine_xf"))
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log("args:", vars(a))
    from surgery import Surgeon
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                kv_cache_dtype=a.kv_cache_dtype)
    ck = Ckpt(a.model)
    fam = S.family
    path = f"{a.out}/core_xf_{fam}_tp{a.tp}_{STAMP}.json"
    R = dict(phase=f"core_xf_{fam}_tp{a.tp}", family=fam, model=a.model, load_secs=round(S.load_seconds, 1),
             engine_args=S.engine_args, default_sampling=S.default_sampling, n_specs=len(S.layout.specs),
             buffers=len(S.layout.buffers), n_aliases=len(S.alias_params), unmapped_params=S.unmapped_params,
             unmapped_not_alias=S.unmapped_real,
             alias_examples=dict(sorted(S.alias_params.items())[:6]), declared_ties=S.layout.aliases)
    prompts = [json.loads(l)["prompt"] for l in open(a.pool)][:8]
    try:
        R["unmapped_only_aliases"] = not S.unmapped_real
        R["declared_ties_confirmed"] = all(S.alias_params.get(a) == t for a, t in S.layout.aliases.items())
        b1, b2, b3 = probe(S), probe(S), probe(S)
        R["determinism"], R["determinism_3"] = cmp(b1, b2), cmp(b1, b3)
        R["batch_composition_prompt0_alone_vs_in_batch"] = cmp(b1[:1], probe(S, prompts=TS.PROBES[:1]))
        R["probe_text"] = [x["text"] for x in b1]
        log("determinism:", R["determinism"], R["determinism_3"], "probe:", R["probe_text"])
        t0 = time.time()
        lay = S.verify_layout()
        R["layout"] = summarize_layout(lay)
        R["layout_secs"] = round(time.time() - t0, 1)
        R["layout_problems"] = layout_ok(S, lay)
        R["layout_PASS"] = not R["layout_problems"] and all(x["checked"] > 0 for x in lay)
        log("layout:", R["layout"], "problems:", R["layout_problems"], f"{R['layout_secs']}s")
        R["derived_scan"] = S.scan_tensors()[0]["tensors"]
        R["param_dtypes"] = S.param_dtypes
        t0 = time.time()
        fp0 = S.model_fingerprint()
        R["fingerprint_secs"] = round(time.time() - t0, 2)
        full_fp = R["fingerprint_secs"] < 15
        log(f"fingerprint {R['fingerprint_secs']}s over {len(fp0)} entries; per-test={full_fp}")
        rng = np.random.default_rng(1)
        tests, zeros, eq = (tests_gemma4 if fam == "gemma4" else tests_olmo3)(S, rng, ck)
        recs = []
        for name, e, *x in tests:
            try:
                recs.append(op_test_xf(S, ck, name, e, b1, fp0, x[0] if x else "change", full_fp))
            except Exception as ex:
                traceback.print_exc()
                recs.append(dict(name=name, PASS=False, error=repr(ex)))
                S.revert_all()
        for name, e in zeros:
            try:
                recs.append(op_test_xf(S, ck, name, e, b1, fp0, "zero", full_fp))
            except Exception as ex:
                traceback.print_exc()
                recs.append(dict(name=name, PASS=False, error=repr(ex)))
                S.revert_all()
        R["op_tests"] = recs
        R["equivalence"] = equiv_tests(S, b1, eq)
        R["residual_semantics"] = residual_semantics(S, np.random.default_rng(3))
        log("residual semantics:", R["residual_semantics"])
        R["final_full_fingerprint_equal"] = S.model_fingerprint() == fp0
        R["final_probe_identical"] = cmp(b1, probe(S))["identical"]
        if "chat" not in a.skip:
            R["chat"] = chat_checks(S, prompts)
            log("chat:", json.dumps(R["chat"], indent=1)[:6000])
        if "score" not in a.skip:
            R["score"] = score_checks(S, prompts)
            log("score:", json.dumps(R["score"], indent=1))
        R["gpu_memory"] = S.gpu_memory()
    finally:
        json.dump(R, open(path, "w"), indent=1, default=str)
        log("results ->", path)
    allr = R.get("op_tests", []) + R.get("equivalence", [])
    log(f"SUMMARY {fam} tp{a.tp}: op/equiv {sum(r['PASS'] for r in allr)}/{len(allr)}; layout {R.get('layout_PASS')}; "
        f"residual {R.get('residual_semantics', {}).get('PASS')}; score {R.get('score', {}).get('PASS')}; "
        f"determinism {R.get('determinism', {}).get('identical')}; final fp {R.get('final_full_fingerprint_equal')}")
    for r in allr:
        if not r["PASS"]:
            log("FAILED:", r["name"], r.get("error", ""))


if __name__ == "__main__":
    main()
