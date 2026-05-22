"""跑两次纯 prefill (Lq=47 vs Lq=48)，逐层逐位置比较 hidden_states，找第一个 M-leak。

之前的诊断引入了 decode 路径（past_kv + cat），可能 muddy 了因果链。这里完全
不用 decode，只比两次独立的 prefill 中所有共有位置 (0..46) 上每层的 hidden 是否
bit-equal。如果模型本身是 M-invariant 的，两次 prefill 在 row 0..46 应完全相同。

捕获每层 self_attn 的 input hidden_states 全行（不只 row 47）以及 attn 输出全行。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


MODEL_ID = "Qwen/Qwen3-1.7B"


def run_prefill_and_capture(model, tok, seq_len: int):
    """跑一次 prefill on input[:seq_len]，返回 dict[layer_idx] = (input_hidden, attn_output, attn_mask)
    """
    prompt = "You are a helpful math assistant.\n\nQuestion: What is 13 * 47?\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to("cuda").input_ids
    if enc.shape[1] >= seq_len:
        enc = enc[:, :seq_len]
    else:
        pad = torch.full((1, seq_len - enc.shape[1]), tok.pad_token_id or tok.eos_token_id,
                         device="cuda", dtype=enc.dtype)
        enc = torch.cat([enc, pad], dim=1)
    assert enc.shape[1] == seq_len

    captures = {}

    def make_pre_hook(idx):
        def pre_hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            attn_mask = kwargs.get("attention_mask", None)
            captures.setdefault(idx, {})["input"] = hidden.detach().clone()
            captures.setdefault(idx, {})["attn_mask"] = (
                None if attn_mask is None else attn_mask.detach().clone()
            )
        return pre_hook

    def make_post_hook(idx):
        def post_hook(module, args, kwargs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            captures.setdefault(idx, {})["attn_out"] = attn_out.detach().clone()
        return post_hook

    # 钩 sdpa wrapper：捕获 post-rotary Q 和 K（key/value 参数）
    from ops_extension import sdpa as sdpa_mod
    orig = sdpa_mod._sdpa_attention_forward_invariant
    counter = {"v": 0}
    layers_per = len(model.model.layers)

    def wrapped(module, query, key, value, attention_mask, **kw):
        idx = counter["v"] % layers_per
        counter["v"] += 1
        captures.setdefault(idx, {})["Q_post_rotary"] = query.detach().clone()
        captures.setdefault(idx, {})["K_post_rotary"] = key.detach().clone()
        captures.setdefault(idx, {})["V_post_proj"] = value.detach().clone()
        return orig(module, query, key, value, attention_mask, **kw)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    saved = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(make_post_hook(i), with_kwargs=True))

    try:
        with torch.no_grad():
            counter["v"] = 0
            _ = model(enc)
    finally:
        for h in handles:
            h.remove()
        ALL_ATTENTION_FUNCTIONS["sdpa"] = saved

    return captures


def main():
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model with attn_implementation='sdpa'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    print(f"\n=== Running pure prefill Lq=47 ===")
    cap_47 = run_prefill_and_capture(model, tok, 47)
    print(f"\n=== Running pure prefill Lq=48 ===")
    cap_48 = run_prefill_and_capture(model, tok, 48)

    print(f"\n=== Layer 0 attention_mask comparison ===")
    m47 = cap_47[0].get("attn_mask")
    m48 = cap_48[0].get("attn_mask")
    print(f"  47-prefill layer 0 mask: {None if m47 is None else f'shape={tuple(m47.shape)} dtype={m47.dtype}'}")
    print(f"  48-prefill layer 0 mask: {None if m48 is None else f'shape={tuple(m48.shape)} dtype={m48.dtype}'}")

    print(f"\n=== Layer-by-layer diff over rows 0..46 (Q/K are post-rotary) ===")
    print(f"  {'layer':>5} {'in_hidden':>10} {'Q':>10} {'K':>10} {'attn_out':>10} {'Q_row':>8} {'K_row':>8} {'ao_row':>8}")
    for i in sorted(cap_47.keys()):
        def row_diff(a, b):
            d = (a - b).abs()
            d_per_row = d.reshape(-1, a.shape[-2], a.shape[-1]).amax(dim=(0, 2))
            mx = d_per_row.max().item()
            argmx = int(d_per_row.argmax().item()) if mx > 0 else -1
            return mx, argmx

        ih_47 = cap_47[i]["input"][:, :47, :]
        ih_48 = cap_48[i]["input"][:, :47, :]
        ih_max, _ = row_diff(ih_47, ih_48)

        q_47 = cap_47[i].get("Q_post_rotary")
        q_48 = cap_48[i].get("Q_post_rotary")
        if q_47 is not None and q_48 is not None:
            # shape (1, H_q, Lq, D)
            q_47s = q_47[:, :, :47, :]
            q_48s = q_48[:, :, :47, :]
            q_max, q_argmx = row_diff(q_47s, q_48s)
        else:
            q_max, q_argmx = float("nan"), -1

        k_47 = cap_47[i].get("K_post_rotary")
        k_48 = cap_48[i].get("K_post_rotary")
        if k_47 is not None and k_48 is not None:
            k_47s = k_47[:, :, :47, :]
            k_48s = k_48[:, :, :47, :]
            k_max, k_argmx = row_diff(k_47s, k_48s)
        else:
            k_max, k_argmx = float("nan"), -1

        ao_47 = cap_47[i]["attn_out"][:, :47, :]
        ao_48 = cap_48[i]["attn_out"][:, :47, :]
        ao_max, ao_argmx = row_diff(ao_47, ao_48)

        print(f"  {i:>5} {ih_max:>10.2e} {q_max:>10.2e} {k_max:>10.2e} {ao_max:>10.2e} "
              f"{q_argmx:>8d} {k_argmx:>8d} {ao_argmx:>8d}")

    # --- Decisive isolated reproducer at layer 4 ---
    # 拿 48-prefill 的 Q/K/V/scaling，slice 出 47 vs 48 长度，调 sdpa_batch_invariant,
    # 比较 row 27。如果 diff > 0 → minimal reproducer 拿到了。
    from ops_extension.sdpa import sdpa_batch_invariant
    leak_layer = 4
    Q_full = cap_48[leak_layer]["Q_post_rotary"]   # (1, H_q, 48, D)
    K_full = cap_48[leak_layer]["K_post_rotary"]   # (1, H_kv, 48, D)
    V_full = cap_48[leak_layer]["V_post_proj"]
    print(f"\n=== Isolated reproducer with REAL captured Q/K/V at layer {leak_layer} ===")
    print(f"  Q shape: {tuple(Q_full.shape)}, K shape: {tuple(K_full.shape)}, V shape: {tuple(V_full.shape)}")
    with torch.no_grad():
        # 用真实 Q/K/V (48-prefill 的值) 调 sdpa_batch_invariant，分别按 47 和 48 长度切
        out_47 = sdpa_batch_invariant(
            Q_full[:, :, :47, :], K_full[:, :, :47, :], V_full[:, :, :47, :], is_causal=True,
        )
        out_48 = sdpa_batch_invariant(
            Q_full, K_full, V_full, is_causal=True,
        )
    diff_per_row = (out_47 - out_48[:, :, :47, :]).abs().reshape(-1, 47, out_47.shape[-1]).amax(dim=(0, 2))
    isolated_max = diff_per_row.max().item()
    isolated_argmax = int(diff_per_row.argmax().item()) if isolated_max > 0 else -1
    print(f"  isolated diff: max={isolated_max:.3e} at row={isolated_argmax}")
    if isolated_max == 0.0:
        # 真实输入下也是 invariant => attention compute 不背锅，那 V 可能在两次 prefill 不一致
        v_47 = cap_47[leak_layer]["V_post_proj"][:, :, :47, :]
        v_48 = cap_48[leak_layer]["V_post_proj"][:, :, :47, :]
        v_diff_per_row = (v_47 - v_48).abs().reshape(-1, 47, v_47.shape[-1]).amax(dim=(0, 2))
        v_max = v_diff_per_row.max().item()
        v_argmax = int(v_diff_per_row.argmax().item()) if v_max > 0 else -1
        print(f"\n  → isolated invariant; check V_post_proj diff between prefills: "
              f"max={v_max:.3e} at row={v_argmax}")


if __name__ == "__main__":
    main()
