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
    """跑一次 prefill on input[:seq_len]，返回 dict[layer_idx] = (input_hidden, attn_output)
    两个都是 (1, seq_len, H) shape。"""
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
            captures.setdefault(idx, {})["input"] = hidden.detach().clone()
        return pre_hook

    def make_post_hook(idx):
        def post_hook(module, args, kwargs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            captures.setdefault(idx, {})["attn_out"] = attn_out.detach().clone()
        return post_hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(make_post_hook(i), with_kwargs=True))

    try:
        with torch.no_grad():
            _ = model(enc)
    finally:
        for h in handles:
            h.remove()

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

    print(f"\n=== Layer-by-layer max diff over rows 0..46 (overlap) ===")
    print(f"  {'layer':>5} {'input_hidden':>15} {'input_argmax_row':>20} "
          f"{'attn_output':>15} {'attn_argmax_row':>20}")
    for i in sorted(cap_47.keys()):
        ih_47 = cap_47[i]["input"][:, :47, :]   # (1, 47, H)
        ih_48 = cap_48[i]["input"][:, :47, :]   # (1, 47, H)  overlap
        ih_diff_per_row = (ih_47 - ih_48).abs().reshape(-1, 47, ih_47.shape[-1]).amax(dim=(0, 2))
        ih_max = ih_diff_per_row.max().item()
        ih_argmax = int(ih_diff_per_row.argmax().item()) if ih_max > 0 else -1

        ao_47 = cap_47[i]["attn_out"][:, :47, :]
        ao_48 = cap_48[i]["attn_out"][:, :47, :]
        ao_diff_per_row = (ao_47 - ao_48).abs().reshape(-1, 47, ao_47.shape[-1]).amax(dim=(0, 2))
        ao_max = ao_diff_per_row.max().item()
        ao_argmax = int(ao_diff_per_row.argmax().item()) if ao_max > 0 else -1

        print(f"  {i:>5} {ih_max:>15.3e} {f'row={ih_argmax}':>20} "
              f"{ao_max:>15.3e} {f'row={ao_argmax}':>20}")


if __name__ == "__main__":
    main()
