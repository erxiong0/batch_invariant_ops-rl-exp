"""逐层定位 prefill vs decode 第一个出现 diff 的环节。

所有单 op invariance 测试都通过（matmul / softmax / RMSNorm / rope / apply_rotary
都 bit-exact），但 model.generate (decode) vs per-step model.forward (prefill) 在
sdpa 模式下还是 16-aligned diff。说明组合起来某处 leak。

本脚本 hook 每层 self_attn 的：
  - 输入 hidden_states
  - position_embeddings (cos, sin)
  - 输出 (attn_output, attn_weights)

跑两次：
  A) 一次 prefill 48 个 token，捕获每层在 position 47 的 hidden 输入 / cos[47] / 输出
  B) 先 prefill 47 个 token + 1 步 decode，捕获 decode 步那一刻每层的 hidden / cos / 输出

逐层比较 A 在 position 47 vs B 在 single-position，找第一个 layer_idx 出现 diff > 0。
那一层 + 那个量（输入 / cos / 输出）就是泄漏点。
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
TARGET_LEN = 48   # 想对比 position 47 (16 的倍数 - 1，目标差异区)
N_PROMPT = 37     # 跟 test_attn_backend.py 一致


def run_and_capture(model, tok, target_len: int, mode: str):
    """mode='prefill': 一次性 forward target_len 个 token，捕获 layer 输入/输出 at row target_len-1
       mode='decode' : 先 prefill (target_len-1) 个 token + 1 步 decode，捕获 decode 步 layer 输入/输出

    新增：也捕获每层 attention 看到的完整 K_cache（含 0..target_len-2 历史 + 当前位置）。
    这就能直接对比 prefill 算的 K[0..46] vs decode 用的 K_cache[0..46]。
    """
    prompt = "You are a helpful math assistant.\n\nQuestion: What is 13 * 47?\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to("cuda").input_ids
    if enc.shape[1] > target_len:
        enc = enc[:, :target_len]
    elif enc.shape[1] < target_len:
        pad = torch.full((1, target_len - enc.shape[1]), tok.pad_token_id or tok.eos_token_id,
                         device="cuda", dtype=enc.dtype)
        enc = torch.cat([enc, pad], dim=1)
    assert enc.shape[1] == target_len

    captures = {"input_hidden": {}, "cos_at_target": {}, "attn_output": {}, "k_cache_full": {}}

    def make_pre_hook(layer_idx: int):
        def pre_hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pos_emb = kwargs.get("position_embeddings", None)
            cos = pos_emb[0] if pos_emb is not None else None
            if mode == "prefill":
                row = target_len - 1
                captures["input_hidden"][layer_idx] = hidden[:, row : row + 1, :].detach().clone()
                if cos is not None:
                    captures["cos_at_target"][layer_idx] = cos[:, row : row + 1, :].detach().clone()
            else:
                captures["input_hidden"][layer_idx] = hidden.detach().clone()
                if cos is not None:
                    captures["cos_at_target"][layer_idx] = cos.detach().clone()
        return pre_hook

    def make_post_hook(layer_idx: int):
        def post_hook(module, args, kwargs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            if mode == "prefill":
                row = target_len - 1
                captures["attn_output"][layer_idx] = attn_out[:, row : row + 1, :].detach().clone()
            else:
                captures["attn_output"][layer_idx] = attn_out.detach().clone()
        return post_hook

    # 钩住我们的 sdpa wrapper，从内部抓 K（key 参数已经包含 cache 拼接的完整 K[0..target_len-1]）
    from ops_extension import sdpa as sdpa_mod
    orig_wrapper = sdpa_mod._sdpa_attention_forward_invariant
    layer_idx_counter = {"v": 0}
    layers_per_forward = len(model.model.layers)

    def wrapped_with_capture(module, query, key, value, attention_mask, **kw):
        # layer_idx 从 0 循环到 N-1，每次 forward 重置
        idx = layer_idx_counter["v"] % layers_per_forward
        layer_idx_counter["v"] += 1
        captures["k_cache_full"][idx] = key.detach().clone()  # (B, H_kv, full_len, D)
        return orig_wrapper(module, query, key, value, attention_mask, **kw)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    saved_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped_with_capture

    handles = []
    for i, layer in enumerate(model.model.layers):
        h1 = layer.self_attn.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)
        h2 = layer.self_attn.register_forward_hook(make_post_hook(i), with_kwargs=True)
        handles.extend([h1, h2])

    try:
        with torch.no_grad():
            if mode == "prefill":
                layer_idx_counter["v"] = 0
                _ = model(enc)
            else:
                # decode: prefill first target_len-1, then decode 1 step
                layer_idx_counter["v"] = 0
                pre = model(enc[:, : target_len - 1], use_cache=True)
                # 重置 counter，让 decode 步的 K cache 覆盖 prefill 阶段的捕获
                layer_idx_counter["v"] = 0
                _ = model(enc[:, target_len - 1 : target_len],
                          past_key_values=pre.past_key_values, use_cache=True)
    finally:
        for h in handles:
            h.remove()
        ALL_ATTENTION_FUNCTIONS["sdpa"] = saved_sdpa

    return captures


def main():
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"\nLoading model with attn_implementation='sdpa'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    print(f"\n=== Running prefill (target_len={TARGET_LEN}) ===")
    cap_prefill = run_and_capture(model, tok, TARGET_LEN, mode="prefill")
    print(f"  captured layers: {sorted(cap_prefill['input_hidden'].keys())[:3]}...{sorted(cap_prefill['input_hidden'].keys())[-3:]}")

    print(f"\n=== Running decode (prefill {TARGET_LEN-1} + 1 step) ===")
    cap_decode = run_and_capture(model, tok, TARGET_LEN, mode="decode")
    print(f"  captured layers: {sorted(cap_decode['input_hidden'].keys())[:3]}...{sorted(cap_decode['input_hidden'].keys())[-3:]}")

    print(f"\n=== Layer-by-layer diff at position {TARGET_LEN-1} ===")
    print(f"  {'layer':>5} {'in_hidden':>12} {'cos':>12} {'attn_out':>12} {'K_cache_max':>12} {'K_cache_argmax':>15}")
    for i in sorted(cap_prefill["input_hidden"].keys()):
        ih_p = cap_prefill["input_hidden"][i]
        ih_d = cap_decode["input_hidden"][i]
        ih_diff = (ih_p - ih_d).abs().max().item()

        cos_p = cap_prefill["cos_at_target"].get(i)
        cos_d = cap_decode["cos_at_target"].get(i)
        cos_diff_str = "n/a" if cos_p is None or cos_d is None else f"{(cos_p - cos_d).abs().max().item():.3e}"

        ao_p = cap_prefill["attn_output"][i]
        ao_d = cap_decode["attn_output"][i]
        ao_diff = (ao_p - ao_d).abs().max().item()

        # K_cache 对比：两边都应该是 (B, H_kv, target_len, D)
        kp = cap_prefill["k_cache_full"].get(i)
        kd = cap_decode["k_cache_full"].get(i)
        if kp is not None and kd is not None and kp.shape == kd.shape:
            k_diff_per_pos = (kp - kd).abs().reshape(-1, kp.shape[-2], kp.shape[-1]).amax(dim=(0, 2))  # (target_len,)
            k_max = k_diff_per_pos.max().item()
            k_argmax = k_diff_per_pos.argmax().item()
            k_max_str = f"{k_max:.3e}"
            k_argmax_str = f"pos={k_argmax}"
        else:
            k_max_str = "shape_mismatch"
            k_argmax_str = f"{kp.shape if kp is not None else None} vs {kd.shape if kd is not None else None}"

        print(f"  {i:>5} {ih_diff:>12.3e} {cos_diff_str:>12} {ao_diff:>12.3e} {k_max_str:>12} {k_argmax_str:>15}")


if __name__ == "__main__":
    main()
