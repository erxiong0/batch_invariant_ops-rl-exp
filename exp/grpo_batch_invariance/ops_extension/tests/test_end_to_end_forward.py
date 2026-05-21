"""End-to-end Qwen3-1.7B forward batch invariance test.

启用 batch_invariant_ops + ops_extension 后，对相同输入 prompt:
  - batch=1 forward 拿 last-token logits
  - batch=8（含相同 prompt 在 slot 0）forward 拿 slot 0 last-token logits
两者必须 torch.equal（bit-equal）。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode
from ops_extension import (
    enable_extended_batch_invariant_mode,
    disable_extended_batch_invariant_mode,
)


MODEL_ID = "Qwen/Qwen3-1.7B"


@pytest.fixture(scope="module")
def model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    return model, tok


def _last_logits(model, input_ids, attn_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    # last non-pad token的 logits
    last_idx = attn_mask.sum(-1) - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    return out.logits[rows, last_idx]


def test_qwen_forward_bit_equal_with_full_invariance(model_and_tok):
    """启用完整 invariance 后 batch=1 与 batch=8 的 slot 0 logits bit-equal。"""
    model, tok = model_and_tok
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()
    try:
        target_prompt = "What is 13 * 47? Step by step:"
        other_prompts = [
            "Hello world.",
            "The capital of France is",
            "Solve for x: 2x + 5 = 11.",
            "List three primes.",
            "Write a haiku about GPUs.",
            "1 + 1 = ",
            "Translate 'cat' to French:",
        ]
        # batch=1
        enc1 = tok([target_prompt], return_tensors="pt", padding=True).to("cuda")
        logits1 = _last_logits(model, enc1.input_ids, enc1.attention_mask)
        # batch=8（target 在 slot 0）
        enc8 = tok([target_prompt] + other_prompts, return_tensors="pt", padding=True).to("cuda")
        logits8 = _last_logits(model, enc8.input_ids, enc8.attention_mask)

        assert torch.equal(logits1[0], logits8[0]), (
            f"forward not batch-invariant: "
            f"max diff = {(logits1[0] - logits8[0]).abs().max()}, "
            f"frac_diff = {((logits1[0] - logits8[0]).abs() > 0).float().mean()}"
        )
    finally:
        disable_extended_batch_invariant_mode()
        disable_batch_invariant_mode()


def test_qwen_forward_NOT_bit_equal_without_invariance(model_and_tok):
    """对照组：不启用 invariance，batch=1 与 batch=8 应有差异（mechanism sanity check）."""
    model, tok = model_and_tok
    target_prompt = "What is 13 * 47? Step by step:"
    other_prompts = ["Hello.", "Hi.", "Bonjour.", "你好.", "Ciao.", "Hola.", "Olá."]

    enc1 = tok([target_prompt], return_tensors="pt", padding=True).to("cuda")
    logits1 = _last_logits(model, enc1.input_ids, enc1.attention_mask)
    enc8 = tok([target_prompt] + other_prompts, return_tensors="pt", padding=True).to("cuda")
    logits8 = _last_logits(model, enc8.input_ids, enc8.attention_mask)

    diff = (logits1[0] - logits8[0]).abs().max().item()
    # baseline 下应有非零差异，否则机制不成立
    assert diff > 0, f"baseline batch invariance unexpectedly holds (diff={diff})"
