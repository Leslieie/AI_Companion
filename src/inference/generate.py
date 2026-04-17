"""Minimal inference script using Qwen2.5-1.5B-Instruct.

Run this script to verify the model loads and generates a response:
    python -m src.inference.generate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "model.yaml"
)

_FALLBACK_GEN_CONFIG: dict[str, Any] = {
    "max_new_tokens": 384,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
}


def _load_gen_config() -> dict[str, Any]:
    """Load the ``generation`` block from configs/model.yaml.

    Missing file or missing keys fall back to safe defaults so generation
    still works if the YAML is unavailable.
    """
    cfg: dict[str, Any] = dict(_FALLBACK_GEN_CONFIG)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return cfg
    gen = raw.get("generation") or {}
    for key in cfg:
        if key in gen and gen[key] is not None:
            cfg[key] = gen[key]
    return cfg


def _build_eos_ids(tokenizer: AutoTokenizer) -> list[int]:
    """Return the list of token ids that should terminate generation.

    Qwen2.5 uses ``<|im_end|>`` as the turn terminator. We include both the
    tokenizer's declared EOS and the explicit ``<|im_end|>`` id to be safe.
    """
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unk_id = tokenizer.unk_token_id
    if im_end_id is not None and im_end_id != unk_id:
        eos_ids.append(int(im_end_id))
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(eos_ids))


def load_model(model_name: str = MODEL_NAME) -> tuple:
    """Load tokenizer and model from Hugging Face or local cache.

    Args:
        model_name: Model identifier or local path.

    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def generate_response(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: list[dict[str, str]],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    do_sample: bool | None = None,
) -> str:
    """Generate a response given a list of chat messages.

    Generation parameters are loaded from ``configs/model.yaml``; any argument
    passed explicitly overrides the corresponding config value.

    Prints a stderr warning if the response appears to have been truncated
    (no EOS emitted and generated length equals ``max_new_tokens``).
    """
    cfg = _load_gen_config()
    if max_new_tokens is None:
        max_new_tokens = int(cfg["max_new_tokens"])
    if temperature is None:
        temperature = float(cfg["temperature"])
    if top_p is None:
        top_p = float(cfg["top_p"])
    if repetition_penalty is None:
        repetition_penalty = float(cfg["repetition_penalty"])
    if do_sample is None:
        do_sample = bool(cfg["do_sample"])

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    eos_ids = _build_eos_ids(tokenizer)
    pad_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else (eos_ids[0] if eos_ids else None)
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        eos_token_id=eos_ids if eos_ids else None,
        pad_token_id=pad_id,
    )

    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    num_generated = int(generated_ids.shape[1])
    last_token_id = (
        int(generated_ids[0, -1].item()) if num_generated > 0 else None
    )
    hit_cap = num_generated >= max_new_tokens
    emitted_eos = last_token_id in eos_ids if last_token_id is not None else False

    if hit_cap and not emitted_eos:
        print(
            "[warn] response may be truncated — hit max_new_tokens cap",
            file=sys.stderr,
        )

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True,
    )[0]
    return response


def main() -> None:
    """Run a single inference test."""
    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": "You are a warm and emotionally attentive AI companion.",
        },
        {
            "role": "user",
            "content": "I feel kind of overwhelmed today.",
        },
    ]

    response = generate_response(tokenizer, model, messages)
    print(response)


if __name__ == "__main__":
    main()
