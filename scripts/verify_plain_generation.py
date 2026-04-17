"""Verify the plain-mode inference fix on 5 diverse prompts.

Runs the same pipeline used by src/inference/generate_interactive.py — it
calls load_model() and generate_response() from src/inference/generate.py —
and prints per-sample diagnostics:

- number of tokens actually generated
- whether the last token is an EOS token (<|im_end|> or tokenizer.eos_token_id)
- whether the cap was hit (generated_tokens == max_new_tokens)
- whether the response ends with sentence-final punctuation
- the effective max_new_tokens actually in force (reflects configs/model.yaml)

Usage:
    python -m scripts.verify_plain_generation
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.generate import (  # noqa: E402
    _build_eos_ids,
    _load_gen_config,
    generate_response,
    load_model,
)


SYSTEM_PROMPT = "You are a warm and emotionally attentive AI companion."

SAMPLE_PROMPTS: list[tuple[str, str]] = [
    ("short_en_emotional", "I feel kind of overwhelmed today."),
    (
        "long_en_emotional",
        "I just got off a two-hour call with my mom where she basically "
        "criticized every decision I've made this year — my job, my "
        "apartment, even the way I've been eating. I know she means well "
        "but I feel small right now and I don't know how to shake it.",
    ),
    ("short_zh_emotional", "我今天心情特别差,什么都不想做。"),
    (
        "long_zh_neutral",
        "最近我在读一本关于互联网早期历史的书,里面提到了 ARPANET 的设计"
        "决策,还有 TCP/IP 协议为什么会取代其他方案。虽然我不是学计算机的,"
        "但读起来还挺有意思的,你平时对这种科技史感兴趣吗?",
    ),
    ("short_en_neutral", "What's a good way to plan a free Saturday?"),
]

SENTENCE_FINAL_PUNCT = {".", "!", "?", "。", "!", "?", "…", "~", "”", "\""}


def _ends_with_sentence_punct(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in SENTENCE_FINAL_PUNCT


def main() -> None:
    cfg = _load_gen_config()
    print(f"Loaded generation config: {cfg}")
    effective_max_new_tokens = int(cfg["max_new_tokens"])
    print(f"Effective max_new_tokens = {effective_max_new_tokens}\n")

    print("Loading model...")
    tokenizer, model = load_model()
    eos_ids = _build_eos_ids(tokenizer)
    print(f"EOS token ids passed to generate(): {eos_ids}\n")

    any_truncated = False

    for label, user_message in SAMPLE_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Reproduce the tokenization path to measure generated length / last token.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=effective_max_new_tokens,
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            repetition_penalty=float(cfg["repetition_penalty"]),
            do_sample=bool(cfg["do_sample"]),
            eos_token_id=eos_ids if eos_ids else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        num_generated = int(generated_ids.shape[1])
        last_token_id = (
            int(generated_ids[0, -1].item()) if num_generated > 0 else None
        )
        hit_cap = num_generated >= effective_max_new_tokens
        emitted_eos = last_token_id in eos_ids if last_token_id is not None else False
        response = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0]
        ends_with_punct = _ends_with_sentence_punct(response)

        print(f"--- [{label}] ---")
        print(f"user: {user_message}")
        print(f"response: {response}")
        print(
            f"tokens_generated={num_generated} "
            f"last_token_id={last_token_id} "
            f"emitted_eos={emitted_eos} "
            f"hit_cap={hit_cap} "
            f"ends_with_sentence_punct={ends_with_punct}"
        )
        if hit_cap and not emitted_eos:
            any_truncated = True
        print()

    print("=== Summary ===")
    print(f"any_truncated={any_truncated}")
    if any_truncated:
        print(
            "At least one response hit the cap without emitting EOS — "
            "consider raising max_new_tokens further.",
            file=sys.stderr,
        )

    # Also exercise the public API to confirm it honours the YAML.
    print("\n=== Sanity check via generate_response() ===")
    response = generate_response(
        tokenizer,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Tell me something interesting."},
        ],
    )
    print(response)


if __name__ == "__main__":
    main()
