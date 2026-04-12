"""Evaluation CLI for comparing companion model variants.

Modes:
- plain:         base model, minimal system prompt, no state/memory/policy.
- stateful:      base model with architecture modules (state + memory + policy).
- stateful_sft:  LoRA-adapted model with architecture modules.

After generation, prints perplexity, emotion_appropriateness, distinct_1,
and distinct_2, and saves both metrics and per-sample generated responses
to outputs/eval/results_{mode}.json.

Usage:
    python -m src.training.evaluate \\
        --model_path Qwen/Qwen2.5-1.5B-Instruct \\
        --test_data data/annotations/reference_samples.jsonl \\
        --mode plain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.generate import generate_response, load_model
from src.inference.prompt_builder import build_prompt
from src.modules.emotion_classifier import classify_emotion
from src.modules.memory_store import MemoryStore
from src.modules.policy_selector import select_policy
from src.modules.state_tracker import StateTracker
from src.utils.metrics import compute_perplexity, distinct_n


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "eval"

PLAIN_SYSTEM_PROMPT = "You are a warm AI companion."
DEFAULT_STATE: dict[str, int | str] = {
    "affection": 50,
    "trust": 50,
    "intimacy": 50,
    "mood": "neutral",
    "energy": 70,
}


def _load_test_samples(test_data_path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of sample dicts."""
    samples: list[dict[str, Any]] = []
    with open(test_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _extract_history(
    sample: dict[str, Any],
) -> tuple[list[dict[str, str]], str | None]:
    """Return (conversation_history_without_system_or_final_assistant, last_user_message)."""
    messages = sample.get("messages", [])
    history = [m for m in messages if m["role"] != "system"]
    if history and history[-1]["role"] == "assistant":
        history = history[:-1]
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"),
        None,
    )
    return history, last_user


def _load_for_mode(model_path: str, mode: str) -> tuple:
    """Load tokenizer and model; wrap in LoRA adapter for stateful_sft."""
    if mode == "stateful_sft":
        from peft import PeftModel

        from src.inference.generate import MODEL_NAME

        tokenizer, base_model = load_model(MODEL_NAME)
        model = PeftModel.from_pretrained(base_model, model_path)
        return tokenizer, model
    return load_model(model_path)


def generate_responses(
    tokenizer,
    model,
    samples: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Generate one response per test sample with the requested pipeline.

    Each record captures both the response and enough metadata to compute
    policy-matching accuracy downstream.
    """
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        history, last_user = _extract_history(sample)
        if last_user is None:
            continue

        expected_policy = sample.get("policy", "neutral")

        if mode == "plain":
            prompt_messages = [
                {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": last_user},
            ]
            state: dict[str, int | str] = dict(DEFAULT_STATE)
            policy_in_prompt: str | None = None
            memories: list[str] = []
        else:
            tracker = StateTracker()
            memory = MemoryStore()
            state = tracker.get_state()
            for turn_idx, m in enumerate(history):
                if m["role"] == "user":
                    state = tracker.update(m["content"])
                    memory.extract_and_store(m["content"], turn_idx)
            emotion = classify_emotion(last_user)
            policy_in_prompt = select_policy(emotion, state)
            memories = memory.retrieve(last_user, top_k=3)
            prompt_messages = build_prompt(
                history, state, memories, policy_in_prompt,
            )

        response = generate_response(tokenizer, model, prompt_messages)
        response_emotion = classify_emotion(response)
        predicted_policy = select_policy(response_emotion, state)

        records.append({
            "index": idx,
            "user": last_user,
            "response": response,
            "expected_policy": expected_policy,
            "predicted_policy": predicted_policy,
            "response_emotion": response_emotion,
            "state": state,
            "policy_in_prompt": policy_in_prompt,
            "memories": memories,
        })
        print(f"[{idx + 1}/{len(samples)}] expected={expected_policy} "
              f"predicted={predicted_policy} emotion={response_emotion}")

    return records


def _emotion_accuracy(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for r in records if r["predicted_policy"] == r["expected_policy"])
    return correct / len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate companion model variants.")
    parser.add_argument(
        "--model_path", required=True,
        help="Model id / path (or LoRA adapter path when --mode=stateful_sft).",
    )
    parser.add_argument(
        "--test_data", required=True,
        help="Path to the test JSONL file.",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["plain", "stateful", "stateful_sft"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--skip_perplexity", action="store_true",
        help="Skip perplexity (useful before SFT is available).",
    )
    args = parser.parse_args()

    samples = _load_test_samples(args.test_data)
    print(f"Loaded {len(samples)} test samples from {args.test_data}")

    tokenizer, model = _load_for_mode(args.model_path, args.mode)

    records = generate_responses(tokenizer, model, samples, args.mode)

    emo_acc = _emotion_accuracy(records)
    responses_only = [r["response"] for r in records]
    d1 = distinct_n(responses_only, n=1)
    d2 = distinct_n(responses_only, n=2)
    ppl: float | None = None
    if not args.skip_perplexity:
        ppl = compute_perplexity(model, tokenizer, args.test_data)

    metrics = {
        "mode": args.mode,
        "model_path": args.model_path,
        "num_samples": len(records),
        "emotion_appropriateness": emo_acc,
        "distinct_1": d1,
        "distinct_2": d2,
        "perplexity": ppl,
    }

    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"results_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "responses": records},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
