"""Evaluation CLI for comparing 7B companion model variants.

Modes:
- plain:         7B base model, minimal system prompt, no state/memory/policy.
- stateful:      7B base model with architecture modules (state + memory + policy).
- stateful_sft:  7B LoRA-adapted model with architecture modules.
- all:           run plain, stateful, and stateful_sft sequentially, then
                 print a side-by-side comparison table.

After generation, prints perplexity, emotion_appropriateness, distinct_1,
and distinct_2, and saves both metrics and per-sample generated responses
to outputs/eval/results_{mode}_7b.json.

Usage:
    python -m src.training.evaluate_7b \\
        --model_path Qwen/Qwen2.5-7B-Instruct \\
        --test_data data/splits/test.jsonl \\
        --mode plain

    python -m src.training.evaluate_7b --mode all
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
from src.utils.metrics import compute_perplexity, distinct_n, response_appropriateness


MODEL_NAME_7B = "Qwen/Qwen2.5-7B-Instruct"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "eval"
DEFAULT_TEST_DATA = str(REPO_ROOT / "data" / "splits" / "test.jsonl")
DEFAULT_SFT_PATH = str(REPO_ROOT / "outputs" / "checkpoints" / "sft_7b_run_01" / "final")

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

        tokenizer, base_model = load_model(MODEL_NAME_7B)
        model = PeftModel.from_pretrained(base_model, model_path)
        return tokenizer, model
    return load_model(model_path)


def generate_responses(
    tokenizer,
    model,
    samples: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Generate one response per test sample with the requested pipeline."""
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


def _run_single_mode(
    mode: str,
    model_path: str,
    test_data: str,
    skip_perplexity: bool = False,
) -> dict[str, Any]:
    """Run evaluation for one mode, save results, return metrics dict."""
    samples = _load_test_samples(test_data)
    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode} [7B]  ({len(samples)} samples)")
    print(f"{'=' * 60}")

    tokenizer, model = _load_for_mode(model_path, mode)
    records = generate_responses(tokenizer, model, samples, mode)

    user_texts = [r["user"] for r in records]
    responses_only = [r["response"] for r in records]

    print(f"\n  Computing response appropriateness (GoEmotions) ...")
    approp_score, breakdown = response_appropriateness(user_texts, responses_only)

    print(f"  Emotion breakdown (user_emo -> response_emo: count):")
    for u_emo in sorted(breakdown):
        pairs = ", ".join(
            f"{r_emo}:{c}" for r_emo, c in sorted(
                breakdown[u_emo].items(), key=lambda x: -x[1],
            )
        )
        print(f"    {u_emo:<20} -> {pairs}")

    d1 = distinct_n(responses_only, n=1)
    d2 = distinct_n(responses_only, n=2)
    ppl: float | None = None
    if not skip_perplexity:
        ppl = compute_perplexity(model, tokenizer, test_data)

    metrics: dict[str, Any] = {
        "mode": mode,
        "model_path": model_path,
        "num_samples": len(records),
        "response_appropriateness": approp_score,
        "distinct_1": d1,
        "distinct_2": d2,
        "perplexity": ppl,
    }

    print(f"\n--- {mode} [7B] results ---")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"results_{mode}_7b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "responses": records, "breakdown": breakdown},
            f, indent=2, ensure_ascii=False,
        )
    print(f"  Saved to {out_path}")

    import gc, torch
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def _print_comparison(all_metrics: list[dict[str, Any]]) -> None:
    """Print a side-by-side comparison table."""
    labels = [f"{m['mode']} (7B)" for m in all_metrics]
    header = f"{'Metric':<26}| " + " | ".join(f"{l:>18}" for l in labels)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  COMPARISON TABLE (7B)")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    rows = [
        ("Perplexity", "perplexity"),
        ("Response Appropriateness", "response_appropriateness"),
        ("Distinct-1", "distinct_1"),
        ("Distinct-2", "distinct_2"),
        ("Num Samples", "num_samples"),
    ]
    for label, key in rows:
        vals: list[str] = []
        for m in all_metrics:
            v = m.get(key)
            if v is None:
                vals.append(f"{'N/A':>18}")
            elif isinstance(v, int):
                vals.append(f"{v:>18d}")
            else:
                vals.append(f"{v:>18.4f}")
        print(f"{label:<26}| " + " | ".join(vals))

    print(f"{'=' * len(header)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 7B companion model variants.")
    parser.add_argument(
        "--model_path", default=None,
        help="Model id / path (or LoRA adapter path when --mode=stateful_sft).",
    )
    parser.add_argument(
        "--test_data", default=DEFAULT_TEST_DATA,
        help="Path to the test JSONL file.",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["plain", "stateful", "stateful_sft", "all"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--skip_perplexity", action="store_true",
        help="Skip perplexity (useful before SFT is available).",
    )
    args = parser.parse_args()

    if args.mode == "all":
        sft_path = args.model_path or DEFAULT_SFT_PATH
        all_metrics: list[dict[str, Any]] = []
        for mode in ("plain", "stateful", "stateful_sft"):
            if mode == "stateful_sft":
                mp = sft_path
            else:
                mp = MODEL_NAME_7B
            m = _run_single_mode(mode, mp, args.test_data, args.skip_perplexity)
            all_metrics.append(m)
        _print_comparison(all_metrics)
    else:
        if args.model_path is None:
            if args.mode == "stateful_sft":
                args.model_path = DEFAULT_SFT_PATH
            else:
                args.model_path = MODEL_NAME_7B
        _run_single_mode(
            args.mode, args.model_path, args.test_data, args.skip_perplexity,
        )


if __name__ == "__main__":
    main()
