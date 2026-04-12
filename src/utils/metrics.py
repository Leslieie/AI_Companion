"""Evaluation metrics for companion response quality.

Metrics implemented:
- compute_perplexity: teacher-forced average per-token perplexity over a
  test JSONL file, using the tokenizer's chat template.
- emotion_appropriateness: policy-matching accuracy between the ground-truth
  `policy` field and the policy implied by the generated response's tone.
- distinct_n: unique n-gram ratio across a set of generated texts (n=1, 2).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

import torch


def compute_perplexity(model, tokenizer, test_data_path: str) -> float:
    """Compute average per-token perplexity on a test JSONL file.

    Applies the tokenizer's chat template to each sample's `messages`
    and evaluates cross-entropy via teacher forcing with no gradients.

    Args:
        model: A loaded Hugging Face causal LM (or PEFT-wrapped LM).
        tokenizer: The matching tokenizer.
        test_data_path: Path to the test JSONL file (one sample per line,
            each containing a `messages` list of {role, content} dicts).

    Returns:
        Average per-token perplexity (float). Returns inf if no tokens
        were scored.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with open(test_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample.get("messages")
            if not messages:
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            n_tokens = inputs.input_ids.shape[1]
            if n_tokens < 2:
                continue

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)

            # Hugging Face returns mean NLL over (n_tokens - 1) predictions.
            n_pred = n_tokens - 1
            total_loss += outputs.loss.item() * n_pred
            total_tokens += n_pred

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def emotion_appropriateness(
    model,
    tokenizer,
    test_data_path: str,
    use_stateful: bool = False,
) -> float:
    """Score how often the generated response's tone matches the expected policy.

    For each test sample, generates a reply with the requested pipeline,
    classifies its emotion, maps that to a policy with `select_policy`,
    and compares against the ground-truth `policy` field in the sample.

    Args:
        model: A loaded causal LM.
        tokenizer: The matching tokenizer.
        test_data_path: Path to the test JSONL file.
        use_stateful: If True, uses StateTracker, MemoryStore, and
            build_prompt(); otherwise uses a minimal system prompt.

    Returns:
        Accuracy (correct / total) in [0.0, 1.0].
    """
    from src.inference.generate import generate_response
    from src.inference.prompt_builder import build_prompt
    from src.modules.emotion_classifier import classify_emotion
    from src.modules.memory_store import MemoryStore
    from src.modules.policy_selector import select_policy
    from src.modules.state_tracker import StateTracker

    default_state: dict[str, int | str] = {
        "affection": 50,
        "trust": 50,
        "intimacy": 50,
        "mood": "neutral",
        "energy": 70,
    }

    correct = 0
    total = 0

    with open(test_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            expected_policy = sample.get("policy")
            if expected_policy is None:
                continue

            messages = sample.get("messages", [])
            history = [m for m in messages if m["role"] != "system"]
            if history and history[-1]["role"] == "assistant":
                history = history[:-1]
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                None,
            )
            if last_user is None:
                continue

            if use_stateful:
                tracker = StateTracker()
                memory = MemoryStore()
                state: dict[str, int | str] = tracker.get_state()
                for turn_idx, m in enumerate(history):
                    if m["role"] == "user":
                        state = tracker.update(m["content"])
                        memory.extract_and_store(m["content"], turn_idx)
                emotion = classify_emotion(last_user)
                policy = select_policy(emotion, state)
                memories = memory.retrieve(last_user, top_k=3)
                prompt_messages = build_prompt(history, state, memories, policy)
            else:
                state = dict(default_state)
                prompt_messages = [
                    {"role": "system", "content": "You are a warm AI companion."},
                    {"role": "user", "content": last_user},
                ]

            response = generate_response(tokenizer, model, prompt_messages)
            response_emotion = classify_emotion(response)
            predicted_policy = select_policy(response_emotion, state)

            if predicted_policy == expected_policy:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def distinct_n(texts: Iterable[str], n: int = 1) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams.

    Args:
        texts: Iterable of response strings.
        n: Size of the n-gram (supports 1 or 2).

    Returns:
        unique_ngrams / total_ngrams in [0.0, 1.0]; 0.0 if no n-grams.
    """
    if n not in (1, 2):
        raise ValueError(f"distinct_n supports n=1 or n=2, got {n}")
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)
