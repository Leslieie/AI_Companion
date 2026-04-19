"""Create a held-out test set from public and team data.

Samples entries that do NOT appear in train or val, validates them,
and saves to data/splits/.

Usage:
    python -m src.training.create_test_set
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.utils.io import load_jsonl, save_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

PUBLIC_FILES = ["empathetic.jsonl", "personachat.jsonl", "dailydialog.jsonl"]
TEAM_FILES = [
    "hengkai_generated_320.jsonl",
    "intimacy_contrast_250.jsonl",
    "yls_cleaned_v2.jsonl",
    "pdd_cleaned_v3.jsonl",
]

PUBLIC_TARGET = 200
TEAM_TARGET = 50
NON_NEUTRAL_MIN_FRAC = 0.30


def _messages_key(entry: dict) -> str:
    return json.dumps(entry["messages"], sort_keys=True, ensure_ascii=False)


def _is_valid(entry: dict) -> bool:
    msgs = entry.get("messages")
    if not msgs or len(msgs) < 3:
        return False
    if msgs[-1]["role"] != "assistant":
        return False
    return True


def main() -> None:
    random.seed(42)

    # Build set of all messages already used in train/val
    used: set[str] = set()
    for name in ("train_sft.jsonl", "val_sft.jsonl"):
        path = PROCESSED_DIR / name
        if path.exists():
            for entry in load_jsonl(str(path)):
                used.add(_messages_key(entry))
    print(f"Train+val fingerprints: {len(used)}")

    # ── Public data: sample 200 not in train/val ──
    print("\n--- Public data ---")
    public_pool_non_neutral: list[dict] = []
    public_pool_neutral: list[dict] = []
    for fname in PUBLIC_FILES:
        path = PROCESSED_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping.")
            continue
        entries = load_jsonl(str(path))
        available = [e for e in entries if _is_valid(e) and _messages_key(e) not in used]
        nn = [e for e in available if e.get("emotion", "neutral") != "neutral"]
        ne = [e for e in available if e.get("emotion", "neutral") == "neutral"]
        print(f"  {fname}: {len(available)} available ({len(nn)} non-neutral, {len(ne)} neutral)")
        public_pool_non_neutral.extend(nn)
        public_pool_neutral.extend(ne)

    random.shuffle(public_pool_non_neutral)
    random.shuffle(public_pool_neutral)

    non_neutral_min = int(PUBLIC_TARGET * NON_NEUTRAL_MIN_FRAC)
    non_neutral_take = min(len(public_pool_non_neutral), max(non_neutral_min, PUBLIC_TARGET))
    public_selected = public_pool_non_neutral[:non_neutral_take]
    remaining = PUBLIC_TARGET - len(public_selected)
    if remaining > 0:
        public_selected.extend(public_pool_neutral[:remaining])

    public_selected = public_selected[:PUBLIC_TARGET]
    nn_count = sum(1 for e in public_selected if e.get("emotion", "neutral") != "neutral")
    print(f"\n  Selected {len(public_selected)} public samples "
          f"({nn_count} non-neutral, {len(public_selected) - nn_count} neutral)")

    # ── Team data: sample 50 not in train/val ──
    print("\n--- Team data ---")
    team_pool: list[dict] = []
    for fname in TEAM_FILES:
        path = PROCESSED_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping.")
            continue
        entries = load_jsonl(str(path))
        available = [e for e in entries if _is_valid(e) and _messages_key(e) not in used]
        print(f"  {fname}: {len(available)} available (not in train/val)")
        team_pool.extend(available)

    random.shuffle(team_pool)
    team_selected = team_pool[:TEAM_TARGET]
    print(f"\n  Selected {len(team_selected)} team samples")

    # ── Combine, shuffle, save ──
    test_set = public_selected + team_selected
    random.shuffle(test_set)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(test_set, str(SPLITS_DIR / "test.jsonl"))
    save_jsonl(
        [{"messages": e["messages"]} for e in test_set],
        str(SPLITS_DIR / "test_clean.jsonl"),
    )

    # ── Summary ──
    source_counts: dict[str, int] = defaultdict(int)
    emotion_counts: dict[str, int] = defaultdict(int)
    for e in test_set:
        source_counts[e.get("source", "unknown")] += 1
        emotion_counts[e.get("emotion", "unknown")] += 1

    print(f"\n{'=' * 50}")
    print(f"Test set: {len(test_set)} samples")
    print(f"  Saved to {SPLITS_DIR / 'test.jsonl'}")
    print(f"  Saved to {SPLITS_DIR / 'test_clean.jsonl'}")

    print(f"\n  Per source:")
    for src in sorted(source_counts):
        print(f"    {src:<30} {source_counts[src]:>5}")

    print(f"\n  Emotion distribution:")
    for emo in sorted(emotion_counts):
        pct = emotion_counts[emo] / len(test_set) * 100
        print(f"    {emo:<15} {emotion_counts[emo]:>5} ({pct:.1f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
