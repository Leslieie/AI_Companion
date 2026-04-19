"""Fix SFT data files by removing trailing user messages.

Ensures every conversation ends with an assistant turn.
Overwrites the four processed files in-place.

Usage:
    python -m src.training.fix_sft_data
"""

from pathlib import Path

from src.utils.io import load_jsonl, save_jsonl

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

TARGETS = [
    "train_sft.jsonl",
    "val_sft.jsonl",
    "train_sft_clean.jsonl",
    "val_sft_clean.jsonl",
]


def fix_file(path: Path) -> None:
    entries = load_jsonl(str(path))
    original = len(entries)
    truncated = 0
    dropped = 0
    fixed: list[dict] = []

    for entry in entries:
        msgs = entry["messages"]
        if msgs[-1]["role"] == "user":
            msgs = msgs[:-1]
            if len(msgs) < 3:
                dropped += 1
                continue
            truncated += 1
            entry["messages"] = msgs
        fixed.append(entry)

    save_jsonl(fixed, str(path))
    print(
        f"  {path.name}: {original} -> {len(fixed)}"
        f"  (truncated {truncated}, dropped {dropped})"
    )


def main() -> None:
    print("Fixing trailing user messages in SFT data files:\n")
    for name in TARGETS:
        path = PROCESSED_DIR / name
        if not path.exists():
            print(f"  {name}: not found, skipping")
            continue
        fix_file(path)
    print("\nDone.")


if __name__ == "__main__":
    main()
