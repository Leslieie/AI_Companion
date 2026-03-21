"""Prepare and format datasets for SFT training.

This module handles:
- Loading raw dialogue datasets (EmpatheticDialogues, Persona-Chat, DailyDialog)
- Converting them into the standard chat message format
- Merging with self-written companion samples
- Outputting a unified JSONL file for training

Output format per line:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

from pathlib import Path


def load_raw_data(raw_dir: str) -> list[dict]:
    """Load raw dialogue data from the raw directory.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        List of raw dialogue entries.
    """
    raise NotImplementedError("Implement after dataset selection is finalized.")


def convert_to_chat_format(raw_entries: list[dict]) -> list[dict]:
    """Convert raw dialogue entries to the standard chat message format.

    Args:
        raw_entries: List of raw dialogue dicts.

    Returns:
        List of dicts with 'messages' key in chat format.
    """
    raise NotImplementedError("Implement after data format is agreed upon.")


def save_jsonl(entries: list[dict], output_path: str) -> None:
    """Save a list of dicts as a JSONL file.

    Args:
        entries: List of training samples.
        output_path: Path to the output JSONL file.
    """
    import json

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
