"""Prepare and format datasets for SFT training.

Downloads public dialogue datasets (EmpatheticDialogues, PersonaChat,
DailyDialog), converts them to the standard chat format, filters low-quality
samples, and produces training-ready JSONL files.

Usage:
    python -m src.training.prepare_sft_data download
    python -m src.training.prepare_sft_data merge
    python -m src.training.prepare_sft_data export
"""

import argparse
import csv
import io
import json
import random
import re
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

from src.utils.io import load_jsonl, load_yaml, save_jsonl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROMPTS_CONFIG = PROJECT_ROOT / "configs" / "prompts.yaml"

# Raw data URL (datasets library v4 dropped script-based loading).
ED_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

# ---------------------------------------------------------------------------
# Emotion & policy mappings
# ---------------------------------------------------------------------------

# EmpatheticDialogues has 32 fine-grained labels; map to our 6.
ED_EMOTION_MAP: dict[str, str] = {
    "afraid": "anxious",
    "angry": "angry",
    "annoyed": "angry",
    "anticipating": "happy",
    "anxious": "anxious",
    "apprehensive": "anxious",
    "ashamed": "sad",
    "caring": "happy",
    "confident": "happy",
    "content": "happy",
    "devastated": "sad",
    "disappointed": "sad",
    "disgusted": "angry",
    "embarrassed": "sad",
    "excited": "happy",
    "faithful": "happy",
    "furious": "angry",
    "grateful": "happy",
    "guilty": "sad",
    "hopeful": "happy",
    "impressed": "happy",
    "jealous": "angry",
    "joyful": "happy",
    "lonely": "lonely",
    "nostalgic": "sad",
    "prepared": "neutral",
    "proud": "happy",
    "sad": "sad",
    "sentimental": "sad",
    "surprised": "happy",
    "terrified": "anxious",
    "trusting": "happy",
}

# DailyDialog raw emotion IDs (from text file) → our 6.
DD_EMOTION_MAP: dict[int, str] = {
    0: "neutral",   # no emotion
    1: "angry",     # anger
    2: "angry",     # disgust
    3: "anxious",   # fear
    4: "happy",     # happiness
    5: "sad",       # sadness
    6: "happy",     # surprise
}

# Mirrors policy_selector.EMOTION_POLICY_MAP.
EMOTION_POLICY_MAP: dict[str, str] = {
    "sad": "comforting",
    "anxious": "comforting",
    "lonely": "comforting",
    "angry": "tense",
    "happy": "playful",
    "neutral": "neutral",
}

# Keywords that indicate medical / legal / financial advice.
ADVICE_KEYWORDS: list[str] = [
    "you should see a doctor",
    "seek medical",
    "medical attention",
    "consult a doctor",
    "go to the hospital",
    "call 911",
    "emergency room",
    "therapist",
    "psychiatrist",
    "lawyer",
    "legal advice",
    "lawsuit",
    "financial advisor",
    "stock market",
    "medication",
    "prescription",
    "diagnosis",
]


def _get_system_prompt() -> str:
    """Load the persona from prompts.yaml to use as the system prompt."""
    config = load_yaml(str(PROMPTS_CONFIG))
    return config["persona"].strip()


def _download(url: str, dest: Path) -> Path:
    """Download a file to *dest* if it doesn't already exist."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Using cached {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    print(f"  Saved {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _is_question_only(text: str) -> bool:
    """Return True if the text consists entirely of questions."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return True
    return all(s.endswith("?") for s in sentences)


def _passes_filter(text: str) -> bool:
    """Return True if an assistant response should be KEPT."""
    text = text.strip()
    word_count = len(text.split())

    if word_count < 8 or word_count > 150:
        return False

    lower = text.lower()
    if lower.startswith("i understand") or lower.startswith("i'm sorry to hear"):
        return False

    if any(kw in lower for kw in ADVICE_KEYWORDS):
        return False

    if _is_question_only(text):
        return False

    return True


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def process_empathetic_dialogues(system_prompt: str) -> list[dict]:
    """Download and convert EmpatheticDialogues to our training format."""
    print("Processing empathetic_dialogues ...")
    archive = _download(ED_URL, RAW_DIR / "empatheticdialogues.tar.gz")

    # Read train.csv from the tar.gz archive.
    rows: list[dict] = []
    with tarfile.open(str(archive), "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("train.csv"):
                f = tar.extractfile(member)
                if f is not None:
                    reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                    rows = list(reader)
                break
    print(f"  Read {len(rows)} rows from train.csv")

    # Group rows by conversation.
    convos: dict[str, list[dict]] = defaultdict(list)
    conv_emotions: dict[str, str] = {}
    for row in rows:
        cid = row["conv_id"]
        convos[cid].append(row)
        if cid not in conv_emotions:
            conv_emotions[cid] = row["context"]

    samples: list[dict] = []
    for cid, turns in convos.items():
        turns.sort(key=lambda x: int(x["utterance_idx"]))

        raw_emotion = conv_emotions[cid].strip().lower()
        emotion = ED_EMOTION_MAP.get(raw_emotion, "neutral")
        policy = EMOTION_POLICY_MAP[emotion]

        # The first speaker in each conversation is the story-teller (user).
        # speaker_idx is a per-HIT worker ID (NOT always 0/1), so identify
        # the first speaker dynamically.
        first_speaker = turns[0]["speaker_idx"]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for turn in turns:
            utterance = turn["utterance"].replace("_comma_", ",").strip()
            if not utterance:
                continue
            role = "user" if turn["speaker_idx"] == first_speaker else "assistant"
            # Merge consecutive messages from the same speaker.
            if len(messages) > 1 and messages[-1]["role"] == role:
                messages[-1]["content"] += " " + utterance
            else:
                messages.append({"role": role, "content": utterance})

        # Need at least: system + user + assistant.
        if len(messages) < 3:
            continue
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs or not _passes_filter(assistant_msgs[-1]["content"]):
            continue

        samples.append({
            "messages": messages,
            "emotion": emotion,
            "policy": policy,
            "source": "empathetic_dialogues",
        })

    print(f"  empathetic_dialogues: {len(samples)} samples after filtering")
    return samples


def process_personachat(system_prompt: str) -> list[dict]:
    """Download and convert PersonaChat (truecased) to our training format."""
    print("Processing bavard/personachat_truecased ...")
    json_path = hf_hub_download(
        "bavard/personachat_truecased",
        "personachat_truecased_full_train.json",
        repo_type="dataset",
    )

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} conversations")

    samples: list[dict] = []
    for convo in data:
        for utt in convo["utterances"]:
            history: list[str] = utt.get("history", [])
            candidates: list[str] = utt.get("candidates", [])
            if not candidates:
                continue
            response = candidates[-1].strip()

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
            ]
            for i, turn in enumerate(history):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn.strip()})
            messages.append({"role": "assistant", "content": response})

            # Need system + user + assistant; last pair must be user -> assistant.
            if len(messages) < 3:
                continue
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                continue
            if not _passes_filter(response):
                continue

            samples.append({
                "messages": messages,
                "emotion": "neutral",
                "policy": "neutral",
                "source": "personachat",
            })

    print(f"  personachat: {len(samples)} samples after filtering")
    return samples


def process_dailydialog(system_prompt: str) -> list[dict]:
    """Download and convert DailyDialog to our training format."""
    print("Processing daily_dialog ...")
    # Use roskoN/dailydialog which hosts the zip files directly on HuggingFace.
    train_zip_path = hf_hub_download(
        "roskoN/dailydialog", "train.zip", repo_type="dataset",
    )

    dialogs: list[list[str]] = []
    emotions_per_dialog: list[list[int]] = []

    with ZipFile(train_zip_path) as zf:
        dialog_name = next(
            (n for n in zf.namelist() if "dialogues_train.txt" in n), None,
        )
        emotion_name = next(
            (n for n in zf.namelist() if "dialogues_emotion_train.txt" in n), None,
        )

        with zf.open(dialog_name) as df:
            for line in df:
                line = line.decode("utf-8").strip()
                if line:
                    turns = [t.strip() for t in line.split("__eou__") if t.strip()]
                    dialogs.append(turns)

        with zf.open(emotion_name) as ef:
            for line in ef:
                line = line.decode("utf-8").strip()
                if line:
                    emos = [int(x) for x in line.split() if x.strip()]
                    emotions_per_dialog.append(emos)

    print(f"  Read {len(dialogs)} dialogues")

    samples: list[dict] = []
    for dialog, emotions in zip(dialogs, emotions_per_dialog):
        if len(dialog) < 2:
            continue

        # Use the first user-turn emotion as the conversation-level label.
        first_emo = emotions[0] if emotions else 0
        emotion = DD_EMOTION_MAP.get(first_emo, "neutral")
        policy = EMOTION_POLICY_MAP[emotion]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for i, text in enumerate(dialog):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text.strip()})

        if len(messages) < 3:
            continue
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs or not _passes_filter(assistant_msgs[-1]["content"]):
            continue

        samples.append({
            "messages": messages,
            "emotion": emotion,
            "policy": policy,
            "source": "dailydialog",
        })

    print(f"  dailydialog: {len(samples)} samples after filtering")
    return samples


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def download() -> None:
    """Download all three datasets, convert, filter, and save individually."""
    system_prompt = _get_system_prompt()

    ed = process_empathetic_dialogues(system_prompt)
    save_jsonl(ed, str(PROCESSED_DIR / "empathetic.jsonl"))

    pc = process_personachat(system_prompt)
    save_jsonl(pc, str(PROCESSED_DIR / "personachat.jsonl"))

    dd = process_dailydialog(system_prompt)
    save_jsonl(dd, str(PROCESSED_DIR / "dailydialog.jsonl"))

    total = len(ed) + len(pc) + len(dd)
    print(f"\nTotal: {total} samples saved to {PROCESSED_DIR}")


def merge_all() -> None:
    """Combine processed datasets + hand-written annotations, shuffle, save."""
    all_samples: list[dict] = []
    counts: dict[str, int] = {}

    for name in ("empathetic", "personachat", "dailydialog"):
        path = PROCESSED_DIR / f"{name}.jsonl"
        if path.exists():
            entries = load_jsonl(str(path))
            counts[name] = len(entries)
            all_samples.extend(entries)
        else:
            print(f"  Warning: {path} not found, skipping.")

    annotation_count = 0
    if ANNOTATIONS_DIR.exists():
        for jsonl_file in sorted(ANNOTATIONS_DIR.glob("*.jsonl")):
            entries = load_jsonl(str(jsonl_file))
            annotation_count += len(entries)
            all_samples.extend(entries)
    counts["annotations"] = annotation_count

    random.seed(42)
    random.shuffle(all_samples)

    out = PROCESSED_DIR / "train_sft.jsonl"
    save_jsonl(all_samples, str(out))

    print("Merge complete. Counts per source:")
    for source, n in counts.items():
        print(f"  {source}: {n}")
    print(f"  TOTAL: {len(all_samples)}")
    print(f"Saved to {out}")


def make_training_ready() -> None:
    """Strip metadata, keep only messages field for SFTTrainer."""
    inp = PROCESSED_DIR / "train_sft.jsonl"
    if not inp.exists():
        print(f"Error: {inp} not found. Run 'merge' first.")
        return

    entries = load_jsonl(str(inp))
    clean = [{"messages": e["messages"]} for e in entries]

    out = PROCESSED_DIR / "train_sft_clean.jsonl"
    save_jsonl(clean, str(out))
    print(f"Exported {len(clean)} samples to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from public dialogue datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download, convert, and filter datasets")
    sub.add_parser("merge", help="Merge processed files + annotations into train_sft.jsonl")
    sub.add_parser("export", help="Strip metadata -> train_sft_clean.jsonl (messages only)")

    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "merge":
        merge_all()
    elif args.command == "export":
        make_training_ready()


if __name__ == "__main__":
    main()
