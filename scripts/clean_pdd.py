"""Clean pdd_train_data.jsonl per datacleanpdd.md cleaning pipeline.

Input:  data/annotations/pdd_train_data.jsonl (408 rows)
Output: data/processed/pdd_cleaned_v2.jsonl
        pdd_cleaning_report.md
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "annotations" / "pdd_train_data.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "pdd_cleaned_v2.jsonl"
REPORT_PATH = ROOT / "pdd_cleaning_report.md"

TARGET_SYSTEM_PROMPT = (
    "You are Ari, a text-only AI companion. You are warm, calm, reflective, "
    "and slightly playful. You validate feelings before offering perspective "
    "or suggestions. Keep responses concise (2-4 sentences) in a natural, "
    "conversational tone. Ask at most one gentle follow-up question per reply. "
    "Prioritize emotional safety and connection over giving solutions."
)

EMOTION_MAP = {
    "happy": "happy", "excited": "happy", "hopeful": "happy", "proud": "happy",
    "content": "happy", "connected": "happy", "positive": "happy",
    "motivated": "happy", "touched": "happy", "grateful": "happy",
    "warm": "happy", "affection": "happy", "loving": "happy", "caring": "happy",
    "amused": "happy", "playful": "happy", "relieved": "happy", "dreamy": "happy",
    "anxious": "anxious", "worried": "anxious", "nervous": "anxious",
    "scared": "anxious", "panicked": "anxious", "stressed": "anxious",
    "overwhelmed": "anxious", "insecure": "anxious", "uncomfortable": "anxious",
    "uneasy": "anxious", "uncertain": "anxious", "confused": "anxious",
    "awkward": "anxious", "hesitant": "anxious", "vulnerable": "anxious",
    "embarrassed": "anxious", "ashamed": "anxious", "self-conscious": "anxious",
    "self-critical": "anxious", "cautious": "anxious", "intimidated": "anxious",
    "sensitive": "anxious", "pleading": "anxious", "conflicted": "anxious",
    "restless": "anxious", "rushed": "anxious", "jealous": "anxious",
    "suspicious": "anxious",
    "sad": "sad", "hurt": "sad", "heartbroken": "sad", "devastated": "sad",
    "disappointed": "sad", "discouraged": "sad", "regretful": "sad",
    "guilty": "sad", "miserable": "sad", "down": "sad", "homesick": "sad",
    "nostalgic": "sad", "tired": "sad", "exhausted": "sad", "drained": "sad",
    "sick": "sad", "lazy": "sad", "pain": "sad", "upset": "sad",
    "withdrawn": "sad", "avoidant": "sad", "stuck": "sad", "whiny": "sad",
    "angry": "angry", "frustrated": "angry", "annoyed": "angry",
    "irritated": "angry", "exasperated": "angry", "resentful": "angry",
    "defiant": "angry", "impatient": "angry", "used": "angry", "picky": "angry",
    "tense": "angry",
    "lonely": "lonely",
    "neutral": "neutral", "curious": "neutral", "thoughtful": "neutral",
    "reflective": "neutral", "persistent": "neutral", "bored": "neutral",
    "mixed": "neutral", "hungry": "neutral",
}

MOOD_MAP = {
    "cheerful": "cheerful", "happy": "cheerful", "joyful": "cheerful",
    "uplifted": "cheerful", "positive": "cheerful", "warm": "cheerful",
    "excited": "cheerful", "proud": "cheerful", "confident": "cheerful",
    "energetic": "cheerful", "triumphant": "cheerful", "light": "cheerful",
    "peaceful": "cheerful", "hopeful": "cheerful", "grateful": "cheerful",
    "touched": "cheerful", "comfortable": "cheerful", "amused": "cheerful",
    "mischievous": "cheerful", "playful": "cheerful", "adoring": "cheerful",
    "affectionate": "cheerful", "generous": "cheerful", "motivated": "cheerful",
    "determined": "cheerful", "creative": "cheerful", "imaginative": "cheerful",
    "wiggly": "cheerful", "competitive": "cheerful", "sneaky": "cheerful",
    "inquisitive": "cheerful", "focused": "cheerful",
    "concerned": "concerned", "soft": "concerned", "gentle": "concerned",
    "low": "concerned", "sad": "concerned", "heavy": "concerned",
    "grieving": "concerned", "remorseful": "concerned", "vulnerable": "concerned",
    "fragile": "concerned", "worried": "concerned", "insecure": "concerned",
    "nervous": "concerned", "helpless": "concerned", "rejected": "concerned",
    "clingy": "concerned", "sheepish": "concerned", "embarrassed": "concerned",
    "disappointed": "concerned", "defeated": "concerned", "sleepy": "concerned",
    "sick": "concerned", "drained": "concerned", "tired": "concerned",
    "complex": "concerned", "guilty": "concerned", "pain": "concerned",
    "tense": "tense", "stressed": "tense", "anxious": "tense", "angry": "tense",
    "frustrated": "tense", "annoyed": "tense", "irritable": "tense",
    "defensive": "tense", "resentful": "tense", "bitter": "tense",
    "stubborn": "tense", "suspicious": "tense", "petty": "tense",
    "grumpy": "tense", "dark": "tense", "resistant": "tense",
    "panicked": "tense", "rushed": "tense", "impatient": "tense",
    "uncomfortable": "tense", "withdrawn": "tense",
    "reflective": "reflective", "thoughtful": "reflective",
    "contemplative": "reflective", "curious": "reflective",
    "observant": "reflective", "aware": "reflective", "knowing": "reflective",
    "quiet": "reflective", "conflicted": "reflective", "nostalgic": "reflective",
    "stuck": "reflective",
    "neutral": "neutral", "calm": "neutral", "steady": "neutral",
    "mixed": "neutral", "uneasy": "neutral", "restless": "neutral",
    "cautious": "neutral", "flat": "neutral",
}

POLICY_MAP = {
    "comforting": "comforting", "gentle": "comforting", "warm": "comforting",
    "reassuring": "comforting", "validating": "comforting",
    "calming": "comforting", "patient": "comforting",
    "protective": "comforting", "affirming": "comforting",
    "supportive": "comforting", "affectionate": "comforting",
    "forgiving": "comforting", "encouraging": "comforting",
    "cautious": "comforting", "emotional": "comforting",
    "neutral": "neutral", "helpful": "neutral", "practical": "neutral",
    "direct": "neutral", "honest": "neutral", "firm": "neutral",
    "calm": "neutral", "accountable": "neutral", "balanced": "neutral",
    "guiding": "neutral", "light": "neutral",
    "playful": "playful", "humorous": "playful", "amused": "playful",
    "cheerful": "playful", "enthusiastic": "playful", "creative": "playful",
    "reflective": "reflective", "curious": "reflective",
    "thoughtful": "reflective",
    "tense": "tense",
}

VALID_EMOTIONS = {"happy", "anxious", "sad", "angry", "lonely", "neutral"}
VALID_MOODS = {"cheerful", "concerned", "tense", "reflective", "neutral"}
VALID_POLICIES = {"comforting", "neutral", "playful", "reflective", "tense"}

CJK_REPLACEMENTS = {
    "\u653b\u7565": "guide",
}

SOUNDS_LIKE_OPENER_RE = re.compile(
    r"^(?:it\s+sounds\s+like|that\s+sounds\s+like|sounds\s+like)\b", re.IGNORECASE
)
SOUNDS_LIKE_ANY_RE = re.compile(r"sounds\s+like", re.IGNORECASE)

# Per-row ? fixes (keyed by post-Step-1 normalized assistant content).
Q_FIXES: dict[str, str] = {
    # 7b: "A dinosaur spaceship?" rhetorical excitement → exclamation
    "Whoa. A dinosaur spaceship? That's the coolest thing I've ever seen. Can I put it on the fridge?":
        "Whoa. A dinosaur spaceship! That's the coolest thing I've ever seen. Can I put it on the fridge?",
    # 7c: merge two questions into one
    "I know that feeling. What kind of mood are you in? Scary, funny, or the kind that makes you cry but in a good way?":
        "I know that feeling. What kind of mood are you in — scary, funny, or the kind that makes you cry but in a good way?",
    # 7b: "Even one?" emphasis → merge with em-dash
    "Haha 'da da da' is not a lot to go on but I appreciate the effort. Do you remember any actual words? Even one?":
        "Haha 'da da da' is not a lot to go on but I appreciate the effort. Do you remember any actual words — even one?",
    # 7c: merge two-option clarification
    "Thirty tries means you're committed. Which phase is killing you? The platform part or the actual fight? Happy to give specific tips.":
        "Thirty tries means you're committed. Which phase is killing you — the platform part or the actual fight? Happy to give specific tips.",
    # 7b: "hum a bit?" suggestion → statement
    "Which song? He's been featuring a lot of people lately. If it's 'Like That' it's Future and Metro Boomin. If not, hum a bit? No judgment.":
        "Which song? He's been featuring a lot of people lately. If it's 'Like That' it's Future and Metro Boomin. If not, try humming a bit. No judgment.",
    # 7c: merge two questions into one
    "The deep cuts. Have you seen The Jinx? Or Evil Genius? Both are wild and less talked about than Making a Murderer.":
        "The deep cuts. Have you seen The Jinx or Evil Genius? Both are wild and less talked about than Making a Murderer.",
    # 7b: "Could be an Adele song?" speculative → statement
    "That's a little vague but I'll try. Could be an Adele song? Or something from a soundtrack. Do you remember if it was a man or woman singing?":
        "That's a little vague but I'll try. Could be an Adele song. Or something from a soundtrack. Do you remember if it was a man or woman singing?",
    # 7b: "chart for the fridge?" suggestion → statement
    "Ugh bathroom duty is the worst. Maybe make a little chore chart for the fridge? Or just say 'hey I did it last time, you mind getting it this week?' Keep it light.":
        "Ugh bathroom duty is the worst. Maybe make a little chore chart for the fridge. Or just say 'hey I did it last time, you mind getting it this week?' Keep it light.",
    # 7c: two real follow-ups → convert second to statement
    "Slow down. Why did you break up in the first place? Has that thing actually changed? If nothing's different, you'll just break up again over the same thing.":
        "Slow down. Why did you break up in the first place? Think about whether that thing has actually changed. If nothing's different, you'll just break up again over the same thing.",
    # 7c: two in-quote suggestions → convert first to statement
    "Haha that would annoy me too. Next time just say 'hey can I pick the photo next time?' Or jokingly 'are you trying to make me look bad?' Let them know you noticed.":
        "Haha that would annoy me too. Next time just say 'hey, let me pick the photo next time.' Or jokingly 'are you trying to make me look bad?' Let them know you noticed.",
    # 7c: merge two pattern-questions into one
    "That's really self-aware of you to even ask. Look for patterns. Do the same fights keep happening? Do people often pull away? A therapist could help you see it clearly.":
        "That's really self-aware of you to even ask. Look for patterns. Do the same fights keep happening, do people often pull away? A therapist could help you see it clearly.",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assistant_contents(row: dict) -> list[str]:
    return [m["content"] for m in row["messages"] if m["role"] == "assistant"]


def first_user_content(row: dict) -> str:
    for m in row["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def set_assistant_content(row: dict, new_contents: list[str]) -> None:
    idx = 0
    for m in row["messages"]:
        if m["role"] == "assistant":
            m["content"] = new_contents[idx]
            idx += 1


def normalize_assistant_text(text: str) -> tuple[str, list[str]]:
    """Step 1: Unicode normalize + CJK replacement. Returns (new_text, log_entries)."""
    log: list[str] = []
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201C", '"')
        .replace("\u201D", '"')
        .replace("\u2026", "...")
    )
    for cjk, eng in CJK_REPLACEMENTS.items():
        if cjk in text:
            log.append(f"Replaced CJK '{cjk}' with '{eng}'")
            text = text.replace(cjk, eng)
    for ch in text:
        if 0x4E00 <= ord(ch) <= 0x9FFF:
            log.append(f"Unexpected CJK char U+{ord(ch):04X} found — dropping row")
            return text, log
    result = "".join(
        ch for ch in text
        if not (unicodedata.category(ch) == "So" or ord(ch) > 0xFFFF)
    )
    return result, log


def case_preserving_sounds_to_seems(text: str) -> str:
    def repl(m: re.Match) -> str:
        original = m.group(0)
        return ("Seems " if original[0].isupper() else "seems ") + (
            "Like" if original.split()[1][0].isupper() else "like"
        )
    return SOUNDS_LIKE_ANY_RE.sub(repl, text)


def starts_with_sounds_like_opener(text: str) -> bool:
    return bool(SOUNDS_LIKE_OPENER_RE.match(text.lstrip()))


def prefix_key(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return " ".join(words[:3]) if len(words) >= 3 else ""


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


LONG_REPLY_CLICHES = ["sounds like", "i hear you", "i understand that"]


def main() -> None:
    rows = load_rows(INPUT_PATH)
    input_count = len(rows)

    report: dict[str, Any] = {
        "input_count": input_count,
        "dropped": defaultdict(list),
        "emotion_before": Counter(),
        "emotion_after": Counter(),
        "mood_before": Counter(),
        "mood_after": Counter(),
        "policy_before": Counter(),
        "policy_after": Counter(),
        "q_fixes": [],
        "sounds_like_rewrites": [],
        "unicode_fixes": [],
        "prefix_drops": defaultdict(int),
    }

    for r in rows:
        report["emotion_before"][r.get("emotion", "MISSING")] += 1
        report["mood_before"][r.get("state", {}).get("mood", "MISSING")] += 1
        report["policy_before"][r.get("policy", "MISSING")] += 1

    # Verify system prompt
    sys_msg = [m for m in rows[0]["messages"] if m["role"] == "system"]
    if sys_msg[0]["content"] == TARGET_SYSTEM_PROMPT:
        report["sys_prompt_status"] = "system prompt already unified — no changes needed"
    else:
        raise SystemExit("System prompt does NOT match target. Aborting.")

    # Step 1: normalize assistant content + CJK
    drop_cjk: set[int] = set()
    for i, r in enumerate(rows):
        new = []
        for a in assistant_contents(r):
            na, log = normalize_assistant_text(a)
            report["unicode_fixes"].extend(log)
            if any("dropping row" in l for l in log):
                drop_cjk.add(i)
            new.append(na)
        set_assistant_content(r, new)
    if drop_cjk:
        rows = [r for i, r in enumerate(rows) if i not in drop_cjk]

    # Step 2: emotion remap
    for r in rows:
        r["emotion"] = EMOTION_MAP.get(r.get("emotion", ""), "neutral")

    # Step 3: mood remap
    for r in rows:
        state = r.setdefault("state", {})
        state["mood"] = MOOD_MAP.get(state.get("mood", ""), "neutral")

    # Step 4: policy remap
    for r in rows:
        r["policy"] = POLICY_MAP.get(r.get("policy", ""), "neutral")

    # Step 5a: drop sounds-like openers
    keep: list[dict] = []
    for r in rows:
        offenders = [a for a in assistant_contents(r) if starts_with_sounds_like_opener(a)]
        if offenders:
            report["dropped"]["cliche_opener"].append(
                (r.get("state", {}).get("intimacy", 0), offenders[0][:100])
            )
        else:
            keep.append(r)
    rows = keep

    # Step 5b: mid-sentence sounds-like rewrites
    for r in rows:
        new = []
        for a in assistant_contents(r):
            if SOUNDS_LIKE_ANY_RE.search(a):
                rewritten = case_preserving_sounds_to_seems(a)
                report["sounds_like_rewrites"].append((a, rewritten))
                new.append(rewritten)
            else:
                new.append(a)
        set_assistant_content(r, new)

    # Step 6: template prefix repetition (threshold=8)
    prefix_counts: Counter = Counter()
    for r in rows:
        key = prefix_key(assistant_contents(r)[0])
        if key:
            prefix_counts[key] += 1
    hot_prefixes = {k for k, v in prefix_counts.items() if v >= 8}

    seen_prefix_count: Counter = Counter()
    keep = []
    for r in rows:
        key = prefix_key(assistant_contents(r)[0])
        if key in hot_prefixes:
            seen_prefix_count[key] += 1
            if seen_prefix_count[key] > 2:
                report["dropped"]["template_repetition"].append(
                    (r.get("state", {}).get("intimacy", 0), assistant_contents(r)[0][:100])
                )
                report["prefix_drops"][key] += 1
                continue
        keep.append(r)
    rows = keep

    # Step 7: fix ? violations
    for r in rows:
        new = []
        for a in assistant_contents(r):
            if a.count("?") >= 2:
                fixed = Q_FIXES.get(a)
                if fixed is None:
                    raise RuntimeError(
                        f"Unrecognized `?` violation not in Q_FIXES table:\n  {a!r}"
                    )
                report["q_fixes"].append((a, fixed))
                new.append(fixed)
            else:
                new.append(a)
        set_assistant_content(r, new)

    # Step 8: handle long replies (7+ sentences)
    keep = []
    for r in rows:
        a = assistant_contents(r)[-1]
        sc = count_sentences(a)
        if sc >= 7:
            al = a.lower()
            if any(c in al for c in LONG_REPLY_CLICHES):
                report["dropped"]["long_reply_with_cliche"].append(
                    (r.get("state", {}).get("intimacy", 0), a[:100])
                )
                continue
        keep.append(r)
    rows = keep

    # Step 9a: drop exact-duplicate assistant content
    seen_assist: set[str] = set()
    keep = []
    for r in rows:
        sig = "\n---\n".join(assistant_contents(r)).strip()
        if sig in seen_assist:
            report["dropped"]["duplicate_assistant"].append(
                (r.get("state", {}).get("intimacy", 0), assistant_contents(r)[0][:100])
            )
            continue
        seen_assist.add(sig)
        keep.append(r)
    rows = keep

    # Step 9b: drop duplicate first-user at same intimacy (±5)
    by_user: defaultdict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_user[first_user_content(r).strip()].append(r)

    keep_ids: set[int] = set()
    for user_text, group in by_user.items():
        if len(group) == 1:
            keep_ids.add(id(group[0]))
            continue
        kept_intimacies: list[int] = []
        for r in group:
            intim = r.get("state", {}).get("intimacy", 0)
            if any(abs(intim - k) <= 5 for k in kept_intimacies):
                report["dropped"]["duplicate_user_same_intimacy"].append(
                    (intim, first_user_content(r)[:100])
                )
                continue
            kept_intimacies.append(intim)
            keep_ids.add(id(r))
    rows = [r for r in rows if id(r) in keep_ids]

    # Step 10: mark source
    for r in rows:
        if r.get("source") == "handwritten":
            r["source"] = "handwritten_pdd_cleaned"

    # Step 11: final validation
    errors: list[str] = []
    for i, r in enumerate(rows):
        if set(r.keys()) != {"messages", "emotion", "policy", "state", "source"}:
            errors.append(f"row {i}: unexpected keys {sorted(r.keys())}")
        sys_msgs = [m for m in r["messages"] if m["role"] == "system"]
        if not sys_msgs or sys_msgs[0]["content"] != TARGET_SYSTEM_PROMPT:
            errors.append(f"row {i}: system prompt mismatch")
        if r["emotion"] not in VALID_EMOTIONS:
            errors.append(f"row {i}: bad emotion {r['emotion']!r}")
        if r["state"]["mood"] not in VALID_MOODS:
            errors.append(f"row {i}: bad mood {r['state']['mood']!r}")
        if r["policy"] not in VALID_POLICIES:
            errors.append(f"row {i}: bad policy {r['policy']!r}")
        for a in assistant_contents(r):
            if a.count("?") > 1:
                errors.append(f"row {i}: ? count > 1 in {a[:80]!r}")
            al = a.lstrip().lower()
            for pat in ("sounds like", "it sounds like", "that sounds like"):
                if al.startswith(pat):
                    errors.append(f"row {i}: opens with {pat!r}")
                    break
            for ch in a:
                if unicodedata.category(ch) == "So" or ord(ch) > 0xFFFF:
                    errors.append(f"row {i}: emoji/symbol U+{ord(ch):04X}")
                    break
                if 0x4E00 <= ord(ch) <= 0x9FFF:
                    errors.append(f"row {i}: CJK char U+{ord(ch):04X}")
                    break

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors[:50]:
            print(f"  {e}")
        raise SystemExit(f"\n{len(errors)} validation errors — aborting.")

    for r in rows:
        report["emotion_after"][r["emotion"]] += 1
        report["mood_after"][r["state"]["mood"]] += 1
        report["policy_after"][r["policy"]] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_report(rows, report)
    print(f"[ok] wrote {len(rows)} rows to {OUTPUT_PATH}")
    print(f"[ok] wrote report to {REPORT_PATH}")


def write_report(rows: list[dict], report: dict) -> None:
    lines: list[str] = []
    ap = lines.append

    ap("# PDD Dataset Cleaning Report\n")
    ap(f"**Input:** `{INPUT_PATH.name}` ({report['input_count']} rows)")
    ap(f"**Output:** `{OUTPUT_PATH.name}` ({len(rows)} rows)\n")

    # 1. Summary
    ap("## 1. Summary Counts\n")
    ap(f"- Input rows: **{report['input_count']}**")
    ap(f"- Output rows: **{len(rows)}**")
    ap(f"- Total rows dropped: **{report['input_count'] - len(rows)}**\n")
    ap(f"System prompt: {report.get('sys_prompt_status', 'N/A')}\n")
    ap("### Drops by reason\n")
    ap("| Reason | Count |")
    ap("|---|---|")
    for k in ("cliche_opener", "template_repetition", "long_reply_with_cliche",
              "duplicate_assistant", "duplicate_user_same_intimacy"):
        ap(f"| {k.replace('_', ' ')} | {len(report['dropped'][k])} |")
    ap("")

    # 2. Emotion remap
    ap("## 2. Emotion Remapping\n")
    ap(f"Before: {len(report['emotion_before'])} unique → After: {len(report['emotion_after'])} unique\n")
    ap("### Before\n")
    ap("| Emotion | Count |")
    ap("|---|---|")
    for k, v in sorted(report["emotion_before"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")
    ap("### After\n")
    ap("| Emotion | Count |")
    ap("|---|---|")
    for k, v in sorted(report["emotion_after"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")

    # 3. Mood remap
    ap("## 3. Mood Remapping\n")
    ap(f"Before: {len(report['mood_before'])} unique → After: {len(report['mood_after'])} unique\n")
    ap("### Before\n")
    ap("| Mood | Count |")
    ap("|---|---|")
    for k, v in sorted(report["mood_before"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")
    ap("### After\n")
    ap("| Mood | Count |")
    ap("|---|---|")
    for k, v in sorted(report["mood_after"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")

    # 4. Policy remap (detailed)
    ap("## 4. Policy Remapping (biggest change)\n")
    ap(f"Before: {len(report['policy_before'])} unique → After: {len(report['policy_after'])} unique\n")
    ap("### Before\n")
    ap("| Policy | Count |")
    ap("|---|---|")
    for k, v in sorted(report["policy_before"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")
    ap("### After\n")
    ap("| Policy | Count |")
    ap("|---|---|")
    for k, v in sorted(report["policy_after"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")

    # 5. Dropped rows log
    ap("## 5. Dropped Rows\n")
    for reason_key, header in [
        ("cliche_opener", "Cliche opener"),
        ("template_repetition", "Template prefix repetition (>=8 occurrences, kept first 2)"),
        ("long_reply_with_cliche", "Long reply (7+ sentences) with cliche filler"),
        ("duplicate_assistant", "Duplicate assistant content"),
        ("duplicate_user_same_intimacy", "Duplicate user prompt at same intimacy (+-5)"),
    ]:
        drops = report["dropped"][reason_key]
        ap(f"### {header} — {len(drops)} dropped\n")
        if not drops:
            ap("_none_\n")
            continue
        ap("| Intimacy | Preview (first 100 chars) |")
        ap("|---|---|")
        for intim, prev in drops:
            safe = prev.replace("|", "\\|")
            ap(f"| {intim} | {safe} |")
        ap("")

    if report["prefix_drops"]:
        ap("### Prefix-repetition breakdown\n")
        ap("| Prefix | Rows dropped |")
        ap("|---|---|")
        for k, v in sorted(report["prefix_drops"].items(), key=lambda x: -x[1]):
            ap(f"| `{k}` | {v} |")
        ap("")

    # 6. Unicode fixes
    ap("## 6. Unicode / CJK Fixes\n")
    if not report["unicode_fixes"]:
        ap("_none_\n")
    else:
        for entry in report["unicode_fixes"]:
            ap(f"- {entry}")
        ap("")

    # 7. ? fix log
    ap("## 7. `?` Rule Violations Fixed\n")
    if not report["q_fixes"]:
        ap("_none_\n")
    else:
        ap(f"Total fixed: **{len(report['q_fixes'])}**\n")
        for before, after in report["q_fixes"]:
            ap(f"- **Before:** {before}")
            ap(f"- **After:**  {after}\n")

    # 8. sounds-like rewrites
    ap("## 8. Mid-sentence \"sounds like\" Rewrites\n")
    if not report["sounds_like_rewrites"]:
        ap("_none_\n")
    else:
        ap(f"Total rewrites: **{len(report['sounds_like_rewrites'])}**\n")
        for before, after in report["sounds_like_rewrites"]:
            ap(f"- **Before:** {before}")
            ap(f"- **After:**  {after}\n")

    # 9. Final distributions
    ap("## 9. Final Label Distributions\n")
    ap("### Emotion\n")
    ap("| Emotion | Count |")
    ap("|---|---|")
    for k, v in sorted(report["emotion_after"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")
    ap("### Mood\n")
    ap("| Mood | Count |")
    ap("|---|---|")
    for k, v in sorted(report["mood_after"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")
    ap("### Policy\n")
    ap("| Policy | Count |")
    ap("|---|---|")
    for k, v in sorted(report["policy_after"].items(), key=lambda x: -x[1]):
        ap(f"| {k} | {v} |")
    ap("")

    # 10. Intimacy histogram
    ap("## 10. Intimacy Distribution (Post-clean)\n")
    buckets: Counter = Counter()
    for r in rows:
        intim = r["state"].get("intimacy", 0)
        buckets[(intim // 10) * 10] += 1
    max_count = max(buckets.values()) if buckets else 1
    ap("| Range | Count | Bar |")
    ap("|---|---|---|")
    for b in sorted(buckets):
        bar = "\u2588" * int(40 * buckets[b] / max_count)
        ap(f"| {b}-{b + 9} | {buckets[b]} | {bar} |")
    ap("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
