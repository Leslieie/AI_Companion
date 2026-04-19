"""Clean yls_train_data_cleaned.jsonl per dataClean.md cleaning pipeline.

Input:  data/annotations/yls_train_data_cleaned.jsonl (605 rows)
Output: data/processed/yls_cleaned_v2.jsonl
        cleaning_report.md
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "annotations" / "yls_train_data_cleaned.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "yls_cleaned_v2.jsonl"
REPORT_PATH = ROOT / "cleaning_report.md"

TARGET_SYSTEM_PROMPT = (
    "You are Ari, a text-only AI companion. You are warm, calm, reflective, "
    "and slightly playful. You validate feelings before offering perspective "
    "or suggestions. Keep responses concise (2-4 sentences) in a natural, "
    "conversational tone. Ask at most one gentle follow-up question per reply. "
    "Prioritize emotional safety and connection over giving solutions."
)

EMOTION_MAP = {
    "happy": "happy", "content": "happy", "relieved": "happy", "hopeful": "happy",
    "proud": "happy", "neutral-positive": "happy",
    "anxious": "anxious", "confused": "anxious", "uncertain": "anxious",
    "insecure": "anxious", "conflicted": "anxious", "awkward": "anxious",
    "embarrassed": "anxious", "self-doubt": "anxious", "self_doubt": "anxious",
    "self-critical": "anxious", "uneasy": "anxious", "restless": "anxious",
    "ruminating": "anxious", "overwhelmed": "anxious", "light_anxious": "anxious",
    "stressed": "anxious", "overloaded": "anxious",
    "sad": "sad", "tired": "sad", "low": "sad", "low-energy": "sad",
    "letdown": "sad", "regretful": "sad", "lost": "sad", "burnout": "sad",
    "numb": "sad", "flat": "sad", "withdrawn": "sad", "detached": "sad",
    "minimizing": "sad",
    "angry": "angry", "frustrated": "angry", "annoyed": "angry",
    "tense": "angry", "dismissive": "angry", "skeptical": "angry",
    "guarded": "angry", "avoidant": "angry",
    "lonely": "lonely",
    "neutral": "neutral", "bored": "neutral", "mixed": "neutral", "unmotivated": "neutral",
}

MOOD_MAP = {
    "cheerful": "cheerful", "positive": "cheerful", "uplifting": "cheerful",
    "bright": "cheerful", "playful": "cheerful", "uplifted": "cheerful",
    "relaxed": "cheerful", "energizing": "cheerful", "peaceful": "cheerful",
    "proud": "cheerful", "warm": "cheerful",
    "concerned": "concerned", "empathetic": "concerned", "supportive": "concerned",
    "reassuring": "concerned", "understanding": "concerned", "soft": "concerned",
    "grounding": "concerned", "gentle": "concerned", "low": "concerned",
    "tense": "tense", "direct": "tense", "muted": "tense",
    "reflective": "reflective", "deep": "reflective", "thoughtful": "reflective",
    "contemplative": "reflective", "curious": "reflective", "observant": "reflective",
    "aware": "reflective", "knowing": "reflective", "quiet": "reflective",
    "engaged": "reflective", "focused": "reflective", "grounded": "reflective",
    "neutral": "neutral", "calm": "neutral", "steady": "neutral",
    "mixed": "neutral", "open": "neutral", "flat": "neutral", "uncertain": "neutral",
}

VALID_POLICIES = {"comforting", "neutral", "playful", "reflective", "tense"}
VALID_EMOTIONS = {"happy", "anxious", "sad", "angry", "lonely", "neutral"}
VALID_MOODS = {"cheerful", "concerned", "tense", "reflective", "neutral"}

SOUNDS_LIKE_OPENER_RE = re.compile(
    r"^(?:it\s+sounds\s+like|that\s+sounds\s+like|sounds\s+like)\b", re.IGNORECASE
)
SOUNDS_LIKE_ANY_RE = re.compile(r"sounds\s+like", re.IGNORECASE)

# Per-row `?`-violation rewrites (Step 8). Keyed by the assistant content AFTER
# Steps 1-7 have been applied (so "sounds like" -> "seems like" has happened).
Q_FIXES = {
    # 8b: rhetorical "Right?" -> em-dash
    "Right? It's the kind of thing where you're like... why did this need to be this hard?":
        "Right — it's the kind of thing where you're like... why did this need to be this hard?",
    # 8c: two real follow-ups -> drop generic second
    "Fair... but are you avoiding the task, or the feeling that comes with it? Sometimes it's not about the thing itself. What makes it feel hard to start?":
        "Fair... but are you avoiding the task, or the feeling that comes with it? Sometimes it's not about the thing itself.",
    # 8b: "Honestly?" rhetorical -> em-dash  (sounds like -> seems like from Step 6)
    "Honestly? That seems like a pretty tough standard you're holding yourself to. No sugarcoating it, that kind of pressure wears you down. When does that voice get the loudest?":
        "Honestly — that seems like a pretty tough standard you're holding yourself to. No sugarcoating it, that kind of pressure wears you down. When does that voice get the loudest?",
    # 8b: "Kinda unfair, right?" rhetorical -> em-dash
    "Mm, yeah... that one tends to replay like it's trying to rewrite the past in real time. Kinda unfair, right? Like your brain acting as both judge *and* prosecutor. Do you feel like you're looking for reassurance... or trying to avoid making the same mistake again?":
        "Mm, yeah... that one tends to replay like it's trying to rewrite the past in real time. Kinda unfair, right — like your brain acting as both judge *and* prosecutor. Do you feel like you're looking for reassurance... or trying to avoid making the same mistake again?",
    # 8b: "But honestly?" rhetorical -> em-dash
    "Yeah, that makes sense. Sometimes it's easier to act unbothered than risk someone seeing it mattered. But honestly? Caring doesn't make you weak—it just means something hit a nerve. What part of it felt the most personal?":
        "Yeah, that makes sense. Sometimes it's easier to act unbothered than risk someone seeing it mattered. But honestly — caring doesn't make you weak—it just means something hit a nerve. What part of it felt the most personal?",
    # 8c: two-option clarification -> merge into one
    "Stuck how? Like can't move forward... or more like you're moving, but it all feels kinda the same?":
        "Stuck how — like can't move forward... or more like you're moving, but it all feels kinda the same?",
    # 8b: rhetorical emphasis "*without spiraling*?" -> em-dash statement
    "That's actually bigger than it sounds. Not just getting things done, but doing it *without spiraling*? That's progress. Your nervous system probably appreciated that more than you think. Did anything feel different while you were doing it?":
        "That's actually bigger than it sounds. Not just getting things done, but doing it *without spiraling* — that's progress. Your nervous system probably appreciated that more than you think. Did anything feel different while you were doing it?",
    # 8b: "Honestly?" rhetorical -> em-dash
    "Honestly? Those are elite-tier days. No chaos, no random plot twists. Just life... cooperating for once. Did it feel calm, or more like quietly satisfying?":
        "Honestly — those are elite-tier days. No chaos, no random plot twists. Just life... cooperating for once. Did it feel calm, or more like quietly satisfying?",
    # 8c: two-option clarification -> merge
    "Mm... compared to what, though? Your actual capacity, or that imaginary version of you that somehow has unlimited energy and zero bad days?":
        "Mm... compared to what, though — your actual capacity, or that imaginary version of you that somehow has unlimited energy and zero bad days?",
    # 8c: merge option-list follow-up into primary question
    "Sometimes it's like your brain hasn't labeled it yet, but your body already decided something's off. If you had to guess—not logically, just instinct—what direction does it point? Stress, people, or something internal?":
        "Sometimes it's like your brain hasn't labeled it yet, but your body already decided something's off. If you had to guess—not logically, just instinct—what direction does it point — stress, people, or something internal?",
    # 8c: merge option-list follow-up into primary question
    "Okay, let's not fight it for a second. When it pops back in—what part hits the hardest? The words, the feeling, or how you think it made you look?":
        "Okay, let's not fight it for a second. When it pops back in—what part hits the hardest — the words, the feeling, or how you think it made you look?",
    # 8b: "...right?" rhetorical tag -> em-dash
    "Not in a scary way—more like you want everything to stop asking things from you for a minute, right? If you actually got that quiet space, what do you think you'd do first?":
        "Not in a scary way—more like you want everything to stop asking things from you for a minute, right — if you actually got that quiet space, what do you think you'd do first?",
    # 8a: surprise emphasis ?? -> ?!
    "Oh?? Look at you casually breaking the narrative.":
        "Oh?! Look at you casually breaking the narrative.",
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


def normalize_assistant_text(text: str) -> str:
    """Step 2: smart quotes/ellipsis + strip emoji/symbol chars. Em-dash preserved."""
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201C", '"')
        .replace("\u201D", '"')
        .replace("\u2026", "...")
    )
    return "".join(
        ch for ch in text
        if not (unicodedata.category(ch) == "So" or ord(ch) > 0xFFFF)
    )


def case_preserving_sounds_to_seems(text: str) -> str:
    """Replace 'sounds like' with 'seems like', preserving the first letter's case."""
    def repl(m: re.Match) -> str:
        original = m.group(0)
        return ("Seems " if original[0].isupper() else "seems ") + (
            "Like" if original.split()[1][0].isupper() else "like"
        )
    return SOUNDS_LIKE_ANY_RE.sub(repl, text)


def set_assistant_content(row: dict, new_contents: list[str]) -> None:
    idx = 0
    for m in row["messages"]:
        if m["role"] == "assistant":
            m["content"] = new_contents[idx]
            idx += 1


def starts_with_sounds_like_opener(text: str) -> bool:
    return bool(SOUNDS_LIKE_OPENER_RE.match(text.lstrip()))


def prefix_key(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return " ".join(words[:3]) if len(words) >= 3 else ""


def main() -> None:
    rows = load_rows(INPUT_PATH)
    input_count = len(rows)

    report: dict[str, Any] = {
        "input_count": input_count,
        "dropped": defaultdict(list),     # reason -> list of (intimacy, preview)
        "emotion_before": Counter(),
        "emotion_after": Counter(),
        "mood_before": Counter(),
        "mood_after": Counter(),
        "policy_before": Counter(),
        "policy_after": Counter(),
        "q_fixes": [],                    # (before, after)
        "sounds_like_rewrites": [],       # (before, after)
        "prefix_drops": defaultdict(int), # prefix -> count dropped
    }

    for r in rows:
        report["emotion_before"][r.get("emotion", "MISSING")] += 1
        report["mood_before"][r.get("state", {}).get("mood", "MISSING")] += 1
        report["policy_before"][r.get("policy", "MISSING")] += 1

    # Step 1: replace system prompt on every row
    for r in rows:
        for m in r["messages"]:
            if m["role"] == "system":
                m["content"] = TARGET_SYSTEM_PROMPT

    # Step 2: normalize assistant content
    for r in rows:
        new = [normalize_assistant_text(a) for a in assistant_contents(r)]
        set_assistant_content(r, new)

    # Step 3: emotion remap
    for r in rows:
        r["emotion"] = EMOTION_MAP.get(r.get("emotion", ""), "neutral")

    # Step 4: mood remap
    for r in rows:
        state = r.setdefault("state", {})
        state["mood"] = MOOD_MAP.get(state.get("mood", ""), "neutral")

    # Step 5: validate policy
    for r in rows:
        if r.get("policy") not in VALID_POLICIES:
            r["policy"] = "neutral"

    # Step 6a: drop rows where ANY assistant reply starts with sounds-like opener
    keep: list[dict] = []
    for r in rows:
        offenders = [a for a in assistant_contents(r) if starts_with_sounds_like_opener(a)]
        if offenders:
            report["dropped"]["cliche_opener"].append(
                (r.get("state", {}).get("intimacy", 0), offenders[0][:80])
            )
        else:
            keep.append(r)
    rows = keep

    # Step 6b: mid-sentence sounds-like rewrites
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

    # Step 7: drop rows with template prefix repetition >= 5, keep first 2
    # Compute keys from FIRST assistant reply of each surviving row.
    prefix_counts: Counter = Counter()
    for r in rows:
        key = prefix_key(assistant_contents(r)[0])
        if key:
            prefix_counts[key] += 1
    hot_prefixes = {k for k, v in prefix_counts.items() if v >= 5}

    seen_prefix_count: Counter = Counter()
    keep = []
    for r in rows:
        key = prefix_key(assistant_contents(r)[0])
        if key in hot_prefixes:
            seen_prefix_count[key] += 1
            if seen_prefix_count[key] > 2:
                report["dropped"]["template_repetition"].append(
                    (r.get("state", {}).get("intimacy", 0), assistant_contents(r)[0][:80])
                )
                report["prefix_drops"][key] += 1
                continue
        keep.append(r)
    rows = keep

    # Step 8: fix ? violations
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

    # Step 9: drop exact-duplicate assistant content (full joined assistant trace)
    seen_assist: set[str] = set()
    keep = []
    for r in rows:
        sig = "\n---\n".join(assistant_contents(r)).strip()
        if sig in seen_assist:
            report["dropped"]["duplicate_assistant"].append(
                (r.get("state", {}).get("intimacy", 0), assistant_contents(r)[0][:80])
            )
            continue
        seen_assist.add(sig)
        keep.append(r)
    rows = keep

    # Step 10: selectively drop duplicate first-user prompts
    # Group by first-user content, keep distinct intimacy buckets (±5).
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
                    (intim, first_user_content(r)[:80])
                )
                continue
            kept_intimacies.append(intim)
            keep_ids.add(id(r))
    rows = [r for r in rows if id(r) in keep_ids]

    # Step 11: mark source
    source_remap = {
        "synthetic_v2": "synthetic_v2_cleaned",
        "synthetic_v3": "synthetic_v3_cleaned",
        "handwritten": "handwritten_cleaned",
    }
    for r in rows:
        src = r.get("source", "")
        r["source"] = source_remap.get(src, src)

    # Step 12: final validation
    errors: list[str] = []
    for i, r in enumerate(rows):
        # schema keys
        if set(r.keys()) != {"messages", "emotion", "policy", "state", "source"}:
            errors.append(f"row {i}: unexpected top-level keys {sorted(r.keys())}")
        # system prompt
        sys_msgs = [m for m in r["messages"] if m["role"] == "system"]
        if not sys_msgs or sys_msgs[0]["content"] != TARGET_SYSTEM_PROMPT:
            errors.append(f"row {i}: system prompt mismatch")
        # emotion / mood / policy
        if r["emotion"] not in VALID_EMOTIONS:
            errors.append(f"row {i}: bad emotion {r['emotion']!r}")
        if r["state"]["mood"] not in VALID_MOODS:
            errors.append(f"row {i}: bad mood {r['state']['mood']!r}")
        if r["policy"] not in VALID_POLICIES:
            errors.append(f"row {i}: bad policy {r['policy']!r}")
        # assistant content constraints
        for a in assistant_contents(r):
            if a.count("?") > 1:
                errors.append(f"row {i}: `?` count > 1 in {a[:80]!r}")
            al = a.lstrip().lower()
            if (
                al.startswith("sounds like")
                or al.startswith("it sounds like")
                or al.startswith("that sounds like")
            ):
                errors.append(f"row {i}: still opens with sounds-like: {a[:80]!r}")
            for ch in a:
                if unicodedata.category(ch) == "So" or ord(ch) > 0xFFFF:
                    errors.append(f"row {i}: emoji/symbol U+{ord(ch):04X} in assistant")
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

    # Write cleaned output
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

    ap("# YLS Dataset Cleaning Report\n")
    ap(f"**Input:** `{INPUT_PATH.name}` ({report['input_count']} rows)")
    ap(f"**Output:** `{OUTPUT_PATH.name}` ({len(rows)} rows)\n")

    # 1. Summary
    ap("## 1. Summary Counts\n")
    ap(f"- Input rows: **{report['input_count']}**")
    ap(f"- Output rows: **{len(rows)}**")
    ap(f"- Total rows dropped: **{report['input_count'] - len(rows)}**\n")
    ap("### Drops by reason\n")
    ap("| Reason | Count |")
    ap("|---|---|")
    for k in ("cliche_opener", "template_repetition", "duplicate_assistant", "duplicate_user_same_intimacy"):
        ap(f"| {k.replace('_', ' ')} | {len(report['dropped'][k])} |")
    ap("")

    # 2. Emotion remap
    ap("## 2. Emotion Remapping\n")
    ap(f"Before: {len(report['emotion_before'])} unique values → After: {len(report['emotion_after'])} unique values\n")
    ap("### Before (all values)\n")
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
    ap(f"Before: {len(report['mood_before'])} unique values → After: {len(report['mood_after'])} unique values\n")
    ap("### Before (all values)\n")
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

    # 4. Dropped rows log
    ap("## 4. Dropped Rows\n")
    for reason_key, header in [
        ("cliche_opener", "Cliché opener (\"sounds like\" / \"it sounds like\" / \"that sounds like\")"),
        ("template_repetition", "Template prefix repetition (>=5 occurrences, kept first 2)"),
        ("duplicate_assistant", "Duplicate assistant content"),
        ("duplicate_user_same_intimacy", "Duplicate user prompt at same intimacy bucket (±5)"),
    ]:
        drops = report["dropped"][reason_key]
        ap(f"### {header} — {len(drops)} dropped\n")
        if not drops:
            ap("_none_\n")
            continue
        ap("| Intimacy | Preview (first 80 chars) |")
        ap("|---|---|")
        for intim, prev in drops:
            safe = prev.replace("|", "\\|")
            ap(f"| {intim} | {safe} |")
        ap("")

    # Prefix drop breakdown
    if report["prefix_drops"]:
        ap("### Prefix-repetition breakdown\n")
        ap("| Prefix | Rows dropped |")
        ap("|---|---|")
        for k, v in sorted(report["prefix_drops"].items(), key=lambda x: -x[1]):
            ap(f"| `{k}` | {v} |")
        ap("")

    # 5. ? fix log
    ap("## 5. `?` Rule Violations Fixed\n")
    if not report["q_fixes"]:
        ap("_none_\n")
    else:
        ap(f"Total fixed: **{len(report['q_fixes'])}**\n")
        for before, after in report["q_fixes"]:
            ap(f"- **Before:** {before}")
            ap(f"- **After:**  {after}\n")

    # 6. Mid-sentence sounds-like rewrites
    ap("## 6. Mid-sentence \"sounds like\" Rewrites\n")
    if not report["sounds_like_rewrites"]:
        ap("_none_\n")
    else:
        ap(f"Total rewrites: **{len(report['sounds_like_rewrites'])}**\n")
        for before, after in report["sounds_like_rewrites"]:
            ap(f"- **Before:** {before}")
            ap(f"- **After:**  {after}\n")

    # 7. Final label distributions
    ap("## 7. Final Label Distributions\n")
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

    # 8. Intimacy histogram
    ap("## 8. Intimacy Distribution (Post-clean)\n")
    buckets: Counter = Counter()
    for r in rows:
        intim = r["state"].get("intimacy", 0)
        buckets[(intim // 10) * 10] += 1
    max_count = max(buckets.values()) if buckets else 1
    ap("| Range | Count | Bar |")
    ap("|---|---|---|")
    for b in sorted(buckets):
        bar = "█" * int(40 * buckets[b] / max_count)
        ap(f"| {b}-{b + 9} | {buckets[b]} | {bar} |")
    ap("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
