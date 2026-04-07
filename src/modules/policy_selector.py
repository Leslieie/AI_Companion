"""Rule-based interaction policy selection.

Chooses a response strategy (comforting, playful, neutral, reflective, tense)
based on the classified emotion label and current companion state.
"""

# Map emotion labels to default policies
EMOTION_POLICY_MAP: dict[str, str] = {
    "sad": "comforting",
    "anxious": "comforting",
    "lonely": "comforting",
    "angry": "tense",
    "happy": "playful",
}


def select_policy(emotion: str, state: dict[str, int | str]) -> str:
    """Select an interaction policy based on emotion label and state.

    Uses the classified emotion as primary signal. Falls back to
    state-based heuristics when the emotion is neutral.

    Args:
        emotion: Classified emotion label from emotion_classifier.
        state: Current companion state dict.

    Returns:
        One of: 'comforting', 'playful', 'neutral', 'reflective', 'tense'.
    """
    if emotion in EMOTION_POLICY_MAP:
        return EMOTION_POLICY_MAP[emotion]

    # Fall back to state-based selection
    if state.get("mood") == "concerned":
        return "reflective"
    if state.get("energy", 70) < 30:
        return "reflective"

    return "neutral"
