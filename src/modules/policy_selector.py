"""Rule-based interaction policy selection.

Chooses a response strategy (comforting, playful, neutral, reflective, tense)
based on user input and current companion state.
"""


def select_policy(user_message: str, state: dict[str, int | str]) -> str:
    """Select an interaction policy based on user message and state.

    Args:
        user_message: The user's input text.
        state: Current companion state dict.

    Returns:
        One of: 'comforting', 'playful', 'neutral', 'reflective', 'tense'.
    """
    text = user_message.lower()

    if any(w in text for w in ["sad", "hurt", "overwhelmed", "tired", "stressed"]):
        return "comforting"
    if any(w in text for w in ["haha", "funny", "joke", "lol"]):
        return "playful"
    if any(w in text for w in ["angry", "furious", "leave me alone", "shut up"]):
        return "tense"
    if state.get("mood") == "concerned":
        return "reflective"

    return "neutral"
