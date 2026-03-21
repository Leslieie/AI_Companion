"""Emotion classifier for user input.

Classifies the primary emotion in a user message. This module will be
upgraded from keyword-based to model-based classification as the project
evolves.

Supported emotions: happy, sad, anxious, angry, lonely, neutral.
"""


def classify_emotion(user_message: str) -> str:
    """Classify the primary emotion in a user message.

    Args:
        user_message: The user's input text.

    Returns:
        One of: 'happy', 'sad', 'anxious', 'angry', 'lonely', 'neutral'.
    """
    text = user_message.lower()

    if any(w in text for w in ["happy", "great", "excited", "awesome", "wonderful"]):
        return "happy"
    if any(w in text for w in ["sad", "upset", "disappointed", "depressed", "cry"]):
        return "sad"
    if any(w in text for w in ["anxious", "worried", "stressed", "nervous", "scared"]):
        return "anxious"
    if any(w in text for w in ["angry", "furious", "annoyed", "frustrated", "mad"]):
        return "angry"
    if any(w in text for w in ["lonely", "alone", "isolated", "no one", "nobody"]):
        return "lonely"

    return "neutral"
