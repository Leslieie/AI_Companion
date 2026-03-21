"""Rule-based relationship state tracker.

Maintains affection, trust, intimacy, mood, and energy variables
that evolve based on user input and conversation history.
"""

from dataclasses import dataclass, asdict


@dataclass
class CompanionState:
    """Represents the current relationship state between companion and user."""

    affection: int = 50
    trust: int = 50
    intimacy: int = 50
    mood: str = "neutral"
    energy: int = 70


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    """Clamp an integer to [lo, hi]."""
    return max(lo, min(hi, value))


class StateTracker:
    """Tracks and updates companion relationship state based on user messages."""

    def __init__(self) -> None:
        self.state = CompanionState()

    def update(self, user_message: str) -> dict[str, int | str]:
        """Update state based on the user message and return the new state.

        Args:
            user_message: The user's input text.

        Returns:
            Dict representation of the updated state.
        """
        text = user_message.lower()

        if any(w in text for w in ["sad", "upset", "overwhelmed", "tired", "stressed"]):
            self.state.mood = "concerned"
            self.state.trust = _clamp(self.state.trust + 1)

        if any(w in text for w in ["thank you", "thanks", "appreciate"]):
            self.state.affection = _clamp(self.state.affection + 2)

        if any(w in text for w in ["leave me alone", "annoying", "stop", "shut up"]):
            self.state.affection = _clamp(self.state.affection - 3)
            self.state.trust = _clamp(self.state.trust - 2)

        if any(w in text for w in ["happy", "great", "excited", "awesome"]):
            self.state.mood = "cheerful"
            self.state.affection = _clamp(self.state.affection + 1)

        if any(w in text for w in ["haha", "funny", "joke", "lol"]):
            self.state.energy = _clamp(self.state.energy + 2)

        return asdict(self.state)

    def get_state(self) -> dict[str, int | str]:
        """Return the current state as a dict."""
        return asdict(self.state)

    def reset(self) -> None:
        """Reset state to defaults."""
        self.state = CompanionState()
