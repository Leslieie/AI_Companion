"""Rule-based relationship state tracker.

Maintains affection, trust, intimacy, mood, and energy variables
that evolve based on user input and conversation history.
Update rules are loaded from configs/state_rules.yaml.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "state_rules.yaml"


def _load_rules(path: Path = CONFIG_PATH) -> dict:
    """Load state update rules from YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    """Clamp an integer to [lo, hi]."""
    return max(lo, min(hi, value))


@dataclass
class CompanionState:
    """Represents the current relationship state between companion and user."""

    affection: int = 50
    trust: int = 50
    intimacy: int = 50
    mood: str = "neutral"
    energy: int = 70


class StateTracker:
    """Tracks and updates companion relationship state based on user messages.

    Loads keyword rules from configs/state_rules.yaml. Also applies
    turn-count-based intimacy growth and short-message energy decay.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        rules = _load_rules(config_path)
        defaults = rules.get("defaults", {})
        self.state = CompanionState(
            affection=defaults.get("affection", 50),
            trust=defaults.get("trust", 50),
            intimacy=defaults.get("intimacy", 50),
            mood=defaults.get("mood", "neutral"),
            energy=defaults.get("energy", 70),
        )
        self._keywords: dict = rules.get("keywords", {})
        clamp_cfg = rules.get("clamp", {})
        self._lo: int = clamp_cfg.get("min", 0)
        self._hi: int = clamp_cfg.get("max", 100)
        self.turn_count: int = 0
        self._short_msg_streak: int = 0

    def update(self, user_message: str) -> dict[str, int | str]:
        """Update state based on the user message and return the new state.

        Applies keyword rules from config, then turn-based intimacy
        growth (after 10 turns) and short-message energy decay (3+
        consecutive messages under 5 words).

        Args:
            user_message: The user's input text.

        Returns:
            Dict representation of the updated state.
        """
        self.turn_count += 1
        text = user_message.lower()

        # Apply keyword-based rules from config
        for _category, rule in self._keywords.items():
            words = rule.get("words", [])
            effects = rule.get("effects", {})
            if any(w in text for w in words):
                for var, delta in effects.items():
                    if var == "mood":
                        self.state.mood = str(delta)
                    else:
                        current = getattr(self.state, var, None)
                        if current is not None:
                            setattr(
                                self.state, var,
                                _clamp(current + int(delta), self._lo, self._hi),
                            )

        # Turn-count-based intimacy growth (after 10 turns)
        if self.turn_count > 10:
            self.state.intimacy = _clamp(
                self.state.intimacy + 1, self._lo, self._hi,
            )

        # Short-message energy decay (3+ consecutive short messages)
        if len(user_message.split()) < 5:
            self._short_msg_streak += 1
        else:
            self._short_msg_streak = 0
        if self._short_msg_streak >= 3:
            self.state.energy = _clamp(
                self.state.energy - 5, self._lo, self._hi,
            )

        return asdict(self.state)

    def get_state(self) -> dict[str, int | str]:
        """Return the current state as a dict."""
        return asdict(self.state)

    def reset(self) -> None:
        """Reset state to defaults from config."""
        rules = _load_rules()
        defaults = rules.get("defaults", {})
        self.state = CompanionState(
            affection=defaults.get("affection", 50),
            trust=defaults.get("trust", 50),
            intimacy=defaults.get("intimacy", 50),
            mood=defaults.get("mood", "neutral"),
            energy=defaults.get("energy", 70),
        )
        self.turn_count = 0
        self._short_msg_streak = 0
