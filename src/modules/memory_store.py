"""Simple in-memory store for companion memories.

Stores long-term memories as dicts with fact, source_turn, and timestamp.
Supports keyword-based memory extraction and recency-based retrieval.
Will be upgraded to embedding-based retrieval in a future iteration.
"""

from datetime import datetime


# Trigger phrases for extracting personal facts from user messages.
# Based on docs/memory_schema.md extraction strategy.
MEMORY_TRIGGERS: list[str] = [
    "i am", "i'm", "i like", "i love", "i have", "i hate",
    "i want", "i need", "my name is", "my favorite",
    "i prefer", "i work", "i study", "i live",
]


class MemoryStore:
    """In-memory store for long-term companion memories.

    Memories are stored as dicts with keys: fact, source_turn, timestamp.
    Maximum capacity is 50 entries; oldest is dropped when full.
    """

    MAX_ENTRIES: int = 50

    def __init__(self) -> None:
        self.long_term_memories: list[dict[str, str | int]] = []

    def add(self, fact: str, source_turn: int) -> None:
        """Add a memory if it is non-empty and not a duplicate.

        When the store is at capacity, the oldest entry is dropped first.

        Args:
            fact: A fact or observation to remember.
            source_turn: The turn number when this fact was observed.
        """
        if not fact:
            return
        if any(m["fact"] == fact for m in self.long_term_memories):
            return
        if len(self.long_term_memories) >= self.MAX_ENTRIES:
            self.long_term_memories.pop(0)
        self.long_term_memories.append({
            "fact": fact,
            "source_turn": source_turn,
            "timestamp": datetime.now().isoformat(),
        })

    def extract_and_store(self, user_message: str, turn: int) -> str | None:
        """Extract a memory from the user message using keyword triggers.

        Scans for trigger phrases like "I am", "I like", "I have" and
        stores the full message as a long-term memory fact.

        Args:
            user_message: The user's input text.
            turn: The current turn number.

        Returns:
            The extracted fact string, or None if nothing was extracted.
        """
        text_lower = user_message.lower()
        for trigger in MEMORY_TRIGGERS:
            if trigger in text_lower:
                fact = user_message.strip()
                self.add(fact, turn)
                return fact
        return None

    def retrieve(self, user_message: str, top_k: int = 3) -> list[str]:
        """Retrieve the most recent memory facts.

        Args:
            user_message: Current user input (unused in MVP, reserved for
                semantic retrieval in future versions).
            top_k: Number of memories to return.

        Returns:
            List of fact strings, most recent last.
        """
        return [m["fact"] for m in self.long_term_memories[-top_k:]]

    def clear(self) -> None:
        """Clear all stored memories."""
        self.long_term_memories.clear()

    def size(self) -> int:
        """Return the number of stored memories."""
        return len(self.long_term_memories)
