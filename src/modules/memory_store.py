"""Simple in-memory store for companion memories.

Stores long-term memories as a list of strings with basic
recency-based retrieval. Will be upgraded to embedding-based
retrieval in a future iteration.
"""


class MemoryStore:
    """In-memory store for long-term companion memories."""

    def __init__(self) -> None:
        self.long_term_memories: list[str] = []

    def add(self, memory: str) -> None:
        """Add a memory if it is non-empty and not already stored.

        Args:
            memory: A fact or observation to remember.
        """
        if memory and memory not in self.long_term_memories:
            self.long_term_memories.append(memory)

    def retrieve(self, user_message: str, top_k: int = 3) -> list[str]:
        """Retrieve the most recent memories.

        Args:
            user_message: Current user input (unused in MVP, reserved for
                semantic retrieval in future versions).
            top_k: Number of memories to return.

        Returns:
            List of memory strings, most recent last.
        """
        return self.long_term_memories[-top_k:]

    def clear(self) -> None:
        """Clear all stored memories."""
        self.long_term_memories.clear()

    def size(self) -> int:
        """Return the number of stored memories."""
        return len(self.long_term_memories)
