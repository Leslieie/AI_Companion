"""Interactive CLI companion that ties together inference, state, memory, and policy.

Usage:
    python -m src.app

Type messages to chat with the companion. Type 'quit' or 'exit' to stop.
"""

from .inference.generate import load_model, generate_response
from .inference.prompt_builder import build_prompt
from .modules.state_tracker import StateTracker
from .modules.memory_store import MemoryStore
from .modules.policy_selector import select_policy
from .modules.emotion_classifier import classify_emotion


def _extract_memory(user_message: str, emotion: str) -> str | None:
    """Extract a simple memory from the user message.

    Keeps messages that are long enough to carry meaningful content.
    Prefixes with the detected emotion for context.

    Args:
        user_message: The user's input text.
        emotion: The classified emotion label.

    Returns:
        A memory string, or None if the message is too short to store.
    """
    if len(user_message.split()) < 3:
        return None
    return f"[{emotion}] {user_message}"


def main() -> None:
    """Run the interactive companion CLI loop."""
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded. Type 'quit' or 'exit' to stop.\n")

    tracker = StateTracker()
    memory = MemoryStore()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        emotion = classify_emotion(user_input)
        state = tracker.update(user_input)

        # Store a memory from this turn before retrieval
        mem = _extract_memory(user_input, emotion)
        if mem:
            memory.add(mem)

        memories = memory.retrieve(user_input)
        policy = select_policy(user_input, state)

        messages = build_prompt(user_input, state, memories, policy)
        response = generate_response(tokenizer, model, messages)

        print(f"Companion: {response}\n")
        print(f"  [emotion={emotion}, policy={policy}, mood={state['mood']}, memories={memory.size()}]")
        print()


if __name__ == "__main__":
    main()
