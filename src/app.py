"""Interactive CLI companion that ties together inference, state, memory, and policy.

Pipeline per turn:
    emotion_classifier -> state_tracker -> memory extract + retrieve
    -> policy_selector -> prompt_builder -> generate

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

# Short-term memory window: last 10 turns (20 messages)
MAX_HISTORY_MESSAGES: int = 20


def main() -> None:
    """Run the interactive companion CLI loop."""
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded. Type 'quit' or 'exit' to stop.\n")

    tracker = StateTracker()
    memory = MemoryStore()
    history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # 1. Perceive: classify emotion
        emotion = classify_emotion(user_input)

        # 2. Update relationship state
        state = tracker.update(user_input)

        # 3. Memory: extract new facts, then retrieve relevant ones
        memory.extract_and_store(user_input, tracker.turn_count)
        memories = memory.retrieve(user_input)

        # 4. Select policy using emotion label + state
        policy = select_policy(emotion, state)

        # 5. Append user message to conversation history
        history.append({"role": "user", "content": user_input})

        # 6. Build prompt with recent history (last N messages)
        recent = history[-MAX_HISTORY_MESSAGES:]
        messages = build_prompt(recent, state, memories, policy)

        # 7. Generate response
        response = generate_response(tokenizer, model, messages)

        # 8. Record assistant response in history
        history.append({"role": "assistant", "content": response})

        # 9. Output response and state debug info
        print(f"Ari: {response}\n")
        print(
            f"  [turn={tracker.turn_count} | emotion={emotion} | policy={policy} | "
            f"mood={state['mood']} | aff={state['affection']} | "
            f"trust={state['trust']} | intm={state['intimacy']} | "
            f"energy={state['energy']} | mem={memory.size()}]"
        )
        print()


if __name__ == "__main__":
    main()
