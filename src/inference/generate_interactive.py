"""Interactive chat loop with the base Qwen2.5-1.5B-Instruct model.

Uses a fixed system prompt with no state, memory, or policy modules.
Type 'quit' to exit.

Usage:
    python -m src.inference.generate_interactive
"""

from src.inference.generate import load_model, generate_response

SYSTEM_PROMPT = "You are a warm and emotionally attentive AI companion."


def main() -> None:
    """Run an interactive chat loop."""
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded. Type 'quit' to exit.\n")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        response = generate_response(tokenizer, model, messages)
        print(f"Companion: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
