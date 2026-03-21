"""Builds the system prompt from persona, state, memory, and policy."""


def build_prompt(
    user_message: str,
    state: dict[str, int | str],
    memories: list[str],
    policy: str,
) -> list[dict[str, str]]:
    """Assemble a chat-format prompt with persona, state, memory, and policy.

    Args:
        user_message: The current user input.
        state: Current relationship state dict.
        memories: List of relevant memory strings.
        policy: The selected interaction policy.

    Returns:
        List of message dicts ready for the chat template.
    """
    persona_text = (
        "You are a text-only AI companion. "
        "You are warm, emotionally attentive, gentle, and slightly playful. "
        "You validate feelings before giving advice."
    )

    state_text = (
        f"affection={state.get('affection', 50)}, "
        f"trust={state.get('trust', 50)}, "
        f"mood={state.get('mood', 'neutral')}, "
        f"intimacy={state.get('intimacy', 50)}"
    )

    memory_text = "\n".join(f"- {m}" for m in memories) if memories else "- none"

    system_prompt = f"""{persona_text}

Current relationship state:
{state_text}

Relevant memories:
{memory_text}

Interaction policy:
- {policy}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
