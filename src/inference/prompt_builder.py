"""Builds the system prompt from persona, state, memory, and policy.

Loads persona text and system prompt template from configs/prompts.yaml.
Supports multi-turn conversation history.
"""

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts.yaml"


def _load_prompts_config(path: Path = CONFIG_PATH) -> dict[str, str]:
    """Load persona and template from YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(
    conversation_history: list[dict[str, str]],
    state: dict[str, int | str],
    memories: list[str],
    policy: str,
) -> list[dict[str, str]]:
    """Assemble a chat-format prompt with persona, state, memory, and policy.

    Args:
        conversation_history: List of {"role": ..., "content": ...} dicts
            representing the multi-turn conversation so far.
        state: Current relationship state dict.
        memories: List of relevant memory strings.
        policy: The selected interaction policy.

    Returns:
        List of message dicts ready for the chat template.
    """
    config = _load_prompts_config()
    persona = config.get("persona", "").strip()
    template = config.get("system_prompt_template", "")

    state_text = (
        f"affection={state.get('affection', 50)}, "
        f"trust={state.get('trust', 50)}, "
        f"mood={state.get('mood', 'neutral')}, "
        f"intimacy={state.get('intimacy', 50)}, "
        f"energy={state.get('energy', 70)}"
    )

    memory_text = "\n".join(f"- {m}" for m in memories) if memories else "- none"

    system_prompt = template.format(
        persona=persona,
        state=state_text,
        memories=memory_text,
        policy=policy,
    ).strip()

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    return messages
