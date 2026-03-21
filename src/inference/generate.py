"""Minimal inference script using Qwen2.5-1.5B-Instruct.

Run this script to verify the model loads and generates a response:
    python -m src.inference.generate
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(model_name: str = MODEL_NAME) -> tuple:
    """Load tokenizer and model from Hugging Face or local cache.

    Args:
        model_name: Model identifier or local path.

    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def generate_response(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: list[dict[str, str]],
    max_new_tokens: int = 160,
    temperature: float = 0.8,
) -> str:
    """Generate a response given a list of chat messages.

    Args:
        tokenizer: The tokenizer instance.
        model: The language model instance.
        messages: List of message dicts with 'role' and 'content' keys.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The generated response text.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main() -> None:
    """Run a single inference test."""
    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": "You are a warm and emotionally attentive AI companion.",
        },
        {
            "role": "user",
            "content": "I feel kind of overwhelmed today.",
        },
    ]

    response = generate_response(tokenizer, model, messages)
    print(response)


if __name__ == "__main__":
    main()
