"""Evaluation script for comparing model variants.

Compares three configurations:
1. Plain model (no state/memory, vanilla Qwen2.5-1.5B-Instruct)
2. Stateful model (architecture modules, no fine-tuning)
3. Stateful + SFT model (architecture modules + LoRA fine-tuned)

Metrics to implement:
- Perplexity on held-out companion dialogues
- Emotion-appropriateness score (does the response match the expected policy?)
- Human evaluation scores (warmth, coherence, persona consistency)
"""


def evaluate_model(model_path: str, test_data_path: str) -> dict[str, float]:
    """Evaluate a model on the test set and return metric scores.

    Args:
        model_path: Path to the model or adapter checkpoint.
        test_data_path: Path to the test JSONL file.

    Returns:
        Dict mapping metric names to scores.
    """
    raise NotImplementedError("Implement after evaluation criteria are finalized.")


def compare_variants(results: dict[str, dict[str, float]]) -> None:
    """Print a comparison table of evaluation results across model variants.

    Args:
        results: Dict mapping variant name to its metric scores.
    """
    raise NotImplementedError("Implement after evaluation criteria are finalized.")
