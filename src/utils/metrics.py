"""Evaluation metrics for companion response quality.

Planned metrics:
- Perplexity on held-out data
- Emotion-appropriateness: does the response match the expected policy?
- Persona consistency: does the response align with the defined persona?
- Human evaluation aggregation: warmth, coherence, engagement scores
"""


def compute_perplexity(model_path: str, test_data_path: str) -> float:
    """Compute perplexity of a model on test data.

    Args:
        model_path: Path to the model checkpoint.
        test_data_path: Path to the test JSONL file.

    Returns:
        Perplexity score.
    """
    raise NotImplementedError("Implement after evaluation pipeline is set up.")


def emotion_appropriateness(
    predicted_policy: str, expected_policy: str
) -> float:
    """Score whether the predicted policy matches the expected one.

    Args:
        predicted_policy: The policy the model implicitly followed.
        expected_policy: The annotated ground-truth policy.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return 1.0 if predicted_policy == expected_policy else 0.0
