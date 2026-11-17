"""Layer normalization implemented with plain Python lists."""

import math


def layer_norm(
    X: list[list[float]],
    gamma: list[float],
    beta: list[float],
    epsilon: float = 1e-5,
) -> list[list[float]]:
    """Normalize each token across its features, then apply scale and shift.

    X has shape sequence_length x d_model. gamma and beta each contain
    d_model parameters shared across tokens. Initialize gamma to ones and
    beta to zeros once, then reuse them for subsequent inputs. This function
    performs the forward calculation only and does not update parameters.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    d_model = len(gamma)
    if d_model == 0 or len(beta) != d_model:
        raise ValueError("gamma and beta must have the same positive length")
    if any(len(row) != d_model for row in X):
        raise ValueError("Each input row must match the length of gamma and beta")

    output = []
    for row in X:
        mean = sum(row) / d_model
        variance = sum((value - mean) ** 2 for value in row) / d_model
        denominator = math.sqrt(variance + epsilon)

        output.append(
            [gamma[j] * (row[j] - mean) / denominator + beta[j] for j in range(d_model)]
        )

    return output
