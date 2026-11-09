"""Single-head self-attention implemented with plain Python lists."""

import math


def self_attention(
    X: list[list[float]],
    W_Q: list[list[float]],
    W_K: list[list[float]],
    W_V: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Return attention outputs and weights for an input without padding.

    X has shape sequence_length x d_model. W_Q and W_K have shape
    d_model x d_k, while W_V has shape d_model x d_v.
    Outputs have shape sequence_length x d_v; weights have shape
    sequence_length x sequence_length. Positional information should
    already be included in X. Projection matrices are supplied by the caller.
    """
    if not X:
        return [], []

    def matmul(A, B):
        if not A or not B or not A[0] or not B[0]:
            raise ValueError("Matrices must have at least one row and column")
        if any(len(row) != len(A[0]) for row in A) or any(
            len(row) != len(B[0]) for row in B
        ):
            raise ValueError("Matrices must be rectangular")
        if len(A[0]) != len(B):
            raise ValueError("Matrix dimensions are incompatible for multiplication")

        columns = list(zip(*B))
        return [
            [sum(a * b for a, b in zip(row, column)) for column in columns] for row in A
        ]

    # Project the same input into queries, keys, and values.
    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)

    # Compute scaled similarity scores: QK^T / sqrt(d_k).
    d_k = len(Q[0])
    K_transposed = list(zip(*K))
    scores = matmul(Q, K_transposed)
    scores = [[score / math.sqrt(d_k) for score in row] for row in scores]

    # Apply a numerically stable softmax to each query's scores.
    weights = []
    for row in scores:
        maximum = max(row)
        exponentials = [math.exp(score - maximum) for score in row]
        total = sum(exponentials)
        weights.append([value / total for value in exponentials])

    # Mix value vectors according to the attention weights.
    output = matmul(weights, V)
    return output, weights
