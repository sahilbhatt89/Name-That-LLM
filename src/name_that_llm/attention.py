"""Single-head and multi-head self-attention using plain Python lists."""

import math


def _matmul(A, B):
    """Multiply rectangular matrices after checking their dimensions."""
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

    # Project the same input into queries, keys, and values.
    Q = _matmul(X, W_Q)
    K = _matmul(X, W_K)
    V = _matmul(X, W_V)

    # Compute scaled similarity scores: QK^T / sqrt(d_k).
    d_k = len(Q[0])
    K_transposed = list(zip(*K))
    scores = _matmul(Q, K_transposed)
    scores = [[score / math.sqrt(d_k) for score in row] for row in scores]

    # Apply a numerically stable softmax to each query's scores.
    weights = []
    for row in scores:
        maximum = max(row)
        exponentials = [math.exp(score - maximum) for score in row]
        total = sum(exponentials)
        weights.append([value / total for value in exponentials])

    # Mix value vectors according to the attention weights.
    output = _matmul(weights, V)
    return output, weights


def multi_head_attention(
    X: list[list[float]],
    W_Q_heads: list[list[list[float]]],
    W_K_heads: list[list[list[float]]],
    W_V_heads: list[list[list[float]]],
    W_O: list[list[float]],
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """Run independent attention heads, concatenate them, and project to d_model.

    X has shape sequence_length x d_model and should already include positions.
    Each projection list contains one matrix per head. For each head, W_Q and
    W_K have shape d_model x d_k; W_V has shape d_model x d_v. Typically
    d_k = d_v = d_model / num_heads. Every head receives the full input X.

    W_O has shape (sum of head value dimensions) x d_model. All projection
    matrices are parameters initialized once and supplied by the caller.

    Returns outputs of shape sequence_length x d_model and attention weights
    of shape num_heads x sequence_length x sequence_length. An empty input
    returns an empty output and an empty weight matrix per head. Padding and
    causal masks are not supported yet.
    """
    num_heads = len(W_Q_heads)
    if num_heads == 0:
        raise ValueError("At least one attention head is required")
    if len(W_K_heads) != num_heads or len(W_V_heads) != num_heads:
        raise ValueError(
            "Query, key, and value projections must have the same head count"
        )
    if not X:
        return [], [[] for _ in range(num_heads)]

    head_outputs = []
    head_weights = []
    for W_Q, W_K, W_V in zip(W_Q_heads, W_K_heads, W_V_heads):
        output, weights = self_attention(X, W_Q, W_K, W_V)
        head_outputs.append(output)
        head_weights.append(weights)

    # For each token, join its head vectors in head order.
    concatenated = [
        [value for head in head_outputs for value in head[position]]
        for position in range(len(X))
    ]

    # Restore d_model features so residual addition can use the original X.
    if any(len(row) != len(X[0]) for row in W_O):
        raise ValueError("W_O must project the concatenated heads back to d_model")
    output = _matmul(concatenated, W_O)
    return output, head_weights
