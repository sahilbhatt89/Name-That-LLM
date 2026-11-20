"""Position-wise Transformer feed-forward network using plain Python lists."""


def feed_forward(
    X: list[list[float]],
    W_1: list[list[float]],
    b_1: list[float],
    W_2: list[list[float]],
    b_2: list[float],
) -> list[list[float]]:
    """Apply max(0, X W_1 + b_1) W_2 + b_2 independently to each token.

    X has shape sequence_length x d_model. W_1 has shape d_model x d_ff
    and W_2 has shape d_ff x d_model. Bias vectors have lengths d_ff
    and d_model, respectively. The same parameters are shared across all
    token positions. Initialize them once and reuse them for every input.
    This function performs the forward pass only; it does not train parameters.
    """
    d_model = len(b_2)
    d_ff = len(b_1)
    if d_model == 0 or d_ff == 0:
        raise ValueError("Bias vectors must have positive lengths")
    if len(W_1) != d_model or any(len(row) != d_ff for row in W_1):
        raise ValueError("W_1 must have shape d_model x d_ff")
    if len(W_2) != d_ff or any(len(row) != d_model for row in W_2):
        raise ValueError("W_2 must have shape d_ff x d_model")
    if any(len(row) != d_model for row in X):
        raise ValueError("Each input row must have d_model features")

    output = []
    for row in X:
        # Expand the features, add bias, and apply ReLU.
        hidden = [
            max(0.0, sum(row[i] * W_1[i][j] for i in range(d_model)) + b_1[j])
            for j in range(d_ff)
        ]

        # Project back to d_model. There is no activation after this projection.
        output.append(
            [
                sum(hidden[i] * W_2[i][j] for i in range(d_ff)) + b_2[j]
                for j in range(d_model)
            ]
        )

    return output
