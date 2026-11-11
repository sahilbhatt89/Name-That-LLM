"""Residual addition implemented with plain Python lists."""


def residual_add(
    X: list[list[float]], attention_output: list[list[float]]
) -> list[list[float]]:
    """Add matching elements without modifying either input."""
    if len(X) != len(attention_output):
        raise ValueError("Inputs must have the same number of rows")

    result = []
    for input_row, attention_row in zip(X, attention_output):
        if len(input_row) != len(attention_row):
            raise ValueError("Corresponding rows must have the same length")

        result.append([x + a for x, a in zip(input_row, attention_row)])

    return result
