"""Numerical checks for single-head self-attention."""

import math

import pytest

from name_that_llm.attention import self_attention


def test_attention_matches_two_token_example_with_distinct_value_dimension():
    X = [[1.0, 0.0], [0.0, 1.0]]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    W_V = [[2.0, 0.0, 1.0], [0.0, 4.0, -1.0]]

    output, weights = self_attention(X, identity, identity, W_V)

    # Self-scores are 1/sqrt(2); cross-token scores are zero.
    p = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    assert weights[0] == pytest.approx([p, 1 - p])
    assert weights[1] == pytest.approx([1 - p, p])
    assert output[0] == pytest.approx([2 * p, 4 * (1 - p), 2 * p - 1])
    assert output[1] == pytest.approx([2 * (1 - p), 4 * p, 1 - 2 * p])
    assert all(sum(row) == pytest.approx(1.0) for row in weights)
    assert X == [[1.0, 0.0], [0.0, 1.0]]
    assert W_V == [[2.0, 0.0, 1.0], [0.0, 4.0, -1.0]]


def test_equal_scores_average_value_vectors():
    output, weights = self_attention([[1.0], [3.0]], [[0.0]], [[1.0]], [[2.0]])
    assert weights == [[0.5, 0.5], [0.5, 0.5]]
    assert output == [[4.0], [4.0]]


def test_large_scores_do_not_overflow_softmax():
    output, weights = self_attention([[1000.0], [-1000.0]], [[1.0]], [[1.0]], [[1.0]])
    assert weights == [[1.0, 0.0], [0.0, 1.0]]
    assert output == [[1000.0], [-1000.0]]


def test_empty_input_returns_empty_outputs():
    assert self_attention([], [[1.0]], [[1.0]], [[1.0]]) == ([], [])


@pytest.mark.parametrize(
    "X, W_Q, W_K, W_V",
    [
        ([[1.0], [1.0, 2.0]], [[1.0]], [[1.0]], [[1.0]]),
        ([[1.0]], [], [[1.0]], [[1.0]]),
        ([[1.0]], [[1.0], [2.0]], [[1.0]], [[1.0]]),
        ([[1.0]], [[1.0, 2.0]], [[1.0]], [[1.0]]),
    ],
)
def test_invalid_matrix_shapes_raise_clear_errors(X, W_Q, W_K, W_V):
    with pytest.raises(ValueError, match="Matrices|Matrix"):
        self_attention(X, W_Q, W_K, W_V)
