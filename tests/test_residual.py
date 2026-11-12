"""Checks for residual addition and its shape requirements."""

import pytest

from name_that_llm.residual import residual_add


def test_residual_adds_matching_elements():
    X = [[1.0, -2.0], [0.5, 0.0]]
    attention_output = [[0.25, 2.0], [-0.5, -3.0]]

    assert residual_add(X, attention_output) == [[1.25, 0.0], [0.0, -3.0]]


def test_residual_preserves_inputs_and_returns_independent_rows():
    X = [[1.0, 2.0], [3.0, 4.0]]
    attention_output = [[0.5, 0.25], [-1.0, 2.0]]

    result = residual_add(X, attention_output)

    assert X == [[1.0, 2.0], [3.0, 4.0]]
    assert attention_output == [[0.5, 0.25], [-1.0, 2.0]]

    result[0][0] = 100.0
    assert X[0][0] == 1.0
    assert attention_output[0][0] == 0.5
    assert result[1] == [2.0, 6.0]


@pytest.mark.parametrize("X, attention_output", [([], []), ([[]], [[]])])
def test_residual_accepts_matching_empty_inputs(X, attention_output):
    assert residual_add(X, attention_output) == X


@pytest.mark.parametrize(
    "X, attention_output",
    [
        ([[1.0]], []),
        ([], [[1.0]]),
    ],
)
def test_residual_rejects_mismatched_row_counts(X, attention_output):
    with pytest.raises(ValueError, match="same number of rows"):
        residual_add(X, attention_output)


@pytest.mark.parametrize(
    "X, attention_output",
    [
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0], [2.0]], [[1.0], [2.0, 3.0]]),
    ],
)
def test_residual_rejects_mismatched_row_lengths(X, attention_output):
    with pytest.raises(ValueError, match="same length"):
        residual_add(X, attention_output)
