"""Numerical and shape checks for the position-wise feed-forward network."""

import copy

import pytest

from name_that_llm.feed_forward import feed_forward
from name_that_llm.normalization import layer_norm
from name_that_llm.residual import residual_add


def test_two_projections_relu_and_biases_match_hand_calculation():
    X = [[1.0, -2.0], [-1.0, 2.0]]
    W_1 = [[1.0, -1.0, 2.0], [2.0, 1.0, 0.0]]
    b_1 = [1.0, 0.0, -1.0]
    W_2 = [[1.0, 2.0], [-1.0, 0.0], [0.5, -2.0]]
    b_2 = [0.25, -0.5]
    original = copy.deepcopy((X, W_1, b_1, W_2, b_2))

    # Hidden vectors are [0, 0, 1] and [4, 3, 0]. Negative final outputs
    # are preserved because ReLU is applied only between the projections.
    output = feed_forward(X, W_1, b_1, W_2, b_2)
    assert output == [[0.75, -2.5], [1.25, 7.5]]
    assert (X, W_1, b_1, W_2, b_2) == original

    normalized = layer_norm(residual_add(X, output), [1.0, 1.0], [0.0, 0.0])
    assert len(normalized) == len(X)
    assert all(len(row) == 2 for row in normalized)
    assert all(sum(row) == pytest.approx(0.0, abs=1e-12) for row in normalized)


def test_tokens_are_processed_independently_with_shared_parameters():
    params = ([[2.0, -1.0]], [0.5, 0.0], [[3.0], [4.0]], [-1.0])
    assert feed_forward([[1.0], [-2.0], [1.0]], *params) == [[6.5], [7.0], [6.5]]
    assert feed_forward([[-2.0]], *params) == [[7.0]]


def test_empty_sequence_returns_empty_output():
    assert feed_forward([], [[1.0]], [0.0], [[1.0]], [0.0]) == []


@pytest.mark.parametrize(
    "X, W_1, b_1, W_2, b_2",
    [
        ([[1.0]], [], [0.0], [[1.0]], [0.0]),
        ([[1.0]], [[1.0, 2.0]], [0.0], [[1.0]], [0.0]),
        ([[1.0]], [[1.0]], [0.0], [], [0.0]),
        ([[1.0]], [[1.0]], [0.0], [[1.0, 2.0]], [0.0]),
        ([[1.0], [1.0, 2.0]], [[1.0]], [0.0], [[1.0]], [0.0]),
        ([[1.0]], [[]], [], [], [0.0]),
        ([[]], [], [0.0], [[]], []),
    ],
)
def test_invalid_dimensions_are_rejected(X, W_1, b_1, W_2, b_2):
    with pytest.raises(ValueError):
        feed_forward(X, W_1, b_1, W_2, b_2)
