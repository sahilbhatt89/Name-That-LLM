"""Numerical and shape checks for per-token layer normalization."""

import copy
import math

import pytest

from name_that_llm.normalization import layer_norm


def test_each_token_uses_its_own_statistics():
    output = layer_norm([[1.0, 3.0], [101.0, 103.0]], [1.0, 1.0], [0.0, 0.0])
    magnitude = 1 / math.sqrt(1 + 1e-5)

    for row in output:
        assert row == pytest.approx([-magnitude, magnitude])
        assert sum(row) / len(row) == pytest.approx(0.0, abs=1e-12)
        assert sum(value**2 for value in row) / len(row) == pytest.approx(1 / 1.00001)


def test_applies_per_feature_scale_and_shift_with_epsilon_inside_square_root():
    output = layer_norm([[1.0, 3.0]], [2.0, 3.0], [10.0, -4.0], epsilon=3.0)
    assert output == [[9.0, -2.5]]


@pytest.mark.parametrize("row", [[7.0, 7.0], [7.0]])
def test_constant_token_vector_returns_beta_without_dividing_by_zero(row):
    beta = [2.0] * len(row)
    assert layer_norm([row], [3.0] * len(row), beta) == [beta]


def test_does_not_modify_or_alias_inputs():
    X = [[1.0, 3.0], [2.0, 4.0]]
    gamma = [1.0, 1.0]
    beta = [0.0, 0.0]
    original = copy.deepcopy((X, gamma, beta))

    output = layer_norm(X, gamma, beta)
    assert (X, gamma, beta) == original
    other_row = output[1].copy()
    output[0][0] = 100.0
    assert (X, gamma, beta) == original
    assert output[1] == other_row


def test_empty_sequence_returns_empty_output():
    assert layer_norm([], [1.0, 1.0], [0.0, 0.0]) == []


@pytest.mark.parametrize(
    "X, gamma, beta",
    [
        ([[1.0, 2.0]], [1.0], [0.0]),
        ([[1.0, 2.0], [3.0]], [1.0, 1.0], [0.0, 0.0]),
        ([[1.0]], [1.0], [0.0, 0.0]),
        ([[]], [], []),
    ],
)
def test_invalid_feature_dimensions_are_rejected(X, gamma, beta):
    with pytest.raises(ValueError, match="length"):
        layer_norm(X, gamma, beta)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, math.inf, math.nan])
def test_epsilon_must_be_finite_and_positive(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        layer_norm([[1.0]], [1.0], [0.0], epsilon=epsilon)
