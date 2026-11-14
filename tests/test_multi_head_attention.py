"""Numerical and shape checks for multi-head attention."""

import copy
import math

import pytest

from name_that_llm.attention import multi_head_attention, self_attention
from name_that_llm.residual import residual_add


def test_two_heads_have_distinct_weights_and_project_concatenated_values():
    X = [[1.0, 0.0], [0.0, 1.0]]
    W_Q_heads = [[[1.0], [0.0]], [[0.0], [1.0]]]
    W_K_heads = copy.deepcopy(W_Q_heads)
    W_V_heads = [[[2.0], [4.0]], [[10.0], [20.0]]]
    W_O = [[1.0, 2.0], [3.0, 4.0]]
    original_inputs = copy.deepcopy((X, W_Q_heads, W_K_heads, W_V_heads, W_O))

    output, weights = multi_head_attention(X, W_Q_heads, W_K_heads, W_V_heads, W_O)

    # Head 0 favors token 0 for its first query; head 1 favors token 1
    # for its second query. The other query in each head has equal scores.
    p = math.e / (math.e + 1)
    assert weights[0][0] == pytest.approx([p, 1 - p])
    assert weights[0][1] == pytest.approx([0.5, 0.5])
    assert weights[1][0] == pytest.approx([0.5, 0.5])
    assert weights[1][1] == pytest.approx([1 - p, p])
    assert output[0] == pytest.approx([49 - 2 * p, 68 - 4 * p])
    assert output[1] == pytest.approx([33 + 30 * p, 46 + 40 * p])
    assert all(sum(row) == pytest.approx(1.0) for head in weights for row in head)
    assert (X, W_Q_heads, W_K_heads, W_V_heads, W_O) == original_inputs

    residual = residual_add(X, output)
    assert residual[0] == pytest.approx([50 - 2 * p, 68 - 4 * p])
    assert residual[1] == pytest.approx([33 + 30 * p, 47 + 40 * p])


def test_one_head_with_identity_output_projection_matches_self_attention():
    X = [[2.0, 1.0], [-1.0, 3.0]]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    expected_output, expected_weights = self_attention(X, identity, identity, identity)

    output, weights = multi_head_attention(
        X, [identity], [identity], [identity], identity
    )

    assert output == expected_output
    assert weights == [expected_weights]


def test_empty_input_keeps_one_empty_weight_matrix_per_head():
    heads = [[[1.0]], [[2.0]]]
    assert multi_head_attention([], heads, heads, heads, [[1.0], [1.0]]) == (
        [],
        [[], []],
    )


def test_zero_heads_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        multi_head_attention([[1.0]], [], [], [], [[1.0]])


@pytest.mark.parametrize("missing_projection", ["keys", "values"])
def test_mismatched_head_counts_are_rejected(missing_projection):
    heads = [[[1.0]], [[2.0]]]
    keys = heads[:1] if missing_projection == "keys" else heads
    values = heads[:1] if missing_projection == "values" else heads

    with pytest.raises(ValueError, match="same head count"):
        multi_head_attention([[1.0]], heads, keys, values, [[1.0], [1.0]])


@pytest.mark.parametrize(
    "W_O",
    [[], [[1.0]], [[1.0, 0.0], [0.0, 1.0]]],
)
def test_invalid_output_projection_shapes_are_rejected(W_O):
    heads = [[[1.0]], [[2.0]]]
    with pytest.raises(ValueError, match="Matrices|Matrix|d_model"):
        multi_head_attention([[1.0]], heads, heads, heads, W_O)
