"""Integration checks for post-normalized encoder layers."""

import random

import pytest

from name_that_llm.attention import multi_head_attention
from name_that_llm.encoder import EncoderLayer, encoder_stack
from name_that_llm.feed_forward import feed_forward
from name_that_llm.normalization import layer_norm
from name_that_llm.residual import residual_add


def test_stack_matches_two_manual_layers_with_distinct_parameters():
    rng = random.Random(42)
    layers = [EncoderLayer.initialize(4, 2, 8, rng) for _ in range(2)]
    X = [[1.0, 0.5, -0.5, 2.0], [0.0, -1.0, 0.5, 1.0]]
    expected = X
    expected_weights = []
    for layer in layers:
        # Distinct shifts make a mix-up between the two normalizations visible.
        layer.attention_beta[0] = 0.25
        layer.ff_beta[1] = -0.5
        attended, weights = multi_head_attention(
            expected, layer.W_Q_heads, layer.W_K_heads, layer.W_V_heads, layer.W_O
        )
        hidden = layer_norm(
            residual_add(expected, attended),
            layer.attention_gamma,
            layer.attention_beta,
        )
        transformed = feed_forward(hidden, layer.W_1, layer.b_1, layer.W_2, layer.b_2)
        expected = layer_norm(
            residual_add(hidden, transformed), layer.ff_gamma, layer.ff_beta
        )
        expected_weights.append(weights)

    assert layers[0].W_Q_heads != layers[1].W_Q_heads
    assert encoder_stack(X, layers) == (expected, expected_weights)
    assert encoder_stack([], layers) == ([], [[[], []], [[], []]])


@pytest.mark.parametrize("dimensions", [(0, 2, 8), (4, 0, 8), (4, 2, 0), (3, 2, 8)])
def test_invalid_layer_dimensions(dimensions):
    with pytest.raises(ValueError):
        EncoderLayer.initialize(*dimensions, random.Random(42))


def test_empty_stack():
    with pytest.raises(ValueError, match="at least one"):
        encoder_stack([], [])
