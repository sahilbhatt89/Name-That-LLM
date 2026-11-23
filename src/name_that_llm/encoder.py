"""Forward pass for a stack of post-normalization Transformer encoder layers."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .attention import multi_head_attention
from .feed_forward import feed_forward
from .normalization import layer_norm
from .residual import residual_add

Matrix = list[list[float]]


@dataclass
class EncoderLayer:
    """Parameters for one layer operating on unpadded, position-aware inputs."""

    W_Q_heads: list[Matrix]
    W_K_heads: list[Matrix]
    W_V_heads: list[Matrix]
    W_O: Matrix
    attention_gamma: list[float]
    attention_beta: list[float]
    W_1: Matrix
    b_1: list[float]
    W_2: Matrix
    b_2: list[float]
    ff_gamma: list[float]
    ff_beta: list[float]

    @classmethod
    def initialize(
        cls, d_model: int, num_heads: int, d_ff: int, rng: random.Random
    ) -> "EncoderLayer":
        """Draw fresh weights using the supplied random generator."""
        if min(d_model, num_heads, d_ff) <= 0 or d_model % num_heads:
            raise ValueError(
                "Dimensions must be positive and num_heads must divide d_model"
            )
        head_width = d_model // num_heads

        def matrix(rows: int, columns: int) -> Matrix:
            return [
                [rng.uniform(-0.1, 0.1) for _ in range(columns)] for _ in range(rows)
            ]

        def heads() -> list[Matrix]:
            return [matrix(d_model, head_width) for _ in range(num_heads)]

        return cls(
            W_Q_heads=heads(),
            W_K_heads=heads(),
            W_V_heads=heads(),
            W_O=matrix(d_model, d_model),
            attention_gamma=[1.0] * d_model,
            attention_beta=[0.0] * d_model,
            W_1=matrix(d_model, d_ff),
            b_1=[0.0] * d_ff,
            W_2=matrix(d_ff, d_model),
            b_2=[0.0] * d_model,
            ff_gamma=[1.0] * d_model,
            ff_beta=[0.0] * d_model,
        )

    def forward(self, X: Matrix) -> tuple[Matrix, list[Matrix]]:
        """Return contextual token vectors and one attention matrix per head."""
        attended, weights = multi_head_attention(
            X, self.W_Q_heads, self.W_K_heads, self.W_V_heads, self.W_O
        )
        hidden = layer_norm(
            residual_add(X, attended), self.attention_gamma, self.attention_beta
        )
        transformed = feed_forward(hidden, self.W_1, self.b_1, self.W_2, self.b_2)
        output = layer_norm(
            residual_add(hidden, transformed), self.ff_gamma, self.ff_beta
        )
        return output, weights


def encoder_stack(
    X: Matrix, layers: Sequence[EncoderLayer]
) -> tuple[Matrix, list[list[Matrix]]]:
    """Apply layers in order. Attention matrices are indexed by layer and head."""
    if not layers:
        raise ValueError("The encoder needs at least one layer")
    weights = []
    for layer in layers:
        X, layer_weights = layer.forward(X)
        weights.append(layer_weights)
    return X, weights
