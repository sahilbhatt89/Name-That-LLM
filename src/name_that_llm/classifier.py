"""Paragraph classification from unpadded encoder outputs."""

import random
from collections.abc import Sequence
from dataclasses import dataclass


def mean_pool(token_vectors: Sequence[Sequence[float]]) -> list[float]:
    """Average each feature across the tokens of one paragraph."""
    if not token_vectors or not token_vectors[0]:
        raise ValueError("Pooling requires at least one nonempty token vector")
    width = len(token_vectors[0])
    if any(len(row) != width for row in token_vectors):
        raise ValueError("Token vectors must have the same width")
    return [sum(feature) / len(token_vectors) for feature in zip(*token_vectors)]


@dataclass
class ClassifierHead:
    """One linear projection per model label, in the supplied label order."""

    labels: tuple[str, ...]
    weights: list[list[float]]
    bias: list[float]

    def __post_init__(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("Supply at least one label and use distinct labels")
        if len(self.weights) != len(self.labels) or len(self.bias) != len(self.labels):
            raise ValueError("Each label requires a weight vector and a bias")
        width = len(self.weights[0])
        if not width or any(len(row) != width for row in self.weights):
            raise ValueError("Weight vectors must have the same positive width")

    @classmethod
    def initialize(
        cls, d_model: int, labels: Sequence[str], rng: random.Random
    ) -> "ClassifierHead":
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        weights = [[rng.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in labels]
        return cls(tuple(labels), weights, [0.0] * len(labels))

    def forward(self, paragraph: Sequence[float]) -> dict[str, float]:
        """Return the raw score (logit) for each model label."""
        return {
            label: sum(x * w for x, w in zip(paragraph, row, strict=True)) + bias
            for label, row, bias in zip(
                self.labels, self.weights, self.bias, strict=True
            )
        }
