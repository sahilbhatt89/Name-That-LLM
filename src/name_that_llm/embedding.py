import math
import random
from collections.abc import Mapping, Sequence


def position_vector(position: int, d: int) -> list[float]:
    """Return the walkthrough's paired sine and cosine features for one position."""
    if d <= 0 or d % 2:
        raise ValueError("d must be a positive even number")
    vector = []
    for dimension in range(0, d, 2):
        angle = position / (10000 ** (dimension / d))
        vector.extend((math.sin(angle), math.cos(angle)))
    return vector


def initialize_embeddings(
    vocab: Mapping[str, int], d: int = 8, seed: int = 42
) -> list[list[float]]:
    if d <= 0:
        raise ValueError("d must be positive")

    rng = random.Random(seed)
    embeddings = [[rng.uniform(-0.1, 0.1) for _ in range(d)] for _ in range(len(vocab))]
    embeddings[vocab["<PAD>"]] = [0.0] * d
    return embeddings


def lookup_embeddings(
    token_ids: Sequence[int], embeddings: list[list[float]]
) -> list[list[float]]:
    vectors = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= len(embeddings):
            raise ValueError(f"Token ID {token_id} is outside the embedding matrix")
        vectors.append(embeddings[token_id])
    return vectors
