import re
from collections import Counter
from collections.abc import Iterable, Mapping


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def build_vocab(texts: Iterable[str]) -> dict[str, int]:
    token_counts: Counter[str] = Counter()
    for text in texts:
        token_counts.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _count in token_counts.most_common():
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: Mapping[str, int]) -> list[int]:
    unknown_id = vocab["<UNK>"]
    return [vocab.get(token, unknown_id) for token in tokenize(text)]
