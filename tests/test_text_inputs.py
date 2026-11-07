import random

import pytest

from name_that_llm.embedding import initialize_embeddings, lookup_embeddings
from name_that_llm.tokenizer import build_vocab, encode, tokenize


def test_tokenization_preserves_case_punctuation_and_unicode():
    assert tokenize("Hello, hello! café\nworld") == [
        "Hello",
        ",",
        "hello",
        "!",
        "café",
        "world",
    ]
    assert tokenize(" \n\t") == []


def test_unknown_tokens_share_embedding_without_changing_vocab():
    vocab = build_vocab(["Hello, world!", "Hello again."])
    original_vocab = vocab.copy()
    embeddings = initialize_embeddings(vocab)
    ids = encode("Hello spaceship telescope", vocab)

    assert ids == [vocab["Hello"], vocab["<UNK>"], vocab["<UNK>"]]
    assert vocab == original_vocab
    assert lookup_embeddings(ids, embeddings) == [
        embeddings[vocab["Hello"]],
        embeddings[vocab["<UNK>"]],
        embeddings[vocab["<UNK>"]],
    ]
    assert lookup_embeddings(encode("", vocab), embeddings) == []


def test_embedding_initialization_is_reproducible_and_rows_are_independent():
    vocab = build_vocab(["Hello world"])
    random_state = random.getstate()
    embeddings = initialize_embeddings(vocab, d=6, seed=42)

    assert random.getstate() == random_state
    assert embeddings == initialize_embeddings(vocab, d=6, seed=42)
    assert embeddings != initialize_embeddings(vocab, d=6, seed=43)
    assert len(embeddings) == len(vocab)
    assert all(len(row) == 6 for row in embeddings)
    assert embeddings[vocab["<PAD>"]] == [0.0] * 6

    other_row = embeddings[vocab["world"]].copy()
    embeddings[vocab["Hello"]][0] = 100.0
    assert embeddings[vocab["world"]] == other_row


def test_empty_training_texts_still_have_special_tokens():
    vocab = build_vocab([])
    assert vocab == {"<PAD>": 0, "<UNK>": 1}
    assert encode("unseen", vocab) == [vocab["<UNK>"]]


@pytest.mark.parametrize("d", [0, -1])
def test_embedding_dimension_must_be_positive(d):
    with pytest.raises(ValueError, match="d must be positive"):
        initialize_embeddings(build_vocab([]), d=d)


@pytest.mark.parametrize("token_id", [-1, 2])
def test_lookup_rejects_ids_outside_embedding_matrix(token_id):
    embeddings = initialize_embeddings(build_vocab([]))
    with pytest.raises(ValueError, match="outside the embedding matrix"):
        lookup_embeddings([token_id], embeddings)
