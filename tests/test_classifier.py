import random

import pytest

from name_that_llm.classifier import ClassifierHead, mean_pool


def test_scores_each_label_using_the_feature_means_and_its_own_bias():
    tokens = [[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]]
    head = ClassifierHead(("z", "a"), [[2.0, -1.0], [-1.0, 3.0]], [0.5, 2.0])

    paragraph = mean_pool(tokens)
    assert paragraph == [3.0, 2.0]
    assert head.forward(paragraph) == {"z": 4.5, "a": 5.0}
    assert head.forward(mean_pool([paragraph])) == head.forward(paragraph)


@pytest.mark.parametrize("tokens", [[], [[]], [[1.0], [1.0, 2.0]]])
def test_invalid_token_vectors(tokens):
    with pytest.raises(ValueError):
        mean_pool(tokens)


def test_initialized_head_can_be_reused_across_sequence_lengths():
    labels = ["model-z", "model-a"]
    head = ClassifierHead.initialize(4, labels, random.Random(7))
    assert head == ClassifierHead.initialize(4, labels, random.Random(7))
    assert head.weights[0] != head.weights[1]
    assert head.bias == [0.0, 0.0]
    for length in [1, 3]:
        scores = head.forward(mean_pool([[1.0] * 4] * length))
        assert list(scores) == labels
        assert list(scores.values()) == [sum(row) for row in head.weights]
    with pytest.raises(ValueError):
        head.forward([0.0] * 3)


@pytest.mark.parametrize("width, labels", [(0, ["a"]), (2, []), (2, ["a", "a"])])
def test_invalid_classifier_configuration(width, labels):
    with pytest.raises(ValueError):
        ClassifierHead.initialize(width, labels, random.Random(7))
