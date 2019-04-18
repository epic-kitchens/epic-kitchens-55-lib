import numpy as np
import pandas as pd
import pytest

from epic_kitchens.metrics import precision_recall, topk_accuracy, compute_metrics


class TestPrecision:
    def test_all_tp(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([1, 2, 3])
        precision, _ = precision_recall(ranks, labels)

        assert np.all(precision == np.array([1, 1, 1]))

    def test_all_fp(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([3, 1, 2])
        precision, _ = precision_recall(ranks, labels)

        assert np.all(precision == np.array([0, 0, 0]))

    def test_no_fp_and_no_tp(self):
        ranks = np.array([[4, 2, 3], [4, 3, 1], [4, 1, 2]])
        labels = np.array([3, 1, 2])
        precision, _ = precision_recall(ranks, labels)

        assert np.all(precision == np.array([0, 0, 0]))

    def test_filter_existing_class(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([1, 2, 3])
        precision, recall = precision_recall(ranks, labels, classes=np.array([1]))

        assert np.all(precision == np.array([1]))
        assert np.all(recall == np.array([1]))

    def test_filter_nonexisting_class(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            precision, _ = precision_recall(ranks, labels, classes=np.array([4]))

    def test_throws_exception_if_labels_and_ranks_are_different_lengths(self):
        ranks = np.array([[2, 3, 1], [3, 1, 2]])
        labels = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            precision, _ = precision_recall(ranks, labels)


class TestRecall:
    def test_all_tp(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([1, 2, 3])

        _, recall = precision_recall(ranks, labels)

        assert np.all(recall == np.array([1, 1, 1]))

    def test_all_fn(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([2, 3, 1])

        _, recall = precision_recall(ranks, labels)

        assert np.all(recall == np.array([0, 0, 0]))


class TestAccuracyAtK:
    def test_at_1(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        labels = np.array([1, 2, 3])

        accuracy = topk_accuracy(ranks, labels, ks=1)

        assert accuracy == 1

    def test_at_2(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]])
        labels = np.array([1, 3, 2, 1])

        accuracy = topk_accuracy(ranks, labels, ks=2)

        assert accuracy == 0.75

    def test_at_3(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]])
        labels = np.array([1, 3, 2, 1])

        accuracy = topk_accuracy(ranks, labels, ks=3)

        assert accuracy == 1

    def test_at_1_and_3(self):
        ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]])
        labels = np.array([1, 3, 2, 1])

        accuracy = topk_accuracy(ranks, labels, ks=(1, 3, 5))

        assert np.all(accuracy == np.array([0.5, 1, 1]))


class TestComputeMetrics:
    groundtruth_df = pd.DataFrame(
        {"verb_class": [0, 0, 1, 1], "noun_class": [0, 0, 1, 2]}
    )
    scores = {
        # Correct: 3
        # Incorrect: 1
        "verb": np.array(
            [
                [0.1, 0.9],  # pred 1, true 0
                [0.6, 0.4],  # pred 0, true 0
                [0.1, 0.9],  # pred 1, true 1
                [0.9, 0.1],  # pred 0, true 1
            ]
        ),
        # Correct: 2
        # Incorrect: 2
        "noun": np.array(
            [
                [0.1, 0.2, 0.7],  # pred 2, true 0
                [0.2, 0.7, 0.1],  # pred 1, true 0
                [0.3, 0.4, 0.3],  # pred 1, true 1
                [0.1, 0.1, 0.8],  # pred 2, true 2
            ]
        ),
    }
    many_shot_verbs = [0, 1]
    many_shot_nouns = [0, 1]
    many_shot_actions = [(0, 0), (1, 1)]

    metrics = compute_metrics(
        groundtruth_df,
        scores,
        many_shot_verbs=many_shot_verbs,
        many_shot_nouns=many_shot_nouns,
        many_shot_actions=many_shot_actions,
    )

    def test_verb_accuracy_at_1(self):
        assert self.metrics["accuracy"]["verb"][0] == (2 / 4)

    def test_verb_accuracy_at_5(self):
        assert self.metrics["accuracy"]["verb"][1] == 1

    def test_noun_accuracy_at_1(self):
        assert self.metrics["accuracy"]["noun"][0] == (2 / 4)

    def test_noun_accuracy_at_5(self):
        assert self.metrics["accuracy"]["noun"][1] == 1

    def test_action_accuracy_at_1(self):
        assert self.metrics["accuracy"]["action"][0] == (1 / 4)

    def test_action_accuracy_at_5(self):
        # the first entry has the correct entry at place 6
        assert self.metrics["accuracy"]["action"][1] == (3 / 4)

    def test_verb_precision(self):
        assert self.metrics["precision"]["verb"] == ((1 / 2) + (1 / 2)) / 2

    def test_verb_recall(self):
        assert self.metrics["recall"]["verb"] == ((1 / 2) + (1 / 2)) / 2

    def test_noun_precision(self):
        assert self.metrics["precision"]["noun"] == (0 + (1 / 2)) / 2

    def test_noun_recall(self):
        assert self.metrics["recall"]["noun"] == (0 + (1 / 1)) / 2

    def test_action_recall(self):
        assert self.metrics["recall"]["action"] == ((0 / 2) + (1 / 1)) / 2

    def test_action_precision(self):
        assert self.metrics["precision"]["action"] == (0 + (1 / 1)) / 2
