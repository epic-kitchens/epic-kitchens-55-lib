import numpy as np
import pytest
from numpy.testing import assert_array_equal

from epic_kitchens.scoring import compute_action_scores


class TestComputeActionScores:
    verb_scores = np.array(
        [[0.01, 0.99], [0.99, 0.1], [0.49, 0.51], [0.01, 0.99], [0.01, 0.99]]
    )
    noun_scores = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )

    def test_action_scores_without_prior(self):
        ((verbs, nouns), action_scores) = compute_action_scores(
            self.verb_scores, self.noun_scores, top_k=3
        )
        assert_array_equal(
            verbs,
            np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1]]),
            verbose=True,
        )
        assert_array_equal(
            nouns,
            np.array([[2, 1, 0], [2, 1, 0], [2, 2, 1], [0, 1, 2], [2, 1, 0]]),
            verbose=True,
        )

    def test_action_scores_with_priors(self):
        priors = np.array([[1e-8, 0, 1], [1, 0, 1e-16]])
        ((verbs, nouns), action_scores) = compute_action_scores(
            self.verb_scores, self.noun_scores, top_k=3, action_priors=priors
        )
        assert_array_equal(
            verbs,
            np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]),
            verbose=True,
        )
        assert_array_equal(
            nouns,
            np.array([[0, 2, 0], [2, 0, 0], [2, 0, 0], [0, 2, 0], [0, 2, 0]]),
            verbose=True,
        )
