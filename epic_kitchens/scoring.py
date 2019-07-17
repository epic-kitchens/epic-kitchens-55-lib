import numpy as np
from typing import Tuple, Union, List, Dict, Optional


def compute_action_scores(
    verb_scores: np.ndarray,
    noun_scores: np.ndarray,
    top_k: int = 100,
    action_priors: Optional[np.ndarray] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Given the predicted verb and noun scores, compute action scores by
    :math:`p(A = (v, n)) = p(V = v)p(N = n)`.

    Args:
        verb_scores: 2D array of verb scores ``(n_instances, n_verbs)``.
        noun_scores: 2D array of noun scores ``(n_instances, n_nouns)``.
        top_k: Number of highest scored actions to compute.
        action_priors: 2D array of action priors ``(n_verbs, n_nouns)``. These don't
            have to sum to one and as such you can provide the training counts of
            :math:`(v, n)` occurrences (to minimize numerical stability issues).

    Returns:
        A tuple ``((verbs, noun), action_scores)`` where ``verbs`` and ``nouns`` are
        2D arrays of shape ``(n_instances, top_k)`` containing the classes
        constituting the top-k action scores.
        ``action_scores`` is a 2D array of shape ``(n_instances, top_k)`` where
        ``action_scores[i, j]`` corresponds to the score for the action class
        ``(verbs[i, j], nouns[i, j])``.
        The scores are sorted in descending order, i.e. ``action_scores[i, j] >=
        action_scores[i, j + 1]``.
    """
    top_verbs, top_verb_scores = top_scores(verb_scores, top_k=top_k)
    top_nouns, top_noun_scores = top_scores(noun_scores, top_k=top_k)
    top_verb_probs = softmax(top_verb_scores)
    top_noun_probs = softmax(top_noun_scores)
    # shape: (n_instances, n_verbs, n_nouns)
    action_probs_matrix = (
        top_verb_probs[:, :, np.newaxis] * top_noun_probs[:, np.newaxis, :]
    )
    instance_count = action_probs_matrix.shape[0]
    segments = np.arange(0, instance_count).reshape(-1, 1)
    if action_priors is not None:
        expected_action_prior_shape = (verb_scores.shape[-1], noun_scores.shape[-1])
        if action_priors.shape != expected_action_prior_shape:
            raise ValueError(
                "Expected action_priors to have the shape {}, but was {}".format(
                    expected_action_prior_shape, action_priors.shape
                )
            )
        action_probs_matrix *= action_priors[
            top_verbs[:, :, np.newaxis], top_nouns[:, np.newaxis, :]
        ]
    # shape: (n_instances, n_verbs*n_nouns)
    action_ranks = action_probs_matrix.reshape(instance_count, -1).argsort(axis=-1)[
        :, ::-1
    ]
    verb_ranks_idx, noun_ranks_idx = np.unravel_index(
        action_ranks[:, :top_k], shape=action_probs_matrix.shape[1:]
    )

    return (
        (top_verbs[segments, verb_ranks_idx], top_nouns[segments, noun_ranks_idx]),
        action_probs_matrix.reshape(instance_count, -1)[
            segments, action_ranks[:, :top_k]
        ],
    )


def scores_to_ranks(scores: Union[np.ndarray, List[Dict[int, float]]]) -> np.ndarray:
    """Convert scores to ranks

    Args:
        scores: A 2D array of scores of shape ``(n_instances, n_classes)`` or a list of
            dictionaries, where each dictionary represents the sparse scores for a task.
            The *key: value* pairs of the dictionary represent the *class: score*
            mapping.

    Returns:
        A 2D array of ranks ``(n_instances, n_classes)``. Each row contains the
        ranked classes in descending order, i.e. ``ranks[0, i]`` is ranked higher than
        ``ranks[0, i+1]``. The index is the rank, and the element the class at that
        rank.
    """
    if isinstance(scores, np.ndarray):
        return _scores_array_to_ranks(scores)
    elif isinstance(scores, list):
        return _scores_dict_to_ranks(scores)
    raise ValueError("Cannot compute ranks for type {}".format(type(scores)))


def scores_dict_to_ranks(scores_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert a dictionary of task to scores to a dictionary of task to ranks

    Args:
        scores_dict: Dictionary of task to scores array ``(n_instances, n_classes)``

    Returns:
        Dictionary of task to ranks array ``(n_instances, n_classes)``
    """
    return {key: scores_to_ranks(scores) for key, scores in scores_dict.items()}


def top_scores(scores: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``top_k`` class indices and scores in descending order.

    Args:
        scores: array of scores, either 1D ``(n_classes,)`` or 2D
            ``(n_instances, n_classes)``.
        top_k: The number of top scored classes to return

    Returns:
        A tuple containing two arrays, ``(ranked_classes, scores)`` where
        ranked_classes contains the classes in descending order of score,
        and ``scores`` contains the corresponding score for each class, i.e.
        ``ranked_classes[..., i]`` has score ``scores[..., i]``.

    Examples:
        >>> top_scores(np.array([0.2, 0.6, 0.1, 0.04, 0.06]), top_k=3)
        (array([1, 0, 2]), array([0.6, 0.2, 0.1]))
    """
    if scores.ndim == 1:
        top_k_idx = scores.argsort()[::-1][:top_k]
        return top_k_idx, scores[top_k_idx]

    top_k_scores_idx = np.argsort(scores)[..., ::-1][:, :top_k]
    top_k_scores = scores[np.arange(0, len(scores)).reshape(-1, 1), top_k_scores_idx]
    return top_k_scores_idx, top_k_scores


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute the softmax of the 1D or 2D array ``x``.

    Args:
        x: a 1D or 2D array. If 1D, then it is assumed that it is a single class score
           vector. Otherwise, if ``x`` is 2D, then each row is assumed to be a class
           score vector.


    Examples:
        >>> res = softmax(np.array([0, 200, 10]))
        >>> np.sum(res)
        1.0
        >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
        True
        >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
        >>> np.argsort(res, axis=1)
        array([[0, 2, 1],
               [0, 1, 2],
               [1, 2, 0]])
        >>> np.sum(res, axis=1)
        array([1., 1., 1.])
        >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
        >>> np.sum(res, axis=1)
        array([1., 1.])
    """
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def _scores_array_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert an array of scores to an array of ranks

    Args:
        scores: with shape: (n_instances, n_classes)

    Returns:
        The rank vector whose elements are classes and index is the rank
        with shape (n_instances, n_classes)

    Examples:
        >>> _scores_array_to_ranks(np.array([[0.1, 0.15, 0.25,  0.3, 0.5], \
                                             [0.5, 0.3, 0.25,  0.15, 0.1], \
                                             [0.2, 0.4,  0.1,  0.25, 0.05]]))
        array([[4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4],
               [1, 3, 0, 2, 4]])
    """
    assert scores.ndim == 2, (
        "Expected scores to be 2D [n_instances, n_classes], "
        "but was {}D".format(scores.ndim)
    )
    return scores.argsort(axis=-1)[..., ::-1]


def _scores_dict_to_ranks(scores: List[Dict[int, float]]) -> np.ndarray:
    """
    Compute ranking from class to score dictionary

    Examples:
        >>> _scores_dict_to_ranks([{0: 0.15, 1: 0.75, 2: 0.1},\
                                   {0: 0.85, 1: 0.10, 2: 0.05}])
        array([[1, 0, 2],
               [0, 1, 2]])
    """
    ranks = []
    for score in scores:
        class_ids = np.array(list(score.keys()))
        score_array = np.array([score[class_id] for class_id in class_ids])
        ranks.append(class_ids[np.argsort(score_array)[::-1]])
    return np.array(ranks)
