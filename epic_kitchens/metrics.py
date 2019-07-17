from abc import ABC
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Any,
    Optional,
    Sized,
    Iterable,
    Container,
    Generic,
    TypeVar,
)
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import logging

from epic_kitchens.meta import (
    action_id_from_verb_noun,
    action_tuples_to_ids,
    ActionClass,
)
from epic_kitchens.scoring import compute_action_scores, scores_dict_to_ranks
from . import meta


T = TypeVar("T")


LOG = logging.getLogger(__name__)
Metric = str
Task = str
MetricsDict = Dict[Metric, Any]


def compute_metrics(
    groundtruth_df: pd.DataFrame,
    scores: Dict[str, Union[np.ndarray, Dict[int, float]]],
    many_shot_verbs: Optional[np.ndarray] = None,
    many_shot_nouns: Optional[np.ndarray] = None,
    many_shot_actions: Optional[np.ndarray] = None,
    action_priors: Optional[np.ndarray] = None,
) -> MetricsDict:
    """Compute the EPIC action recognition evaluation metrics from ``scores`` given
    ground truth labels in ``groundtruth_df``.

    Args:
        groundtruth_df:
            DataFrame containing ``verb_class``: :py:class:`int`,
            ``noun_class``: :py:class:`int`. This function will add an
            ``action_class`` column containing the action ID obtained from
            :py:func:`epic_kitchens.meta.action_id_from_verb_noun`.
        scores:
            Dictionary containing: ``'verb'``, ``'noun'`` and (optionally) ``'action'`` entries.
            ``'verb'`` and ``'noun'`` should map to a 2D :py:class:`np.ndarray` of shape
            ``(n_instances, n_classes)`` where each element is the predicted score of
            that class. ``'action'`` should map to a dictionary of action keys to
            scores. The order of the scores array should be the same as the order in
            ``groundtruth_df``.
        many_shot_verbs:
            The set of verb classes that are considered many shot. If not provided
            they are loaded from :py:func:`epic_kitchens.meta.many_shot_verbs`
        many_shot_nouns:
            The set of noun classes that are considered many shot. If not provided
            they are loaded from :py:func:`epic_kitchens.meta.many_shot_nouns`
        many_shot_actions:
            The set of action classes that are considered many shot. If not provided
            they are loaded from :py:func:`epic_kitchens.meta.many_shot_actions`
        action_priors:
            A ``(n_verbs, n_nouns)`` shaped array containing the action prior used to
            weight action predictions.

    Returns:
        A dictionary containing all metrics with the following structure::

            accuracy:
                verb: list[float, length 2]
                noun: list[float, length 2]
                action: list[float, length 2]
            precision:
                verb: float
                noun: float
                action: float
            recall:
                verb: float
                noun: float
                action: float

        Accuracy lists contain the top-k metrics like so ``[top_1, top_5]``,
        the precision and recall metrics are macro averaged and computed over the
        many-shot classes.

    Raises:
        ValueError
            If the shapes of the ``scores`` arrays are not correct, or the lengths of
            ``groundtruth_df`` and the ``scores`` arrays are not equal, or if
            ``grountruth_df`` doesn't have the specified columns.
    """
    if many_shot_verbs is None:
        many_shot_verbs = np.array(list(meta.many_shot_verbs()))
    if many_shot_nouns is None:
        many_shot_nouns = np.array(list(meta.many_shot_nouns()))
    if many_shot_actions is None:
        many_shot_action_ids = np.array(action_tuples_to_ids(meta.many_shot_actions()))
    else:
        many_shot_action_ids = np.array(action_tuples_to_ids(many_shot_actions))

    for entry in "verb", "noun":
        class_col = entry + "_class"
        if class_col not in groundtruth_df.columns:
            raise ValueError("Expected '{}' column in groundtruth_df".format(class_col))

    groundtruth_df["action_class"] = action_id_from_verb_noun(
        groundtruth_df["verb_class"], groundtruth_df["noun_class"]
    )

    if "action" not in scores:
        (clip_verbs, clip_nouns), clip_scores = compute_action_scores(
            scores["verb"], scores["noun"], top_k=100, action_priors=action_priors
        )
        scores["action"] = [
            {
                action_id_from_verb_noun(verb, noun): score
                for verb, noun, score in zip(verbs, nouns, scores)
            }
            for verbs, nouns, scores in zip(clip_verbs, clip_nouns, clip_scores)
        ]

    ranks = scores_dict_to_ranks(scores)
    top_k = (1, 5)

    accuracies = compute_class_aware_metrics(groundtruth_df, ranks, top_k)
    precision_recall_metrics = compute_class_agnostic_metrics(
        groundtruth_df, ranks, many_shot_verbs, many_shot_nouns, many_shot_action_ids
    )

    return {
        "accuracy": {
            "verb": accuracies["verb"],
            "noun": accuracies["noun"],
            "action": accuracies["action"],
        },
        **precision_recall_metrics,
    }


def compute_class_aware_metrics(
    groundtruth_df: pd.DataFrame,
    ranks: Dict[str, np.ndarray],
    top_k: Union[int, Tuple[int, ...]] = (1, 5),
) -> Dict[str, Union[float, Union[float, List[float]]]]:
    """Compute class aware metrics (accuracy @ 1/5) from ranks.

    Args:
        groundtruth_df:
            DataFrame containing ``'verb_class'``: :py:class:`int`, ``'noun_class'``:
            :py:class:`int` and ``'action_class'``: :py:class:`int` columns.
        ranks:
            Dictionary containing three entries: ``'verb'``, ``'noun'`` and
            ``'action'``. Entries should map to a 2D :py:class:`np.ndarray`
            of shape ``(n_instances, n_classes)`` where the index is the predicted
            rank of the class at that index.
        top_k:
            The set of k values to compute top-k accuracy for.

    Returns:
        Dictionary with the structure::

            verb: list[float, length = len(top_k)]
            noun: list[float, length = len(top_k)]
            action: list[float, length = len(top_k)]
    """
    verb_accuracies = topk_accuracy(
        ranks["verb"], groundtruth_df["verb_class"].values, ks=top_k
    )
    noun_accuracies = topk_accuracy(
        ranks["noun"], groundtruth_df["noun_class"].values, ks=top_k
    )
    action_accuracies = topk_accuracy(
        ranks["action"], groundtruth_df["action_class"].values
    )
    return {
        "verb": verb_accuracies,
        "noun": noun_accuracies,
        "action": action_accuracies,
    }


def compute_class_agnostic_metrics(
    groundtruth_df: pd.DataFrame,
    ranks: Dict[str, np.ndarray],
    many_shot_verbs: Optional[np.ndarray] = None,
    many_shot_nouns: Optional[np.ndarray] = None,
    many_shot_actions: Optional[np.ndarray] = None,
) -> Dict[Metric, Dict[Task, Union[np.float, Dict[str, np.float]]]]:
    """
    Compute class agnostic metrics (many-shot precision and recall) from ranks.

    Args:
        groundtruth_df:
            DataFrame containing ``'verb_class'``: :py:class:`int`,
            ``'noun_class'``: :py:class:`int` and ``'action_class'``: :py:class:`int`
            columns.
        ranks:
            Dictionary containing three entries: ``'verb'``, ``'noun'`` and
            ``'action'``. Entries should map to a 2D :py:class:`np.ndarray`
            of shape ``(n_instances, n_classes)`` where the index is the predicted
            rank of the class at that index.
        many_shot_verbs:
            The set of verb classes that are considered many shot. If not provided
            they are loaded from :py:func:`epic_kitchens.meta.many_shot_verbs`
        many_shot_nouns:
            The set of noun classes that are considered many shot. If not provided
            they are loaded from :py:func:`epic_kitchens.meta.many_shot_nouns`
        many_shot_actions:
            The set of action classes that are considered many shot. If not provided
            they are loaded from :py:func:`epic_kitchens.meta.many_shot_actions`

    Returns:
        Dictionary with the structure::

            precision:
                verb: float
                noun: float
                action: float
                verb_per_class: dict[str:float, length = n_verbs]
            recall:
                verb: float
                noun: float
                action: float
                verb_per_class: dict[str:float, length = n_verbs]

        The ``'verb'``, ``'noun'``, and ``'action'`` entries of the metric dictionaries
        are the macro-averaged mean precision/recall over the set of many shot classes,
        whereas the 'verb_per_class' entry is a breakdown for each verb_class in the
        format of a dictionary mapping stringified verb class to that class'
        precision/recall.
    """

    if many_shot_verbs is None:
        many_shot_verbs = np.array(list(meta.many_shot_verbs()))

    if many_shot_nouns is None:
        many_shot_nouns = np.array(list(meta.many_shot_nouns()))

    if many_shot_actions is None:
        many_shot_actions = np.array(action_tuples_to_ids(meta.many_shot_actions()))

    many_shot_verbs = _exclude_non_existent_classes(
        many_shot_verbs, groundtruth_df["verb_class"]
    )
    many_shot_nouns = _exclude_non_existent_classes(
        many_shot_nouns, groundtruth_df["noun_class"]
    )
    many_shot_actions = _exclude_non_existent_classes(
        many_shot_actions, groundtruth_df["action_class"]
    )

    verb_precision, verb_recall = precision_recall(
        ranks["verb"], groundtruth_df.verb_class, classes=many_shot_verbs
    )
    noun_precision, noun_recall = precision_recall(
        ranks["noun"], groundtruth_df.noun_class, classes=many_shot_nouns
    )
    LOG.debug(
        "{} many shot actions before intersecting with actions present in test".format(
            len(many_shot_actions)
        )
    )
    LOG.info(
        "{} many shot actions after intersecting with actions present in test".format(
            len(many_shot_actions)
        )
    )
    action_precision, action_recall = precision_recall(
        ranks["action"], groundtruth_df["action_class"], classes=many_shot_actions
    )
    precision_many_shot_verbs = {
        str(verb): score for verb, score in zip(many_shot_verbs, verb_precision)
    }
    recall_many_shot_verbs = {
        str(verb): score for verb, score in zip(many_shot_verbs, verb_recall)
    }

    return {
        "precision": {
            "action": action_precision.mean(),
            "verb": verb_precision.mean(),
            "noun": noun_precision.mean(),
            "verb_per_class": precision_many_shot_verbs,
        },
        "recall": {
            "action": action_recall.mean(),
            "verb": verb_recall.mean(),
            "noun": noun_recall.mean(),
            "verb_per_class": recall_many_shot_verbs,
        },
    }


def topk_accuracy(
    rankings: np.ndarray, labels: np.ndarray, ks: Union[Tuple[int, ...], int] = (1, 5)
) -> Union[float, List[float]]:
    """Computes top-k accuracies for different values of k from rankings.

    Args:
        rankings: 2D rankings array ``(n_instances, n_classes)``
        labels: 1D correct labels array ``(n_instances,)``
        ks: The k values in top-k

    Returns:
        Top-k accuracy for each ``k`` in ``ks``. If only one ``k`` is provided,
        then only a single float is returned.

    Raises:
        ValueError
             If the dimensionality of the ``rankings`` or ``labels`` is incorrect, or
             if the length of ``rankings`` and ``labels`` aren't equal.
    """
    if isinstance(ks, int):
        ks = (ks,)
    _check_label_predictions_preconditions(rankings, labels)

    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    accuracies = [tp[:, :k].max(1).mean() for k in ks]
    if len(accuracies) == 1:
        return accuracies[0]
    else:
        return accuracies


def precision_recall(
    rankings: np.ndarray, labels: np.ndarray, classes: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes precision and recall from rankings.

    Args:
        rankings: 2D array of shape ``(n_instances, n_classes)``
        labels: 1D array of shape = ``(n_instances,)``
        classes: Iterable of classes to compute the metrics over.

    Returns:
        Tuple of ``(precision, recall)`` where
        ``precision`` is a 1D array of shape ``(len(classes),)``, and
        ``recall`` is a 1D array of shape ``(len(classes),)``

    Raises:
        ValueError
             If the dimensionality of the ``rankings`` or ``labels`` is incorrect, or if
             the length of the ``rankings`` and ``labels`` are not equal, or if the set
             of the provided ``classes`` is not a subset of the classes present in
             ``labels``.
    """
    _check_label_predictions_preconditions(rankings, labels)
    y_pred = rankings[:, 0]
    if classes is None:
        classes = np.unique(labels)
    else:
        provided_class_presence = np.in1d(classes, np.unique(labels))
        if not np.all(provided_class_presence):
            raise ValueError(
                "Classes {} are not in labels".format(classes[provided_class_presence])
            )
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, y_pred, labels=classes, average=None, warn_for=tuple("recall")
    )
    return precision, recall


def _exclude_non_existent_classes(classes: np.ndarray, labels: pd.Series) -> np.ndarray:
    return np.intersect1d(classes, labels.unique())


def _check_label_predictions_preconditions(
    rankings: np.ndarray, labels: np.ndarray
) -> None:
    if not rankings.ndim == 2:
        raise ValueError(
            "Rankings should be a 2D matrix, but was {}D".format(rankings.ndim)
        )
    if not labels.ndim == 1:
        raise ValueError("Labels should be a 1D vector but was {}D".format(labels.ndim))
    if not labels.shape[0] == rankings.shape[0]:
        raise ValueError(
            "Number of labels provided does not match number of predictions"
        )
