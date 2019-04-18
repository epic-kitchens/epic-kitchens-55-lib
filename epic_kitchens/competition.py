"""
Submissions are in the format below

..highlight:: json
    {
      "version": "0.1",
      "challenge": "action_recognition",
      "results": {
        "1924": {
          "verb": {
            "0": 1.223,
            "1": 4.278,
            ...
            "124": 0.023
          },
          "noun": {
            "0": 0.804,
            "1": 1.870,
            ...
            "351": 0.023
          }
        },
        "1925": { ... },
        ...
      }
    }

where the key in the ``"results"`` dictionary is an action UID, and the corresponding value
is a dictionary containing at least ``"verb"`` and ``"noun"`` score dictionaries.

This module contains tools for generating such a JSON file from verb, noun and action scores.
"""
from typing import Dict, TextIO, Optional, Union
import numpy as np
import json


def create_submission_json(
    uids: np.ndarray,
    scores: Dict[str, np.ndarray],
    f: Optional[TextIO] = None,
    challenge="action_recognition",
    version="0.1",
) -> Union[str, None]:
    """
    Args:
        uids:  Array of action UIDs that map 1-to-1 to the entries in the verbs/nouns arrays in
            the ``scores`` dictionary.
        scores: Dictionary containing scores for both verbs and nouns,
        f: File handle to write the JSON to, if this is passed, a JSON string is
            not returned.
        challenge: The name of the challenge being submitted to: must be one of
            ``action_recognition``, ``action_anticipation``.
        version: Version string of the results JSON schema.

    Returns:
        JSON string if ``f`` is not set.
    """
    scores_json_dict = _create_submission_dict(uids, scores, challenge, version)

    if f is not None:
        json.dump(scores_json_dict, f)
        return None
    else:
        return json.dumps(scores_json_dict)


def _scores_to_dict(scores: np.ndarray) -> Dict[str, float]:
    return {str(id_): score for id_, score in zip(range(len(scores)), scores)}


def _create_submission_dict(
    uids: np.ndarray,
    scores: Dict[str, np.ndarray],
    challenge: str = "action_recognition",
    version: str = "0.1",
):
    if uids.ndim != 1:
        raise ValueError("UIDs should be a 1D array but was {}D".format(uids.ndim))
    for task in ["verb", "noun"]:
        if task not in scores.keys():
            raise ValueError(
                "{} scores must be provided in the 'scores' dict".format(task)
            )
        if len(scores[task]) != len(uids):
            raise ValueError("The UIDs array should be the same length as the scores")
        if scores[task].ndim != 2:
            raise ValueError(
                "{} scores must be a 2D ndarray, but was {}D".format(
                    task, scores[task].ndim
                )
            )
    results_dict = dict()
    for uid, verb_scores, noun_scores in zip(uids, scores["verb"], scores["noun"]):
        results_dict[uid] = {
            "verb": _scores_to_dict(verb_scores),
            "noun": _scores_to_dict(noun_scores),
        }
    scores_json_dict = {
        "version": version,
        "challenge": challenge,
        "results": results_dict,
    }
    return scores_json_dict
