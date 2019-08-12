# Submission

To submit your results to the leaderboard you must construct a submission zip file containing two
files:

- `seen.json` - Model inference on the seen kitchens test set (S1)
- `unseen.json` - Model inference on the unseen kitchens test set (S2)

Both of these files follow the same format detailed below:


## JSON Submission Format

The JSON submission format is composed of a single JSON object containing entries for every action
in the test set. Specifically, the JSON file
should contain:

 - a `'version'` property, set to `'0.1'` (the only supported version so far);
 - a `'challenge'` property, which can assume the following values, depending on the challenge: `['action_recognition', 'action_anticipation']`;
 - a `'results'` object containing entries for every action in the test set (e.g. `'1924'` is the first action ID in the *seen* test set).

Each action segment entry is a nested object composed of two entries: `'verb'`,  specifying the class score
for every verb class and
the other, `'noun'` specifying the score for every noun class. Action scores are automatically computed
by applying softmax to the verb and noun scores and computing the probability of each possible action.


```json
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
```

If you wish to compute your own action scores, you can augment each segment submission with
exactly 100 action scores with the key `'action'`

```json
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
      },
      "action": {
        "0,1": 1.083,
        ...
        "124,351": 0.002
      }
    },
    "1925": { ... },
    ...
  }
}
```

The keys of the `action` object are of the form `<verb_class>,<noun_class>`.

You can provide scores in any float format that numpy is capable of reading (i.e. you do not need to stick
to 3 decimal places).

If you fail to provide your own action scores we will compute them by

1. Obtaining softmax probabilites from your *verb* and *noun* scores
2. Find the top 100 action probabilities where `p(a = (v, n)) = p(v) * p(n)`


## Submission archive

To upload your results to CodaLab you have to zip both files into a flat zip archive (they can't be inside
a folder within the archive).

You can create a flat archive using the command providing the JSON files are in your current directory.

```
$ zip -j my-submission.zip seen.json unseen.json
```
