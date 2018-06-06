# Version 1.1.1

## Bug fix

* `setup.py` used to import `epic_kitchens` which would fail if all dependencies
  weren't already satisfied, which is the case in fresh virtual environments
  causing the installation to fail, now the metadata is kept in a separate file
  read in to `setup.py` to avoid this issue.


# Version 1.1.0

## Bug fix

* `epic_kitchens.gulp` in 1.0.0rc0 didn't read in the pickled dataframe before
  constructing the `EpicDatasetAdapter` causing an exception to be thrown
  therefore rendering the script useless.

## Features

* Change CLI interface of `epic_kitchens.preprocessing.split_segments` to match
  that of `epic_kitchens.gulp` in terms of argument ordering and whether
  arguments are mandatory or not.
