# Development


# Version 1.7.0

## Features

* Add `epic_kitchens.metrics` and `epic_kitchens.scoring` containing method for
  computing metrics and manipulating model scores. Of particular note is
  `epic_kitchens.metrics.compute_metrics` which computes top-{1,5} accuracy,
  many-shot precision and recall for the verb, noun and action prediction tasks.

## Dependencies

* We now depend on `scikit-learn` for computing metrics.

# Version 1.6.3

## Dependencies
* Depend on `moviepy` >= 1.0.0 due to `imageio` breaking `ffmpeg` download
  capabilities.


# Version 1.6.2

## Bug fix

* `epic_kitchens.gulp.visualisation` visualisers returned `ImageSequenceClip`
  instead of IPython HTML display element, which caused the videos not to show
  up in Jupyter, these methods now return an `IPython.display.HTML` element
* Fix bug in `epic_kitchens.gulp.visualisation.combine_flow_uv_frames` where
  `hstack_frames` wasn't called with `width_axis` correctly causing a
  RuntimeError

# Version 1.6.1

## Bug fix

* Add sphinx 1.8.2 to requirements.txt to get RTFD to build the docs

# Version 1.6.0

## Features

* Add `epic_kitchens.meta` package containing helpers for downloading and reading
  label class definitions, and training and test set annotations.
* Add `epic_kitchens.data.visualisation` package containing tools that allow you
  to view arbitrary action segments RGB or Flow contained in a `EpicVideoDataset`
  using `moviepy`. This is useful when sifting through results per instance.

## Changes
* Improve documentation and bring it all in line with google doc string
  standards

# Version 1.5.0

## Features

* Add `__getitem__` method to `EpicVideoDataset` to enable segments to be
  retrieved by ID, rather than having to obtain the `video_segments` and filter
  by ID.
* Make `idx` arg optional in `VideoDataset.load_frames`, by default we now load
  all the frames for the segment, this behaviour is equivalent to
  `VideoDataset.load_frames(segment, range(0, segment.num_frames)`
* Support string paths to constructor of `EpicVideoDataset` and `EpicVideoFlowDataset`
* Support test datasets in `EpicVideoDataset` and `EpicVideoFlowDataset`: Simply set
  `class_type` as `None` and the video segment's `label` property will be `None`.
* Bugfix: Gulped flow frames were read in a random order (due to the results of
  `glob.glob` not being sorted lexicographically), so if you have used a version
  prior to 1.5.0 you should discard your gulped flow, and regulp using the fixed
  adapter - our sincere apologies for the inconvenience caused.

**WARNING: People who have gulped flow**:
If you have used any previous version of the library to gulp flow,
then you should discard that flow and regulp the flow due to the previous
version of the adapter reading frames in an unsorted manner resulting in
flow segments within the gulp file being randomly shuffled.


# Version 1.4.0

## Features

* Add `sample_transform` kwarg to `VideoDataset`, this allows you to
  transform each frame or optical flow stack by providing a function that takes
  in a list of PIL images and produces a list of PIL images
* Add `segment_filter` kwarg to `VideoDataset`, this allows you to selectively
  filter action segments from a video by providing a function that takes a
  `VideoSegment` and makes a decision on whether to include the segment in the
  dataset or not, thus allowing you to filter to include or exclude specific
  classes etc.

# Version 1.3.0

## Features

* Both `epic_kitchens.preprocessing.split_segments` and
  `epic_kitchens.gulp` now support reading CSV labels as well as a pickled
  labels.
* Support un-narrated action segment splitting in
  `epic_kitchens.preprocessing.split_segments` to enable splitting using the
  test timestamp CSVs.
* Add `--unlabelled` option to `epic_kitchens.gulp` to enable gulping of test
  set that doesn't have label data.


# Version 1.2.1

## Bug fix

* Fix crash due to passing `Path` object to `os.path.lexists` in
  `epic_kitchens.preprocessing.split_segments` on Python 3.5 (3.6+ supports
  this)


# Version 1.2.0

## Features

* Expose `epic_kitchens.preprocessing.split_segments` as an entrypoint
* Add docs for `epic_kitchens.preprocessing.*`


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
