# EPIC-KITCHENS python library

[![CircleCI status badge](https://img.shields.io/circleci/project/github/epic-kitchens/epic-lib/master.svg)](https://circleci.com/gh/epic-kitchens/epic-lib)
[![codecov](https://codecov.io/gh/epic-kitchens/epic-lib/branch/master/graph/badge.svg)](https://codecov.io/gh/epic-kitchens/epic-lib)
[![Documentation Status](https://readthedocs.org/projects/epic-kitchens/badge/?version=latest)](http://epic-kitchens.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/epic-kitchens.svg)](https://pypi.org/project/epic-kitchens/#description)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/epic-kitchens.svg)](https://pypi.org/project/epic-kitchens/)

> A library for easily integrating the EPIC-KITCHENS egocentric dataset in your
> experiments

## Install

```console
$ pip install epic-kitchens
```

## Details

* Segmentation scripts for splitting raw video frames/flow into action segments
* [GulpIO adapter](https://github.com/TwentyBN/GulpIO#loading-data) for ingesting and reading the
  dataset. This works particularly well for PyTorch and Tensorflow models.
* Dataset classes for loading and augmenting data
* Utilities for converting between frame indices for RGB and flow.

See the [CHANGELOG](CHANGELOG.md) for release notes

Check out the documentation on [Read the
docs](https://epic-kitchens.readthedocs.io/en/latest/index.html)
