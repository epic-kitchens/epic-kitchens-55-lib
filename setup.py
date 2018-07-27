#!/usr/bin/env python

import os
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, 'epic_kitchens', '__version__.py'), 'r') as f:
    exec(f.read(), about)


setup(
        name=about['__title__'],
        description=about['__description__'],
        version=about['__version__'],
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'gulp_epic = epic_kitchens.gulp:main',
                'segment_epic = epic_kitchens.preprocessing.split_segments:main'
            ],
        },
        install_requires=[
            'gulpio>=500',
            'pandas',
            'numpy',
            'pillow-simd',
        ],
        extras_require={
            'dev': ['Sphinx', 'pygments', 'sphinx_rtd_theme', 'pytest', 'mypy']
        },
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        keywords=['dataset', 'egocentric', 'action-recogntion', 'epic', 'epic-kitchens'],
        author=about['__author__'],
        author_email=about['__author_email__'],
        license=about['__license__'],
        url='http://github.com/epic-kitchens/epic-lib',
        project_urls={
            'Bug Tracker': 'https://github.com/epic-kitchens/epic-lib/issues',
            'Documentation': 'https://epic-kitchens.readthedocs.io',
            'Dataset Website': 'https://epic-kitchens.github.io/',
            'Source Code': 'http://github.com/epic-kitchens/epic-lib',
        }
)
