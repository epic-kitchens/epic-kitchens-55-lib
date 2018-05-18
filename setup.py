#!/usr/bin/env python

from setuptools import find_packages, setup

import epic_kitchens

setup(
        name='epic-kitchens',
        description='EPIC-KITCHENS dataset utilities',
        version=epic_kitchens.__version__,
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'gulp_epic = epic_kitchens.gulp:main'
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
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        keywords=['dataset', 'egocentric', 'action-recogntion', 'epic', 'epic-kitchens'],
        author='EPIC-KITCHENS',
        author_email='uob-epic-kitchens2018@bristol.ac.uk',
        license='MIT',
        url='http://github.com/epic-kitchens/epic-lib',
        project_urls={
            'Bug Tracker': 'https://github.com/epic-kitchens/epic-lib/issues',
            'Documentation': 'https://epic-kitchens.readthedocs.io',
            'Dataset Website': 'https://epic-kitchens.github.io/',
            'Source Code': 'http://github.com/epic-kitchens/epic-lib',
        }
)
