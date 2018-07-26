CLI tools
=========

.. _cli_tools_action_segmentation:

Action segmentation
-------------------

A script is provided to segment frames from a video into a set of
action-segments.


.. code-block:: shell

    $ python -m epic_kitchens.preprocessing.split_segments \
        P03 \
        path/to/frames \
        path/to/frame-segments \
        path/to/labels.pkl \
        RGB \
        --fps 60 \
        --frame-format 'frame_%010d.jpg' \
        --of-stride 2 \
        --of-dilation 3

.. _cli_tools_gulp_ingestor:

Gulp data ingestor
------------------

.. code-block:: shell

    $ python -m epic_kitchens.gulp \
        path/to/frame-segments \
        path/to/gulp-dir \
        path/to/labels.pkl \
        RGB \
        --num-workers $(nproc) \
        --segments-per-chunk 100 \

