#!/usr/bin/env python3

"""Program for splitting frames into action segments
See :ref:`cli_tools_action_segmentation` for usage details """

import argparse
import logging
import os
import pathlib
import sys

import pandas as pd

from epic_kitchens.labels import VIDEO_ID_COL
from epic_kitchens.video import Modality, FlowModality, RGBModality, split_video_frames

HELP = """\
Process frame dumps, and a set of annotations in a pickled dataframe
to produce a set of segmented action videos using symbolic links.


Taking a set of videos in the directory format (for RGB):

    P01_01
    |--- frame_0000000001.jpg
    |--- frame_0000000002.jpg
    |--- ...

Produce a set of action segments in the directory format:

    P01_01_0_chop-wood
    |--- frame_0000000001.jpg
    |--- ...
    |--- frame_0000000735.jpg


The final number `Z` in `PXX_YY_Z-narration` denotes the index of the segment, this can then
be used to look up the corresponding information on the segment such as the raw narration,
verb class, noun classes etc

If segmenting optical flow then frames are contained in a `u` or `v` subdirectory.
"""

LOG = logging.getLogger(__name__)


def main(args):
    print(args.frame_dir)
    print(args.links_dir)

    logging.basicConfig(level=logging.INFO)
    if not args.labels_pkl.exists():
        LOG.error('Annotations pickle: \'{}\' does not exist'.format(args.labels_pkl))
        sys.exit(1)

    annotations = pd.read_pickle(args.labels_pkl)
    fps = float(args.fps)
    if args.modality.lower() == 'rgb':
        frame_dirs = [args.frame_dir]
        links_dirs = [args.links_dir]
        modality = RGBModality(fps=fps)  # type: Modality
    elif args.modality.lower() == 'flow':
        axes = ['u', 'v']
        frame_dirs = [args.frame_dir.joinpath(axis) for axis in axes]
        links_dirs = [args.links_dir.joinpath(axis) for axis in axes]
        modality = FlowModality(rgb_fps=fps, stride=int(args.of_stride), dilation=int(args.of_dilation))
    else:
        print('Modality \'{}\' is not recognised'.format(args.modality))
        sys.exit(1)

    video_annotations = annotations[annotations[VIDEO_ID_COL] == args.video]
    for frame_dir, links_dir in zip(frame_dirs, links_dirs):
        common_root = os.path.commonpath([frame_dir, links_dir])
        print(common_root)
        split_video_frames(modality, args.frame_format, video_annotations, links_dir, frame_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=HELP,
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     )
    parser.add_argument('video', type=str,
                        help='Video ID to segment')
    parser.add_argument('frame_dir', type=lambda p: pathlib.Path(p).absolute(),
                        help='Path to frames, if RGB should contain images, if flow, should contain u, '
                             'v subdirectories with images')
    parser.add_argument('links_dir', type=lambda p: pathlib.Path(p).absolute(),
                        help='Path to save segments into')
    parser.add_argument('labels_pkl', type=pathlib.Path,
                        help='Path to the pickle file which contains the meta information about the dataset.')
    parser.add_argument('modality', type=str.lower, default='rgb', choices=['rgb', 'flow'],
                        help='Modality of frames that are being segmented')
    parser.add_argument('--frame-format', type=str, default='frame_%010d.jpg',
                        help='Pattern of frame filenames (default: %(default)s)')
    parser.add_argument('--fps', type=float, default=60,
                        help='FPS of extracted frames (default: %(default)s)')
    parser.add_argument('--of-stride', type=int, default=2,
                        help='Optical flow stride parameter used for frame extraction (default: %(default)s)')
    parser.add_argument('--of-dilation', type=int, default=3,
                        help='Optical flow dilation parameter used for frame extraction '
                             '(default: %(default)s)')
    main(parser.parse_args())
