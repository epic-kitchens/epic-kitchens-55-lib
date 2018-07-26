"""Program for building GulpIO directory of frames for training
See :ref:`cli_tools_gulp_ingestor` for usage details """

from . import adapter
import pandas as pd


if __name__ == '__main__':
    import argparse
    from gulpio import GulpIngestor
    parser = argparse.ArgumentParser(
            'Gulp the EPIC dataset allowing for faster read times during training.')
    parser.add_argument('in_folder', type=str,
                        help='Directory where subdirectory is a segment name containing frames for that segment.')
    parser.add_argument('out_folder', type=str,
                        help='Directory to store the gulped files.')
    parser.add_argument('labels_pkl', type=str,
                        help='Path to the pickle file which contains the meta information about the dataset.')
    parser.add_argument('modality', choices=['flow', 'rgb'])
    parser.add_argument('--extension', type=str, default='jpg',
                        help='Which file extension the frames are saved as.')
    parser.add_argument('--frame-size', type=int, default=-1,
                        help='Size of frames.')
    parser.add_argument('--segments-per-chunk', type=int, default=100,
                        help='Number of videos per chunk to save.')
    parser.add_argument('-j', '--num-workers', type=int, default=4,
                        help='Number of workers to run the task.')

    args = parser.parse_args()

    labels = pd.read_pickle(args.labels_pkl)
    if args.modality.lower() == 'flow':
        epic_adapter = adapter.EpicFlowDatasetAdapter(args.in_folder, labels, args.frame_size,
                                                      args.extension)
    elif args.modality.lower() == 'rgb':
        epic_adapter = adapter.EpicDatasetAdapter(args.in_folder, labels, args.frame_size,
                                                  args.extension)
    else:
        raise ValueError("Modality '{}' not supported".format(args.modality))

    ingestor = GulpIngestor(epic_adapter,
                            args.out_folder,
                            args.segments_per_chunk,
                            args.num_workers)
    ingestor()
