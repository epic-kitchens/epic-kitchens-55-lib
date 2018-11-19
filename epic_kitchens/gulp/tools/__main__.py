import pandas as pd

from epic_kitchens.gulp.tools.meta_modification import modify_metadata


def main(args):
    annotations = pd.read_pickle(args.annotations_pkl)

    def update_annotation(segment_id, old_segment_metadata):
        segment_metadata = annotations.loc[int(segment_id)].to_dict()
        # Update the existing dictionary in case have extra items in
        # the old_segment_metadata dict that we would lose if we simply replaced it.
        for key, val in segment_metadata.items():
            old_segment_metadata[key] = val
        return old_segment_metadata

    for dir in args.gulp_dirs:
        modify_metadata(dir, update_annotation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update the metadata in a gulp directory from a given dataframe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "annotations_pkl",
        type=Path,
        help="Path to pickled dataframe containing all annotations",
    )
    parser.add_argument("gulp_dirs", type=Path, nargs="+", metavar="GULP_DIR")
    parser.add_argument(
        "--drop-unknown",
        action="store_true",
        help="Drop entries for segments not found in annotations dataframe",
    )
    args = parser.parse_args()

    main(args)
