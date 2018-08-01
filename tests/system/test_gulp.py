from pathlib import Path

from epic_kitchens.gulp.__main__ import main, parser
from gulpio.dataset import GulpDirectory
import pandas as pd

DATASET_DIR = Path(__file__).parent.parent / 'dataset'
SEGMENT_DIR =  DATASET_DIR / 'media' / 'segments'
ANNOTATIONS_DIR = DATASET_DIR / 'annotations'


def test_gulping_labelled_rgb_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir('rgb_labelled_gulped')
    segment_dir = SEGMENT_DIR / 'rgb'
    annotations_path = ANNOTATIONS_DIR / 'EPIC_train_action_labels.csv'

    gulp(segment_dir, gulp_dir_path, annotations_path, 'rgb', True)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 11)
    annotations = pd.read_csv(annotations_path, index_col='uid')
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]['meta_data'][0]
        assert metadata['verb_class'] == annotation.verb_class
        assert metadata['noun_class'] == annotation.noun_class


def test_gulping_unlabelled_rgb_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir('rgb_labelled_gulped')
    segment_dir = SEGMENT_DIR / 'rgb'
    annotations_path = ANNOTATIONS_DIR / 'EPIC_test_s1_timestamps.csv'

    gulp(segment_dir, gulp_dir_path, annotations_path, 'rgb', False)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 7)
    annotations = pd.read_csv(annotations_path, index_col='uid')
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]['meta_data'][0]
        assert metadata['uid'] == annotation.Index


def test_gulping_labelled_flow_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir('flow_labelled_gulped')
    segment_dir = SEGMENT_DIR / 'flow'
    annotations_path = ANNOTATIONS_DIR / 'EPIC_train_action_labels.csv'

    gulp(segment_dir, gulp_dir_path, annotations_path, 'flow', True)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 11)
    annotations = pd.read_csv(annotations_path, index_col='uid')
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]['meta_data'][0]
        assert metadata['verb_class'] == annotation.verb_class
        assert metadata['noun_class'] == annotation.noun_class


def test_gulping_unlabelled_flow_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir('flow_unlabelled_gulped')
    segment_dir = SEGMENT_DIR / 'flow'
    annotations_path = ANNOTATIONS_DIR / 'EPIC_test_s1_timestamps.csv'

    gulp(segment_dir, gulp_dir_path, annotations_path, 'flow', False)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 7)
    annotations = pd.read_csv(annotations_path, index_col='uid')
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]['meta_data'][0]
        assert metadata['uid'] == annotation.Index


def assert_number_of_segments(gulp_dir, number_of_segments):
    segment_count = number_of_segments
    assert len(gulp_dir.merged_meta_dict) == segment_count


def gulp(segment_root_dir, gulp_dir, label_path, modality, labelled):
    main(parser.parse_args([str(segment_root_dir), str(gulp_dir), str(label_path), modality] +
                           (["--unlabelled"] if not labelled else [])))