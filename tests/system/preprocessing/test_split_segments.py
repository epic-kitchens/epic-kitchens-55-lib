from pathlib import Path

from epic_kitchens.preprocessing.split_segments import main, parser
from tests import ANNOTATIONS_DIR, MEDIA_DIR


def test_splitting_labelled_rgb(tmpdir):
    participant_id = "P01"
    video_id = "P01_01"
    modality = "rgb"
    p01_01_dir = get_segment_dir(tmpdir, participant_id, video_id)
    frame_dir = get_frame_dir(modality, participant_id, video_id)
    annotations_path = ANNOTATIONS_DIR / "EPIC_train_action_labels.csv"

    split_actions(video_id, frame_dir, p01_01_dir, annotations_path, modality)

    assert (p01_01_dir / "P01_01_0_open-door" / "frame_0000000001.jpg").exists()
    assert (p01_01_dir / "P01_01_0_open-door" / "frame_0000000194.jpg").exists()
    assert (p01_01_dir / "P01_01_10_open-drawer" / "frame_0000000001.jpg").exists()
    assert (p01_01_dir / "P01_01_10_open-drawer" / "frame_0000000074.jpg").exists()


def test_splitting_unlabelled_rgb(tmpdir):
    participant_id = "P01"
    video_id = "P01_11"
    modality = "rgb"
    p01_11_dir = get_segment_dir(tmpdir, participant_id, video_id)
    frame_dir = get_frame_dir(modality, participant_id, video_id)
    annotations_path = ANNOTATIONS_DIR / "EPIC_test_s1_timestamps.csv"

    split_actions(video_id, frame_dir, p01_11_dir, annotations_path, modality)

    assert (p01_11_dir / "P01_11_1924_unnarrated").exists()
    assert (p01_11_dir / "P01_11_1924_unnarrated" / "frame_0000000001.jpg").exists()
    assert (p01_11_dir / "P01_11_1924_unnarrated" / "frame_0000000112.jpg").exists()

    assert (p01_11_dir / "P01_11_1930_unnarrated").exists()
    assert (p01_11_dir / "P01_11_1930_unnarrated" / "frame_0000000001.jpg").exists()
    assert (p01_11_dir / "P01_11_1930_unnarrated" / "frame_0000000407.jpg").exists()


def test_splitting_labelled_flow(tmpdir):
    participant_id = "P01"
    video_id = "P01_01"
    modality = "flow"
    p01_01_dir = get_segment_dir(tmpdir, participant_id, video_id)
    frame_dir = get_frame_dir(modality, participant_id, video_id)
    annotations_path = ANNOTATIONS_DIR / "EPIC_train_action_labels.csv"

    split_actions(video_id, frame_dir, p01_01_dir, annotations_path, modality)

    for axis in "u", "v":
        assert (
            p01_01_dir / axis / "P01_01_0_open-door" / "frame_0000000001.jpg"
        ).exists()
        assert (
            p01_01_dir / axis / "P01_01_0_open-door" / "frame_0000000097.jpg"
        ).exists()
        assert (
            p01_01_dir / axis / "P01_01_10_open-drawer" / "frame_0000000001.jpg"
        ).exists()
        assert (
            p01_01_dir / axis / "P01_01_10_open-drawer" / "frame_0000000037.jpg"
        ).exists()


def test_splitting_unlabelled_flow(tmpdir):
    participant_id = "P01"
    video_id = "P01_11"
    modality = "flow"
    p01_11_dir = get_segment_dir(tmpdir, participant_id, video_id)
    frame_dir = get_frame_dir(modality, participant_id, video_id)
    annotations_path = ANNOTATIONS_DIR / "EPIC_test_s1_timestamps.csv"

    split_actions(video_id, frame_dir, p01_11_dir, annotations_path, modality)

    for axis in "u", "v":
        assert (
            p01_11_dir / axis / "P01_11_1924_unnarrated" / "frame_0000000001.jpg"
        ).exists()
        assert (
            p01_11_dir / axis / "P01_11_1924_unnarrated" / "frame_0000000056.jpg"
        ).exists()

        assert (p01_11_dir / axis / "P01_11_1930_unnarrated").exists()
        assert (
            p01_11_dir / axis / "P01_11_1930_unnarrated" / "frame_0000000001.jpg"
        ).exists()
        assert (
            p01_11_dir / axis / "P01_11_1930_unnarrated" / "frame_0000000204.jpg"
        ).exists()


def split_actions(video_id, frame_dir, segment_dir, annotations_path, modality):
    main(
        parser.parse_args(
            [
                video_id,
                str(frame_dir),
                str(segment_dir),
                str(annotations_path),
                modality,
            ]
        )
    )


def get_frame_dir(modality, participant_id, video_id):
    return MEDIA_DIR / modality / participant_id / video_id


def get_segment_dir(tmpdir, participant_id, video_id):
    segments_dir = Path(str(tmpdir.mkdir("segments")))
    video_segments_dir = segments_dir / participant_id / video_id
    video_segments_dir.mkdir(parents=True, exist_ok=False)
    return video_segments_dir
