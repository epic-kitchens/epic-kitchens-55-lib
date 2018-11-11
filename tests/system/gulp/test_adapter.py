import cv2
from skimage.measure import compare_ssim as ssim

from epic_kitchens.gulp.__main__ import main, parser
from gulpio.dataset import GulpDirectory
import pandas as pd
import numpy as np

from tests import SEGMENT_DIR, ANNOTATIONS_DIR


def test_gulping_labelled_rgb_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir("rgb_labelled_gulped")
    segment_dir = SEGMENT_DIR / "rgb"
    annotations_path = ANNOTATIONS_DIR / "EPIC_train_action_labels.csv"

    gulp(segment_dir, gulp_dir_path, annotations_path, "rgb", True)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 11)
    annotations = pd.read_csv(annotations_path, index_col="uid")
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]["meta_data"][0]
        assert metadata["verb_class"] == annotation.verb_class
        assert metadata["noun_class"] == annotation.noun_class

    assert_gulped_rgb_frames_similar_to_on_disk(
        gulp_dir, annotations, segment_dir, 1, max_discrepancy=10
    )


def test_gulping_unlabelled_rgb_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir("rgb_labelled_gulped")
    segment_dir = SEGMENT_DIR / "rgb"
    annotations_path = ANNOTATIONS_DIR / "EPIC_test_s1_timestamps.csv"

    gulp(segment_dir, gulp_dir_path, annotations_path, "rgb", False)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 7)
    annotations = pd.read_csv(annotations_path, index_col="uid")
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]["meta_data"][0]
        assert metadata["uid"] == annotation.Index


def test_gulping_labelled_flow_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir("flow_labelled_gulped")
    segment_dir = SEGMENT_DIR / "flow"
    annotations_path = ANNOTATIONS_DIR / "EPIC_train_action_labels.csv"

    gulp(segment_dir, gulp_dir_path, annotations_path, "flow", True)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 11)
    annotations = pd.read_csv(annotations_path, index_col="uid")
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]["meta_data"][0]
        assert metadata["verb_class"] == annotation.verb_class
        assert metadata["noun_class"] == annotation.noun_class

    assert_gulped_flow_frames_similar_to_on_disk(gulp_dir, annotations, segment_dir, 0)


def test_gulping_unlabelled_flow_segments(tmpdir):
    gulp_dir_path = tmpdir.mkdir("flow_unlabelled_gulped")
    segment_dir = SEGMENT_DIR / "flow"
    annotations_path = ANNOTATIONS_DIR / "EPIC_test_s1_timestamps.csv"

    gulp(segment_dir, gulp_dir_path, annotations_path, "flow", False)
    gulp_dir = GulpDirectory(str(gulp_dir_path))

    assert_number_of_segments(gulp_dir, 7)
    annotations = pd.read_csv(annotations_path, index_col="uid")
    for annotation in annotations.itertuples():
        metadata = gulp_dir.merged_meta_dict[str(annotation.Index)]["meta_data"][0]
        assert metadata["uid"] == annotation.Index


def assert_number_of_segments(gulp_dir, number_of_segments):
    segment_count = number_of_segments
    assert len(gulp_dir.merged_meta_dict) == segment_count


def assert_gulped_flow_frames_similar_to_on_disk(
    gulp_dir, annotations, segment_dir, uid, min_ssim=0.95
):
    """
    Assert that the first 2 gulped frames for a given UID are close to those on disk.
    ``max_discrepancy`` controls how much of a difference per pixel is tolerable, set it to a value in
    [0, 255]
    """
    for axis in "u", "v":
        segment_path = get_segment_path(segment_dir, annotations, axis, uid)
        frame_paths = [segment_path / ("frame_%010d.jpg" % i) for i in (1, 2)]
        u_frames = read_images(frame_paths)
        gulp_frames, _ = gulp_dir[uid, slice(0, 4)]
        gulp_u_frames = gulp_frames[::2]
        gulp_v_frames = gulp_frames[1::2]
        gulp_frames = gulp_u_frames if axis == "u" else gulp_v_frames
        for frame, gulp_frame in zip(u_frames, gulp_frames):
            assert frame.shape == gulp_frame.shape

            computed_ssim = ssim(gulp_frame, frame, data_range=255.0)
            assert computed_ssim >= min_ssim


def assert_gulped_rgb_frames_similar_to_on_disk(
    gulp_dir, annotations, segment_dir, uid, max_discrepancy=1
):
    """
    Assert that the first 2 gulped frames for a given UID are close to those on disk.
    ``max_discrepancy`` controls how much of a difference per pixel is tolerable, set it to a value in [0, 255]
    """
    segment_path = get_segment_path(segment_dir, annotations, None, uid)
    frame_paths = [segment_path / ("frame_%010d.jpg" % i) for i in range(1, 3)]
    frames = read_images(frame_paths)
    gulp_frames, _ = gulp_dir[uid, slice(0, 2)]
    for i, (frame, gulp_frame) in enumerate(zip(frames, gulp_frames)):
        assert frame.shape == gulp_frame.shape

        # We let there be a maximum discrepancy of 1 per pixel throughout the image
        max_discrepancy = np.prod(frame.shape) * max_discrepancy
        discrepancy = np.sum(np.abs(frame - gulp_frame))
        assert discrepancy <= max_discrepancy


def get_segment_path(root_segment_path, annotations, axis, uid):
    annotation = annotations.loc[uid]
    participant_id, video_id, narration = annotation[
        ["participant_id", "video_id", "narration"]
    ]
    segment_name = "{}_{}_{}".format(video_id, uid, narration.strip().replace(" ", "-"))
    pre_axis_path = root_segment_path / participant_id / video_id
    if axis:
        return pre_axis_path / axis / segment_name
    else:
        return pre_axis_path / segment_name


def read_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR)
        if image.ndim == 3:
            image = bgr_to_rgb(image)
        images.append(image)
    return images


def bgr_to_rgb(image):
    return image[..., ::-1]


def gulp(segment_root_dir, gulp_dir, label_path, modality, labelled):
    main(
        parser.parse_args(
            [str(segment_root_dir), str(gulp_dir), str(label_path), modality]
            + (["--unlabelled"] if not labelled else [])
        )
    )
