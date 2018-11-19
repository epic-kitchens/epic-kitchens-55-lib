from epic_kitchens import meta


def test_default_version_is_1_5_0():
    assert meta._annotation_repository.version == "v1.5.0"


def test_changing_annotation_version():
    version = "v1.4.0"
    assert meta._annotation_repository.version != version
    meta.set_version(version)
    assert meta._annotation_repository.version == version
