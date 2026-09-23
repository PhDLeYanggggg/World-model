"""Synthetic archive/schema checks; real-source evidence is reported separately."""
import io
import stat
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from scripts.audit_m3w_dronecrowd_annotations import require_archive_identity, require_authorization
from src.evaluation.m3w_dronecrowd_archive import (
    ARCHIVE_FILE_ID, archive_inventory, audit_archive, compare_clean_text,
    compare_mat, contiguous_lengths, inspect_recording, window_counts,
)


def xml(n=21, hidden=()):
    boxes = "".join(f'<box frame="{f}" xtl="{f}" ytl="0" xbr="{f+2}" ybr="2" '
                    f'outside="0" occluded="{int(f in hidden)}"/>' for f in range(n))
    return ('<annotations count="1"><track id="0" label="human">' + boxes
            + '</track></annotations>').encode()


def mat_bytes(rows):
    buffer = io.BytesIO()
    savemat(buffer, {"anno": np.asarray(rows), "label": "human"})
    return buffer.getvalue()


def test_visibility_gaps_not_interpolated_or_counted_as_complete_windows():
    report, rows, _ = inspect_recording(xml(21, hidden={10}), "00001")
    assert report["raw_boxes"] == 21 and len(rows) == 20
    assert report["structural_windows"]["obs8_pred12_stride1"] == 0
    assert report["history_only_windows"]["8"] == 6
    assert report["admitted_recordings"] == 0
    assert report["interpolation_provenance"].startswith("unresolved")


def test_window_off_by_one_and_stride_distinction():
    assert window_counts([20])["obs8_pred12_stride1"] == 1
    assert window_counts([19])["obs8_pred12_stride1"] == 0
    assert window_counts([229])["obs8_pred12_stride12"] == 1
    assert window_counts([58])["obs8_pred50_stride1"] == 1
    assert contiguous_lengths([0, 1, 2, 5, 6]) == [3, 2]
    assert contiguous_lengths([]) == []
    with pytest.raises(ValueError):
        contiguous_lengths([0, 0])


def test_mat_compared_without_hiding_coordinate_changes():
    _, rows, _ = inspect_recording(xml(), "00001")
    assert compare_mat(mat_bytes(rows[::-1]), rows)["exact_rows_match_visible_xml"]
    changed = rows.copy()
    changed[0, 2] += 1
    result = compare_mat(mat_bytes(changed), rows)
    assert not result["exact_rows_match_visible_xml"]
    assert result["mismatched_scalar_values"] == 1
    assert result["mismatched_rows"] == 1 and result["identity_columns_match"]
    assert result["max_absolute_coordinate_difference"] == 1


def test_clean_text_geometry_hypothesis_is_checked_not_assumed():
    _, rows, _ = inspect_recording(xml(), "00001")
    clean = np.c_[rows.copy(), np.ones((len(rows), 2)), np.zeros((len(rows), 2))]
    clean[:, 0] += 1
    clean[:, 4:6] -= clean[:, 2:4]
    buffer = io.BytesIO()
    np.savetxt(buffer, clean, delimiter=",")
    assert compare_clean_text(buffer.getvalue(), rows)["frame_plus_one_xywh_hypothesis_exact_match"]
    clean[:, 1] += 1
    buffer = io.BytesIO()
    np.savetxt(buffer, clean, delimiter=",")
    assert not compare_clean_text(buffer.getvalue(), rows)["frame_plus_one_xywh_hypothesis_exact_match"]
    assert not compare_clean_text(buffer.getvalue(), rows)["identity_columns_match"]


def test_bound_archive_bytes_must_not_change(tmp_path):
    import hashlib
    path = tmp_path / "archive.zip"
    path.write_bytes(b"snapshot")
    receipt = {"archive_sha256": hashlib.sha256(b"snapshot").hexdigest(), "downloaded_bytes": 8}
    require_archive_identity(path, receipt)
    path.write_bytes(b"snapshoT")
    with pytest.raises(ValueError, match="differs"):
        require_archive_identity(path, receipt)


def test_visible_out_of_image_is_reported_separately():
    data = xml().replace(b'xtl="0"', b'xtl="-1"')
    report, _, _ = inspect_recording(data, "00001")
    assert report["visible_boxes_outside_nominal_1920x1080"] == 1


@pytest.mark.parametrize("name", ["../a.xml", "/a.xml", "annotations/../00001.xml",
    "annotations//00001.xml", "annotations/00001.py", "annotations/00113.xml", "annotations\\00001.xml"])
def test_unsafe_or_unreviewed_zip_members_refused(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(name, b"abc")
    with zipfile.ZipFile(buffer) as z, pytest.raises(ValueError):
        archive_inventory(z)


def test_zip_symlink_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        member = zipfile.ZipInfo("annotations/00001.xml")
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(member, b"../../outside")
    with zipfile.ZipFile(buffer) as z, pytest.raises(ValueError, match="link"):
        archive_inventory(z)


def test_authorization_is_not_implicit_model_or_role_approval():
    receipt = {"route": "independent_external_scenes_first", "annotation_download_and_audit": True,
               "archive_file_id": ARCHIVE_FILE_ID, "scientific_roles_assigned": False,
               "execute_downloaded_content": False, "user_message": "explicit scoped decision"}
    require_authorization(receipt)
    for key, value in (("annotation_download_and_audit", False), ("scientific_roles_assigned", True),
                       ("execute_downloaded_content", True), ("user_message", "")):
        with pytest.raises(ValueError):
            require_authorization({**receipt, key: value})


def test_archive_audit_never_exports_model_rows_or_assigns_roles(tmp_path):
    path = tmp_path / "annotations.zip"
    _, rows, _ = inspect_recording(xml(), "00001")
    with zipfile.ZipFile(path, "w") as z:
        for seq in ("00001", "00002"):
            z.writestr(f"annotations/{seq}.xml", xml())
            z.writestr(f"annotations/{seq}.mat", mat_bytes(rows))
    result = audit_archive(path, ["00001"], ["00002"])
    assert result["aggregate"]["all"]["raw_boxes"] == 42
    assert result["aggregate"]["test"]["structural_windows"]["obs8_pred12_stride1"] == 2
    assert len(result["identical_xml_groups"]) == 1
    assert len(result["exact_cross_sequence_track_groups_min20_rows"]) == 1
    assert result["admitted_recordings"] == result["model_feature_rows_exported"] == 0
    assert not result["scientific_roles_assigned"]
    assert result["forecast_evaluation"] == "not_run"
    assert list(tmp_path.iterdir()) == [path]


def test_missing_release_member_stops_audit(tmp_path):
    path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("annotations/00001.xml", xml())
    with pytest.raises(ValueError, match="MAT coverage"):
        audit_archive(path, ["00001"], [])


def test_bad_recording_is_quarantined_not_silently_fixed(tmp_path):
    path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("annotations/00001.xml", xml().replace(b'count="1"', b'count="2"'))
        z.writestr("annotations/00001.mat", b"not read after XML refusal")
    result = audit_archive(path, ["00001"], [])
    assert len(result["structural_failures"]) == 1
    assert result["aggregate"]["all"]["sequences"] == 0
    assert result["admitted_recordings"] == 0
