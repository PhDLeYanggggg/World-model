"""Synthetic parser fixtures, never actual DroneCrowd annotation evidence."""
import json

import pytest

from scripts.audit_m3w_dronecrowd_metadata import acquire, plain_path, verify_local
from src.evaluation.m3w_dronecrowd_intake import (
    METADATA_FILES, audit_release, derived_identity, inspect_xml,
    require_forecast_admission, require_independent_release_uses, sequence_ids,
    validate_metadata,
)


def metadata_fixture():
    # Minimal signatures, not a copy of the author's source or real annotation.
    return {
        'README.md': b'# The DroneCrowd Dataset\nval set is sampled from the test set\n'
                     b'only for academic and non-commercial uses\n',
        'trainlist.txt': '\r\n'.join(f'{i:05d}' for i in range(1, 83)).encode(),
        'testlist.txt': '\r\n'.join(f'{i:05d}' for i in range(83, 113)).encode(),
        'xml2mat.m': b"function [anno, countNum, label] = xml2mat(\n"
                     b"if(exist(matfile, 'file'))\nif(outside~=1 && occluded~=1)\n"
                     b"[frame id xtl ytl xbr ybr]\nsave(matfile, 'anno', 'countNum', 'label')",
        'saveGT.m': b'xml2mat(annoPath, seqName)\nidx = anno(:,1)==i-1;\n'
                    b'anno(idx,2)+1\nfor i = 1:300\n'
                    b'(anno(idx,3)+anno(idx,5))/2, (anno(idx,4)+anno(idx,6))/2',
    }


@pytest.mark.parametrize('bad', [b'', b'1', b'00000', b'00001\n00001', b'00001\n\n00002',
                                  b'../00001', b'00001 ', b'00001\n1.0', b'1e001'])
def test_sequence_aliases_and_duplicate_ids_refused(bad):
    with pytest.raises(ValueError):
        sequence_ids(bad)


def test_windows_line_endings_preserve_sequence_identity():
    assert sequence_ids(b'00001\r\n00112\r\n') == ['00001', '00112']


def test_small_metadata_allowlist_cannot_download_annotations():
    with pytest.raises(ValueError, match='no archives'):
        validate_metadata('annotations.zip', b'PK\x03\x04')
    assert len(METADATA_FILES) == 5
    assert not any(n.endswith(('.zip', '.xml')) for n in METADATA_FILES)


@pytest.mark.parametrize('data', [b'<html>Download anyway</html>', b'<!DOCTYPE html>login',
                                  b'<form>Confirm download</form>', b'a' * 65537, b''])
def test_download_warning_and_invalid_response_are_not_data(data):
    with pytest.raises(ValueError):
        validate_metadata('README.md', data)


def test_release_partition_and_independence_are_different():
    result = audit_release(metadata_fixture())
    assert result['official_id_partition_pass'] is True
    assert result['validation_independent_of_test'] is False
    assert result['physical_sites'] is None
    assert result['admitted_recordings'] == 0
    assert result['raw_xml_status'].startswith('not_run')
    assert result['conversion']['visibility_and_keyframe_provenance_in_six_column_mat'] is False
    with pytest.raises(ValueError, match='cannot approve'):
        require_forecast_admission(result)


def test_release_overlap_is_reported_not_silently_removed():
    files = metadata_fixture()
    files['testlist.txt'] = b'00001\n00112'
    result = audit_release(files)
    assert result['sequence_list_overlap'] == ['00001']
    assert result['official_id_partition_pass'] is False


def test_changed_source_semantics_require_review():
    files = metadata_fixture()
    files['saveGT.m'] = files['saveGT.m'].replace(b'anno(idx,2)+1', b'anno(idx,2)')
    with pytest.raises(ValueError, match='differs from reviewed'):
        audit_release(files)


@pytest.mark.parametrize('left,right', [('fit', 'confirmation'), ('development', 'calibration'),
                                       ('calibration', 'confirmation')])
def test_supplied_validation_and_test_cannot_be_independent(left, right):
    with pytest.raises(ValueError, match='Dependent release'):
        require_independent_release_uses([
            {'release_folder': 'val', 'role': left}, {'release_folder': 'test', 'role': right}])


def test_excluded_or_same_role_is_not_automatically_admitted():
    uses = [{'release_folder': 'val', 'role': 'excluded'},
            {'release_folder': 'test', 'role': 'confirmation'}]
    assert require_independent_release_uses(uses) is None
    with pytest.raises(ValueError, match='cannot approve'):
        require_forecast_admission({'uses': uses, 'looks_good': True})
    with pytest.raises(ValueError, match='Unknown'):
        require_independent_release_uses([{'release_folder': 'validation', 'role': 'fit'}])


def test_identity_offsets_and_sequence_local_ids():
    result = derived_identity('00001', 0, 0)
    assert result == {'image_name': 'img001001.jpg', 'derived_mat_id': 1,
                      'canonical_agent_key': 'dronecrowd:00001:0'}
    assert derived_identity('00112', 299, 0)['image_name'] == 'img112300.jpg'
    assert derived_identity('00112', 299, 0)['canonical_agent_key'] != result['canonical_agent_key']


@pytest.mark.parametrize('sequence,frame,agent', [('00000', 0, 0), ('1', 0, 0), ('00113', 0, 0),
                                               ('00001', -1, 0), ('00001', 300, 0),
                                               ('00001', True, 0), ('00001', 0, 1.1),
                                               ('00001', 0, -1), ('00001', 0, 2**53)])
def test_non_exact_or_out_of_range_identity_refused(sequence, frame, agent):
    with pytest.raises(ValueError):
        derived_identity(sequence, frame, agent)


def box(frame, outside=0, occluded=0, extra=''):
    return (f'<box frame="{frame}" xtl="0" ytl="0" xbr="4" ybr="6" '
            f'outside="{outside}" occluded="{occluded}" {extra}/>')


def xml_fixture(boxes, label='pedestrian'):
    return f'<annotations count="1"><track id="0" label="{label}">{boxes}</track></annotations>'.encode()


def test_raw_xml_inspection_preserves_filtered_gaps_without_interpolation():
    data = xml_fixture(box(0) + box(1, occluded=1) + box(2, outside=1) + box(3, extra='keyframe="1"'))
    compact = inspect_xml(data, '00001')
    spaced = inspect_xml(data.replace(b'><', b'>\n  <'), '00001')
    assert compact['raw_boxes'] == spaced['raw_boxes'] == 4
    assert compact['converter_retained_boxes'] == 2
    assert compact['all_box_discontinuous_edges'] == 0
    assert compact['retained_box_discontinuous_edges'] == 1
    assert compact['box_attribute_counts']['keyframe'] == 1
    assert compact['interpolation_provenance'].startswith('unresolved')
    assert compact['causal_feature_rows_exported'] == compact['admitted_recordings'] == 0


@pytest.mark.parametrize('data', [
    b'<!DOCTYPE foo [<!ENTITY x "a">]><foo/>',
    b'<annotations count="0"/>',
    xml_fixture(box(0)).replace(b'count="1"', b'count="2"'),
    xml_fixture(box(0) + box(0)), xml_fixture(box(3) + box(2)),
    xml_fixture(box(0, outside=2)), xml_fixture(box(300)),
    xml_fixture(box(0)).replace(b'xbr="4"', b'xbr="nan"'),
    xml_fixture(box(0)).replace(b'xbr="4"', b'xbr="-1"'),
    xml_fixture(box(0, extra='keyframe="unknown"')),
    xml_fixture(box(0), label=''), xml_fixture(''),
])
def test_invalid_xml_never_becomes_eligible(data):
    with pytest.raises(ValueError):
        inspect_xml(data, '00001')


def test_duplicate_track_id_and_alternate_xml_encoding_refused():
    data = xml_fixture(box(0)).replace(b'</annotations>', b'<track id="0" label="other">'
                + box(1).encode() + b'</track></annotations>').replace(b'count="1"', b'count="2"')
    with pytest.raises(ValueError, match='Duplicate'):
        inspect_xml(data, '00001')
    with pytest.raises(ValueError):
        inspect_xml(xml_fixture(box(0)).decode().encode('utf-16'), '00001')


def test_source_capture_and_offline_hash_verification(tmp_path):
    files = metadata_fixture()
    source, output = tmp_path / 'raw', tmp_path / 'reports'
    acquired, downloaded = acquire(source, output, files.__getitem__)
    assert acquired == files and downloaded == 5
    manifest = json.loads((output / 'source_manifest.json').read_text())
    assert verify_local(source, manifest) == files
    assert manifest['annotations_images_videos_downloaded'] is False
    (source / 'trainlist.txt').write_bytes(b'00099')
    with pytest.raises(ValueError, match='changed'):
        verify_local(source, manifest)


def test_invalid_download_does_not_create_a_raw_file(tmp_path):
    source, output = tmp_path / 'raw', tmp_path / 'reports'
    with pytest.raises(ValueError):
        acquire(source, output, lambda name: b'<html>Confirm</html>')
    assert not (source / 'README.md').exists()


def test_interrupted_unbound_acquisition_compares_existing_bytes(tmp_path):
    source, output = tmp_path / 'raw', tmp_path / 'reports'
    source.mkdir()
    (source / 'README.md').write_bytes(b'old')
    with pytest.raises(ValueError, match='Unbound'):
        acquire(source, output, metadata_fixture().__getitem__)


def test_symlink_path_is_refused_and_parent_paths_resolved(tmp_path):
    link = tmp_path / 'link'
    link.symlink_to(tmp_path / 'target')
    with pytest.raises(ValueError, match='Symlinked'):
        plain_path(link / 'x')
    assert plain_path(tmp_path / 'subdir' / '..' / 'other') == tmp_path / 'other'
