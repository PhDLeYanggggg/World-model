"""DroneCrowd release/format checks, not scientific-role admission.

Only source metadata is acquired here. Raw XML inspection is an explicit local
operation and never exports coordinates, futures, velocities or training rows.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


FOLDER_URL = 'https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN'
METADATA_FILES = {
    'README.md': '1H0BpOCa7qqZ6E-SPyQENaR7HMtuw1R5c',
    'trainlist.txt': '1ZnKYzsCZ16RdNnyc2AkHKpmAQtO4AxPq',
    'testlist.txt': '1ZfT8RIcOnz_fiReREA0A89xNB_BO2sZ2',
    'xml2mat.m': '1IEghMNXjI8l0jIkSj9tu4J9YP3ZKmJIo',
    'saveGT.m': '1fP-XXJCaGH-IWifA6EWDI1qQ0SJxHUtB',
}
MAX_METADATA_BYTES = 65536
MAX_XML_BYTES = 64 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sequence_ids(data):
    lines = data.decode('utf-8-sig').splitlines()
    if not lines or any(not re.fullmatch(r'[0-9]{5}', s) or int(s) == 0 for s in lines):
        raise ValueError('Expected nonempty five-digit sequence IDs, not paths or aliases')
    if len(set(lines)) != len(lines):
        raise ValueError('Duplicate sequence identity')
    return lines


def validate_metadata(name, data):
    if name not in METADATA_FILES:
        raise ValueError('Only the five small metadata files are allowed; no archives')
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_METADATA_BYTES:
        raise ValueError('Metadata response has invalid size')
    text = data.decode('utf-8-sig')
    if '\x00' in text or re.search(r'<(?:!doctype\s+html|html|form|script)\b', text, re.I):
        raise ValueError('HTML, sign-in or download confirmation is not source metadata')
    if name.endswith('list.txt'):
        sequence_ids(data)
    if name == 'README.md' and '# The DroneCrowd Dataset' not in text:
        raise ValueError('Unrecognized release README')
    if name == 'xml2mat.m' and not text.startswith('function [anno, countNum, label] = xml2mat('):
        raise ValueError('Unrecognized converter source')
    if name == 'saveGT.m' and 'xml2mat(annoPath, seqName)' not in text:
        raise ValueError('Unrecognized point converter source')
    return text


def audit_release(files):
    if set(files) != set(METADATA_FILES):
        raise ValueError('Exactly the fixed metadata file set is required')
    texts = {n: validate_metadata(n, data) for n, data in files.items()}
    train, test = sequence_ids(files['trainlist.txt']), sequence_ids(files['testlist.txt'])
    checks = {
        'validation_sampled_from_test': 'val set is sampled from the test set' in texts['README.md'],
        'academic_noncommercial_statement': 'only for academic and non-commercial uses' in texts['README.md'],
        'cached_mat_preferred': "if(exist(matfile, 'file'))" in texts['xml2mat.m'],
        'visibility_filter': 'if(outside~=1 && occluded~=1)' in texts['xml2mat.m'],
        'six_column_mat': '[frame id xtl ytl xbr ybr]' in texts['xml2mat.m'],
        'single_label_output': "save(matfile, 'anno', 'countNum', 'label')" in texts['xml2mat.m'],
        'frame_index_plus_one': 'idx = anno(:,1)==i-1;' in texts['saveGT.m'],
        'agent_index_plus_one': 'anno(idx,2)+1' in texts['saveGT.m'],
        'box_center_not_footpoint': '(anno(idx,3)+anno(idx,5))/2, (anno(idx,4)+anno(idx,6))/2' in texts['saveGT.m'],
        'three_hundred_frames': 'for i = 1:300' in texts['saveGT.m'],
    }
    if not all(checks.values()):
        raise ValueError('Release source differs from reviewed semantics; review before conversion')
    overlap = sorted(set(train) & set(test))
    union = set(train) | set(test)
    return {
        'schema_version': 1, 'result_source': 'fresh_run',
        'scope': 'release_metadata_and_conversion_source_only',
        'source_folder': FOLDER_URL,
        'source_sha256': {n: sha(b) for n, b in sorted(files.items())},
        'upstream_immutable_version': None,
        'counts': {'train_sequences': len(train), 'test_sequences': len(test),
                   'union_sequences': len(union)},
        'sequence_list_overlap': overlap,
        'official_id_partition_pass': len(train) == 82 and len(test) == 30 and not overlap
            and union == {f'{i:05d}' for i in range(1, 113)},
        'source_pattern_checks_not_code_execution': checks,
        'validation_independent_of_test': False,
        'validation_membership': 'not_audited_no_val_archive_read',
        'physical_sites': None,
        'sequence_disjointness_is_not_site_independence': True,
        'conversion': {'xml_frame_base': 0, 'derived_image_frame_base': 1,
                       'derived_id_offset': 1, 'point_semantics': 'bounding_box_center',
                       'occluded_and_outside_rows': 'removed_by_official_converter',
                       'per_track_labels_in_six_column_mat': False,
                       'visibility_and_keyframe_provenance_in_six_column_mat': False},
        'raw_xml_status': 'not_run_download_confirmation_pending',
        'raw_interpolation_or_keyframe_semantics': 'unknown',
        'camera_motion_and_physical_scene_map': 'unknown',
        'historical_forecast_exposure': 'unreviewed',
        'coordinate_unit': 'image_pixels_not_metric',
        'effective_seconds': 'unknown_no_annotation_time_mapping',
        'admitted_recordings': 0, 'scientific_roles_assigned': False,
        'model_training': 'not_run', 'forecast_evaluation': 'not_run',
        'stage5c_executed': False, 'smc_enabled': False,
    }


def require_independent_release_uses(uses):
    """Reject known official val/test reuse; passing is not role approval.

    This limited check does not establish independence between distinct clips,
    locate physical sites, or replace the experiment's full lineage contract.
    """
    known = {'train': 'train', 'val': 'test', 'test': 'test'}
    active = {'fit', 'development', 'calibration', 'confirmation'}
    owners = {}
    for use in uses:
        folder, role = use.get('release_folder'), use.get('role')
        if folder not in known or role not in active | {'excluded'}:
            raise ValueError('Unknown release folder or scientific role')
        if role == 'excluded':
            continue
        family = known[folder]
        if family in owners and owners[family] != role:
            raise ValueError('Dependent release folders cannot define independent scientific roles')
        owners[family] = role


def derived_identity(sequence_id, xml_frame, xml_agent_id):
    if not re.fullmatch(r'[0-9]{5}', sequence_id) or not 1 <= int(sequence_id) <= 112:
        raise ValueError('Sequence outside reviewed full-release ID range')
    if type(xml_frame) is not int or not 0 <= xml_frame < 300:
        raise ValueError('XML frame outside reviewed converter range')
    if type(xml_agent_id) is not int or not 0 <= xml_agent_id < 2**53 - 1:
        raise ValueError('Agent identity must be a nonnegative exact integer')
    return {'image_name': f'img{int(sequence_id):03d}{xml_frame+1:03d}.jpg',
            'derived_mat_id': xml_agent_id + 1,
            'canonical_agent_key': f'dronecrowd:{sequence_id}:{xml_agent_id}'}


def _integer(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'0|[1-9][0-9]*', value):
        raise ValueError(f'{name}: nonnegative canonical integer required')
    return int(value)


def inspect_xml(data, sequence_id):
    """Structural screen only. Raw-data compatibility remains untested until read.

    Unknown attributes are inventoried, never inferred as causal provenance.
    Gaps and removed visibility records are preserved in counts, not interpolated.
    """
    derived_identity(sequence_id, 0, 0)
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_XML_BYTES:
        raise ValueError('XML size outside bounded single-recording screen')
    # Reject alternate encodings and entity declarations before ElementTree parse.
    text = data.decode('utf-8-sig')
    if '\x00' in text or re.search(r'<!\s*(?:DOCTYPE|ENTITY)\b', text, re.I):
        raise ValueError('DTD/entities/alternate XML encodings refused')
    root = ET.fromstring(text)
    tracks = list(root.iter('track'))
    if not tracks or _integer(root.get('count'), 'track count') != len(tracks):
        raise ValueError('Declared and actual nonempty track counts must agree')
    ids, labels, attributes = set(), Counter(), Counter()
    kept, outside, occluded, total, gaps, discarded_gaps = 0, 0, 0, 0, 0, 0
    frame_values = []
    for track in tracks:
        agent = _integer(track.get('id'), 'agent ID')
        if agent in ids:
            raise ValueError('Duplicate track ID')
        ids.add(agent)
        label = track.get('label')
        if not label or not label.strip():
            raise ValueError('Missing per-track label')
        labels[label] += 1
        previous, previous_kept = None, None
        if not list(track):
            raise ValueError('Empty track')
        for box in track:
            frame = _integer(box.get('frame'), 'frame')
            derived_identity(sequence_id, frame, agent)
            if previous is not None:
                if frame <= previous:
                    raise ValueError('Repeated or unordered frame within track')
                gaps += frame != previous + 1
            previous = frame
            coords = [float(box.attrib[k]) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
            if not all(math.isfinite(v) for v in coords) or coords[0] >= coords[2] or coords[1] >= coords[3]:
                raise ValueError('Invalid finite positive-area xyxy box')
            out, occ = box.get('outside'), box.get('occluded')
            if out not in {'0', '1'} or occ not in {'0', '1'}:
                raise ValueError('Explicit binary visibility flags required')
            if box.get('keyframe') not in {None, '0', '1'}:
                raise ValueError('Unknown keyframe flag')
            attributes.update(box.attrib.keys())
            total += 1
            outside += out == '1'
            occluded += occ == '1'
            frame_values.append(frame)
            if out == occ == '0':
                kept += 1
                if previous_kept is not None:
                    discarded_gaps += frame != previous_kept + 1
                previous_kept = frame
    return {
        'schema_version': 1, 'sequence_id': sequence_id, 'source_xml_sha256': sha(data),
        'status': 'structure_only_not_forecast_admission', 'tracks': len(tracks),
        'labels': dict(sorted(labels.items())), 'raw_boxes': total,
        'outside_boxes': outside, 'occluded_boxes': occluded,
        'converter_retained_boxes': kept, 'all_box_discontinuous_edges': gaps,
        'retained_box_discontinuous_edges': discarded_gaps,
        'observed_frame_range': [min(frame_values), max(frame_values)],
        'box_attribute_counts': dict(sorted(attributes.items())),
        'interpolation_provenance': 'unresolved_even_if_keyframe_flag_present',
        'camera_or_physical_site_verified': False,
        'causal_feature_rows_exported': 0, 'admitted_recordings': 0,
    }


def require_forecast_admission(_screen):
    raise ValueError('Metadata/structural screen cannot approve forecast roles or causal provenance')


def render_release_report(result):
    c = result['counts']
    return '\n'.join([
        '# DroneCrowd Release Metadata Audit', '', '## Material Passport', '',
        'Fresh small-file acquisition and structural metadata analysis. No annotation',
        'archive, images or video read; no third-party code executed; no new forecasts.',
        '', f"Official lists: {c['train_sequences']} train / {c['test_sequences']} test / "
        f"{c['union_sequences']} distinct sequence IDs.",
        f"ID partition check: {result['official_id_partition_pass']}.",
        'This is clip-ID disjointness, not physical-site independence or full no-leakage.',
        '', 'The release README states that val is sampled from test. These folders',
        'cannot define independent selection/calibration/confirmation roles. Their',
        'actual frame membership has not been read. All scientific roles remain unassigned.',
        '', '## Converter Semantics', '',
        '- Source XML frame f maps to image index f+1 for frames 0..299.',
        '- Source agent ID maps to derived MAT ID+1; canonical ID is sequence-local.',
        '- The point is a bounding-box center, not a ground-plane footpoint.',
        '- Outside and occluded records are filtered, which can break continuity.',
        '- The six-column output lacks per-track types, visibility and keyframe provenance.',
        '- Existing MAT is loaded without checking whether its XML source changed.',
        '- These are reviewed-source/pattern checks, not execution of author code.',
        '', '## Remaining Requirements', '',
        'Original XML structure, interpolation provenance, camera motion, physical sites,',
        'annotation-time mapping, historical exposure and approved roles remain unresolved.',
        'A raw-XML structural screen is implemented and tested on synthetic fixtures only.',
        'It is not evidence that actual DroneCrowd XML is readable or causally suitable.',
        'No metric/seconds, deployment, independent-generalization or CVPR-readiness claim.',
        'Stage5C/SMC remain off.', '',
        f'[Official source folder]({FOLDER_URL}). File IDs and SHA256 are in source_manifest.json.', '',
    ])
