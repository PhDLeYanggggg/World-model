import json

import pytest

from scripts.audit_m3w_dronecrowd_metadata import acquire
from scripts.verify_m3w_dronecrowd_metadata import check
from src.evaluation.m3w_dronecrowd_intake import audit_release, sha
from tests.test_m3w_dronecrowd_intake import metadata_fixture


def prepare(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    files, _ = acquire(source, output, metadata_fixture().__getitem__)
    analysis = audit_release(files)
    analysis['source_manifest_sha256'] = sha((output / 'source_manifest.json').read_bytes())
    (output / 'analysis.json').write_text(json.dumps(analysis))
    return source, output


def test_separate_counting_and_no_admission(tmp_path):
    source, output = prepare(tmp_path)
    result = check(source, output)
    assert result['counts']['union_sequences'] == 112
    assert result['admitted_recordings'] == 0
    assert result['actual_xml_screening'] == 'not_run'


def test_separate_verifier_detects_wrong_reduction(tmp_path):
    source, output = prepare(tmp_path)
    path = output / 'analysis.json'
    analysis = json.loads(path.read_text())
    analysis['counts']['train_sequences'] = 83
    path.write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match='recount disagrees'):
        check(source, output)


def test_separate_verifier_refuses_modified_source(tmp_path):
    source, output = prepare(tmp_path)
    (source / 'trainlist.txt').write_text('00001\n')
    with pytest.raises(ValueError, match='bytes mismatch'):
        check(source, output)
