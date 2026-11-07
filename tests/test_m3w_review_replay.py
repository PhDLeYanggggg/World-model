import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.build_m3w_review_replay import RISK, trim, write_zip
from src.reproduction.m3w_review_replay import verify_manifest


def test_archive_reproducible_and_manifest_rejects_mutation(tmp_path):
    data = b'aggregate data'
    files = {'statistics.json': data, 'manifest.json': json.dumps(
        {'statistics.json': hashlib.sha256(data).hexdigest()}).encode()}
    a, b = tmp_path/'a.zip', tmp_path/'b.zip'
    write_zip(a, files)
    write_zip(b, dict(reversed(list(files.items()))))
    assert a.read_bytes() == b.read_bytes()
    root = tmp_path/'extracted'
    with ZipFile(a) as z:
        z.extractall(root)
    verify_manifest(root)
    (root/'statistics.json').write_bytes(data+b'!')
    with pytest.raises(ValueError, match='checksum'):
        verify_manifest(root)


def test_export_keeps_risk_row_denominator_but_no_original_names():
    risk = dict.fromkeys(RISK, None)
    risk.update(rows=5, selected=1, supported=False, unknown_selected=1)
    rows = [dict(recording='raw_video', source='raw_source', seed=17, comparisons={},
                 metrics={'test': dict(rows=5, p95_known_ADE_pixel=3., risk=risk)})]
    output = trim(rows, {'raw_video': 'V001'}, {'raw_source': 'S01'}, {17: 1})
    assert output[0]['metrics']['test']['risk']['rows'] == 5
    assert 'supported' not in output[0]['metrics']['test']['risk']
    assert 'raw_' not in json.dumps(output)
