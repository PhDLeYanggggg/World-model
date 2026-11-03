import hashlib
import json

import pytest

from scripts.backup_m3w_duration_checkpoints_create import require_completed


def test_completed_backup_refuses_missing_or_partial_fitting_replay(tmp_path):
    with pytest.raises(ValueError, match='Complete18'):
        require_completed(tmp_path)
    (tmp_path/'summary.json').write_text(json.dumps({'models': 6}))
    (tmp_path/'verification.json').write_text(json.dumps({'all18_fitting_decompositions_exact': True}))
    with pytest.raises(ValueError, match='partial or stale'):
        require_completed(tmp_path)


def test_completed_backup_binds_full_result_hash_and_exact_replay(tmp_path):
    summary = tmp_path/'summary.json'
    summary.write_text(json.dumps({'models':18, 'all18_old_budget_prefixes_exact':True}))
    verification = tmp_path/'verification.json'
    verification.write_text(json.dumps({'all18_fitting_decompositions_exact':True,
        'summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest()}))
    require_completed(tmp_path)
    summary.write_text(summary.read_text()+' ')
    with pytest.raises(ValueError, match='partial or stale'):
        require_completed(tmp_path)
