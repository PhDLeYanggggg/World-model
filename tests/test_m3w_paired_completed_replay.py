import json
import pytest
from scripts import replay_m3w_moving_paired_decision as replay


def test_completed_replay_does_not_create_or_overwrite_receipts(tmp_path):
    path = tmp_path/'missing'/'receipt.json'
    with pytest.raises(ValueError, match='Completed receipt required'):
        replay.readonly_receipt(path, {'value': 1})
    assert not path.parent.exists()
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps({'value': 1}))
    before = replay.manifest(tmp_path)
    replay.readonly_receipt(path, {'value': 1})
    with pytest.raises(ValueError, match='disagrees'):
        replay.readonly_receipt(path, {'value': 2})
    assert replay.manifest(tmp_path) == before


def test_incomplete_experiment_cannot_use_completed_replay(tmp_path):
    with pytest.raises(ValueError, match='Complete original run required'):
        replay.require_completed(tmp_path)
    assert list(tmp_path.iterdir()) == []
