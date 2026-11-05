import json
import pytest
from scripts import replay_m3w_neighbor_order as replay


def test_replay_keeps_original_runtime_and_requires_exact_scientific_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.run, 'PUBLIC', tmp_path)
    execution = tmp_path/'verify_execution.json'; execution.write_text('{"pid":1}')
    replay.receipt(execution, {'pid': 2, 'seconds': 5., 'peak_RSS_bytes': 200})
    assert execution.read_text() == '{"pid":1}'
    analysis = tmp_path/'analysis.json'; analysis.write_text(json.dumps({'result': False}))
    replay.receipt(analysis, {'result': False})
    with pytest.raises(ValueError, match='disagrees'):
        replay.receipt(analysis, {'result': True})
    with pytest.raises(ValueError):
        replay.receipt(tmp_path/'missing.json', {})
