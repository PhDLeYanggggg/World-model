import json

import numpy as np
import pytest

from scripts import run_m3w_european_cross_moment as run


def bank(tmp_path, mode):
    root = tmp_path/'private'/mode; root.mkdir(parents=True, exist_ok=True)
    public = tmp_path/'public'/mode; public.mkdir(parents=True, exist_ok=True)
    identity = {'mode': mode}
    (root/'identity.json').write_text(json.dumps(identity))
    archive = root/'choices.npz'
    np.savez(archive, ids=np.array([1, 2]), hurdle_original=np.array([True, False]))
    records = [dict(group=str(i), path=str(archive.relative_to(tmp_path)), sha256=run.digest(archive)) for i in range(36)]
    (root/'decisions_complete.json').write_text(json.dumps(dict(identity=identity, archives=records)))
    (public/'head_replay.json').write_text(json.dumps(dict(identity=identity, all_passed=True, checks=[{}]*36)))
    return identity, records


def configure_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'PRIVATE_ROOT', tmp_path/'private')
    monkeypatch.setattr(run, 'PUBLIC_ROOT', tmp_path/'public')
    monkeypatch.setattr(run, 'assert_identity', lambda _: None)


def test_evaluation_barrier_requires_both_complete_decision_banks(monkeypatch, tmp_path):
    configure_fixture(monkeypatch, tmp_path)
    bank(tmp_path, 'batch')
    with pytest.raises(FileNotFoundError):
        run.ensure_all_decisions()
    bank(tmp_path, 'fitting')
    run.ensure_all_decisions()


@pytest.mark.parametrize('failure', ['duplicate', 'hash'])
def test_decision_bank_rejects_duplicates_and_changed_artifacts(monkeypatch, tmp_path, failure):
    configure_fixture(monkeypatch, tmp_path)
    identity, records = bank(tmp_path, 'batch')
    if failure == 'duplicate':
        records[-1] = records[0]
    else:
        records[0]['sha256'] = '0'*64
    (tmp_path/'private/batch/decisions_complete.json').write_text(json.dumps(dict(identity=identity, archives=records)))
    with pytest.raises(ValueError):
        run.read_decisions(identity)


def test_fixed_scale_decision_control_does_not_require_batch_outcome_metrics(monkeypatch, tmp_path):
    configure_fixture(monkeypatch, tmp_path)
    identity, _ = bank(tmp_path, 'batch')
    current = dict(mode='fitting', control_identity=identity)
    value = run.expected_control_choice('0', current)
    assert value == run.array_hash(np.array([1, 2]), np.array([True, False]))
    assert not (tmp_path/'public/batch/analysis.json').exists()


def test_mode_is_required_and_unknown_modes_rejected():
    with pytest.raises(ValueError):
        run.configure('best_after_readout')
