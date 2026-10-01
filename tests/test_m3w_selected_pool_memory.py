import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import report_m3w_selected_pool_memory as memory


def synthetic_rows():
    rows = []
    y = np.array([[1., 0., 2., 2., 0.], [0., 1., 2., 2., 1.], [np.nan]*5])
    p = np.ones((3, 5)); env = np.ones(3)
    for group in range(24):
        for seed in (17, 29, 43):
            for role in ('source_oof', 'source_resubstitution', 'transfer'):
                for target in range(3 if role == 'transfer' else 1):
                    for mode in ('harm', 'reference', 'joint'):
                        kept = np.array([True, False, True]) if group % 2 else np.zeros(3, bool)
                        s = memory.run.api.account(y, p, p, env, np.ones(3, bool), kept, audit=True)
                        rows.append(dict(group=f'g{group}', view=f'g{group}_s{seed}_{role}_{target}',
                            head_seed=seed, role=role, mode=mode, site=f'site{(group+target)%12}',
                            source_screen=True, statistics=s,
                            supported_pool_bias={k: dict(shift=s[k+'_contrast']['signed_bias_shift']) for k in ('all','easy')},
                            recordings=[dict(recording=f'rec{group}')]))
    return rows


@pytest.fixture
def report_inputs():
    rows = synthetic_rows()
    cfg = dict(bootstrap_resamples=30, bootstrap_seed=20261001)
    public = {name: json.dumps({key: True}).encode() for name, key in
              (('source_replay.json', 'all72_exact_replay'), ('transfer_replay.json', 'all216_exact_replay'))}
    return rows, cfg, public


def test_original_disk_and_memory_numerics_are_exact(report_inputs, tmp_path, monkeypatch):
    rows, cfg, public = report_inputs
    # The real scoped subprocess is tested by the actual report; this fixture isolates I/O parity.
    calls = []
    def test_command(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=0, stdout='synthetic subprocess fixture\n', stderr='')
    monkeypatch.setattr(memory.run.subprocess, 'run', test_command)
    frozen_public = memory.run.PUBLIC
    frozen_reader = memory.run.read_rows
    in_memory = memory.execute_original_report(rows, cfg, {'bindings': {}}, public, render_findings=False)
    assert memory.run.PUBLIC == frozen_public and memory.run.read_rows is frozen_reader
    for name, data in public.items(): (tmp_path/name).write_bytes(data)
    monkeypatch.setattr(memory.run, 'PUBLIC', tmp_path)
    monkeypatch.setattr(memory.run, 'read_rows', lambda: rows)
    memory.run.report(cfg, {'bindings': {}})
    assert json.loads((tmp_path/'summary.json').read_text()) == json.loads(in_memory['summary.json'])
    assert json.loads((tmp_path/'readout.json').read_text()) == json.loads(in_memory['readout.json'])
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0][1:3] == ['-m', 'pytest']
    assert json.loads(in_memory['summary.json'])['groups']['joint_transfer']['easy']['bias_shift']['all_views']['mean'] is None


def test_memory_storage_is_bounded_flat_and_immutable():
    root = memory.MemoryPath({}, cap=30)
    memory.immutable(root/'x.json', {'a': 1})
    memory.immutable(root/'x.json', {'a': 1})
    with pytest.raises(ValueError): memory.immutable(root/'x.json', {'a': 2})
    with pytest.raises(ValueError): (root/'large').write_text('x'*31)
    for name in ('../escape', '/absolute', '', '.'):
        with pytest.raises(ValueError): root/name
    assert set(root.files) == {'x.json'}


def test_original_findings_renderer_uses_verified_memory_artifacts(report_inputs, monkeypatch):
    rows, cfg, public = report_inputs
    monkeypatch.setattr(memory.run.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=0, stdout='synthetic subprocess fixture\n', stderr=''))
    files = memory.execute_original_report(rows, cfg, {'bindings': {}}, public)
    receipt = json.loads(files['findings_verification.json'])
    assert receipt['source_screened_joint_counts']['views'] == 216
    assert receipt['report_sha256'] == memory.digest(memory.MemoryPath(files)/'results.md')
    assert receipt['deployment_changed'] is False


def test_missing_or_duplicate_rows_rejected(report_inputs):
    rows, cfg, public = report_inputs
    with pytest.raises(AssertionError): memory.execute_original_report(rows[:-1], cfg, {}, public)
    bad = copy.deepcopy(rows); bad[-1] = bad[0]
    with pytest.raises(AssertionError): memory.execute_original_report(bad, cfg, {}, public)


def test_failed_scoped_tests_stop_report(report_inputs, monkeypatch):
    rows, cfg, public = report_inputs
    monkeypatch.setattr(memory.run.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1,stdout='failed',stderr=''))
    with pytest.raises(RuntimeError, match='failed'):
        memory.execute_original_report(rows, cfg, {}, public)


def test_missing_full_replay_stops_report(report_inputs):
    rows, cfg, public = report_inputs
    public['transfer_replay.json'] = b'{"all216_exact_replay":false}'
    with pytest.raises(AssertionError): memory.execute_original_report(rows, cfg, {}, public)
