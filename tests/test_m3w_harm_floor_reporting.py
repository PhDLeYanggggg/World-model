from copy import deepcopy
import pytest
from scripts.report_m3w_moving_harm_floor import verify_roster
from scripts import run_m3w_moving_harm_floor as run


def test_reporting_requires_all_three_arms_in_all_contexts():
    jobs = [dict(key='critic'+str(i)) for i in range(3)]
    rows = [dict(job=j['key'], arm=a, step=2000, parameters=64260, both_neural_encoders_updated=True)
        for j in jobs for a in ('joint', 'uniform_harm', 'harm_floor')]
    verify_roster(rows, jobs)
    with pytest.raises(ValueError):
        verify_roster(rows[:-1], jobs)
    duplicate = deepcopy(rows); duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError):
        verify_roster(duplicate, jobs)
    partial = deepcopy(rows); partial[0]['step'] = 1999
    with pytest.raises(ValueError):
        verify_roster(partial, jobs)


def test_weight_receipt_serializes_without_changing_checkpoint_array():
    counts = run.source.np.array([2, 3, 5], dtype=run.source.np.int64)
    original = dict(planned_query_counts=counts, uniform_multiplier=1.5)
    encoded = run.source.stable(run.json_weighting(original))
    assert encoded == dict(planned_query_counts=[2, 3, 5], uniform_multiplier=1.5)
    assert original['planned_query_counts'] is counts


def test_amendment_only_accepts_original_runner_binding_change(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'REG', tmp_path/'registration.json')
    monkeypatch.setattr(run, 'AMENDMENT', tmp_path/'amendment.json')
    monkeypatch.setattr(run.source, 'artifact', lambda p: dict(path=p.name, sha256=run.source.sha(p)))
    runner = str(run.Path(run.__file__).relative_to(run.ROOT))
    original = dict(bindings={runner: run.ORIGINAL_RUNNER_SHA, 'model': 'same'}, threshold=.02)
    run.registered_payload(original)
    digest = run.source.sha(run.REG)
    amended = deepcopy(original); amended['bindings'][runner] = 'new_reporting_only_runner'
    run.registered_payload(amended)
    run.registered_payload(amended)
    assert run.source.sha(run.REG) == digest
    assert run.source.read(run.AMENDMENT)['training_objective_changed'] is False
    altered = deepcopy(amended); altered['threshold'] = .03
    with pytest.raises((AssertionError, ValueError)):
        run.registered_payload(altered)
    altered = deepcopy(amended); altered['bindings']['model'] = 'changed'
    with pytest.raises((AssertionError, ValueError)):
        run.registered_payload(altered)


def test_fitting_figures_are_reproducible_and_use_only_supplied_receipts(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    from scripts import report_m3w_moving_harm_floor as report
    public, private = tmp_path/'public', tmp_path/'private'
    public.mkdir(); private.mkdir()
    monkeypatch.setattr(report, 'PUBLIC', public)
    monkeypatch.setattr(report.run, 'PRIVATE', private)
    rows = [dict(job='synthetic_critic'+str(i), arm=arm,
        trace=[dict(step=1, risk_MSE=.1), dict(step=2000, risk_MSE=.05)],
        joint_attribution=dict(total=dict(false_safe_cost=.01, false_reject_cost=.02, total_joint_cost=.03)))
        for i in range(3) for arm in run.api.ARMS]
    report.figures(rows)
    originals = {p.name: p.read_bytes() for p in (public/'figures').glob('*.svg')}
    assert set(originals) == {'common_risk_loss.svg', 'decision_costs.svg'}
    for name in originals:
        ET.parse(public/'figures'/name)
    report.figures(rows)
    assert originals == {p.name: p.read_bytes() for p in (public/'figures').glob('*.svg')}
