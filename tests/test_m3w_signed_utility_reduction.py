import copy
import itertools
import json
import numpy as np
import pytest
from scripts import run_m3w_signed_utility_allocation as run
from tests.test_m3w_active_bank_readout import fixture


def test_support_donor_example_also_respects_conditional_moment_bounds():
    p = np.zeros((2, 7, 5)); ok = np.zeros((2, 7), bool); ok[:, 0] = True
    p[0, 0] = [10., .04, 20., .1, .04]
    p[1, 0] = [0., .01, 4., 3., 0.]
    assert (p[..., 0] <= p[..., 2]).all() and (p[..., 3] <= p[..., 2]).all()
    assert (p[..., 4] <= p[..., 1]).all()
    c, info = run.api.solve(p, ok, np.full(2, -1), exact_count=None, seconds=2.)
    assert c.tolist() == [0, 0] and info['optimal']
    assert run.api.base.terms(p, c)[0] == pytest.approx(9.95)
    assert not run.api.base.feasible(p, ok, c) and run.api.feasible(p, ok, c)


def test_nonzero_matched_count_matches_exhaustive_restricted_optima():
    rng = np.random.default_rng(502)
    for _ in range(50):
        p = np.zeros((3, 7, 5)); p[:, :2, :2] = rng.uniform(0, 2, (3, 2, 2))
        p[:, :2, 2:4] = rng.uniform(.1, 60, (3, 2, 2))
        p[:, :2, 4] = p[:, :2, 1]*rng.uniform(0, 1, (3, 2))
        ok = np.zeros((3, 7), bool); ok[:, :2] = True
        candidates = [np.array(c) for c in itertools.product((-1, 0, 1), repeat=3)]
        anchors = [c for c in candidates if run.api.base.feasible(p, ok, c) and (c >= 0).any()]
        if not anchors: continue
        anchor = anchors[0]; k = int((anchor >= 0).sum())
        choice, info = run.api.solve(p, ok, anchor, exact_count=k, seconds=2.)
        scores = [run.api.base.terms(p, c)[0] for c in candidates
            if int((c >= 0).sum()) == k and run.api.feasible(p, ok, c)]
        assert int((choice >= 0).sum()) == k and info['optimal']
        assert run.api.base.terms(p, choice)[0] == pytest.approx(max(scores), abs=1e-10)


def test_support_donor_can_help_at_the_same_nonzero_intervention_count():
    p = np.zeros((4, 7, 5)); ok = np.zeros((4, 7), bool); ok[:, 0] = True
    p[0, 0] = [10., .04, 20., .1, .04]; p[1, 0] = [0., .01, 4., 3., 0.]
    p[2:, 0] = [1., 0., 2., 1., 0.]
    anchor = np.array([-1, -1, 0, 0])
    assert run.api.base.feasible(p, ok, anchor)
    old, _ = run.api.base.solve_frame(p, ok, anchor, exact_count=2, seconds=2.)
    new, info = run.api.solve(p, ok, anchor, exact_count=2, seconds=2.)
    assert info['optimal'] and int((new >= 0).sum()) == int((old >= 0).sum()) == 2
    assert run.api.base.terms(p, old)[0] == 2.
    assert run.api.base.terms(p, new)[0] == pytest.approx(9.95)


def synthetic(monkeypatch):
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    class Registration:
        def read_text(self): return json.dumps(dict(contexts=contexts))
    monkeypatch.setattr(run, 'REG', Registration()); rows = []
    for c, sources in contexts.items():
        for s in sources:
            packet = fixture(s, True)[0]['new_neural']
            choices = {a: np.full(2, 6) for a in run.api.NAMES}
            r = run.readout(packet, choices, dict(matched_count_every_query=True, nonoptimal={}), 1.5)
            for seed in (17, 29, 43):
                rows.append(dict(context=c, source=s, seed=seed, result=copy.deepcopy(r),
                    original_control_exact=True, original_oracle_exact=True))
    return rows, dict(seeds=[17, 29, 43], bootstrap_seed=3, bootstrap_resamples=32)


def test_positive_known_gain_does_not_bypass_full_unknown_risk(monkeypatch):
    rows, cfg = synthetic(monkeypatch); result = run.summarize(rows, cfg)
    assert result['accuracy']['signed_matched']['point'] > 0
    assert result['safety']['signed_matched']['selected_unknown'] > 0
    assert not result['fitting_gates']['complete_original_risk'] and not result['fitting_advancement_gate']
    assert not result['independent_confirmation'] and not result['submission_ready']
    assert not result['deployment_changed'] and not result['stage5c_executed'] and not result['smc_enabled']
    assert result['new_training_updates'] == 0


def test_reducer_refuses_missing_roster_and_seed_dependent_oracles(monkeypatch):
    rows, cfg = synthetic(monkeypatch)
    with pytest.raises(ValueError): run.summarize(rows[:-1], cfg)
    rows[1]['result']['oracle_admissibility']['r0'] = dict(full={'positive_feasible': False})
    with pytest.raises(ValueError): run.summarize(rows, cfg)
