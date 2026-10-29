import copy
import numpy as np
import pytest

from scripts.verify_m3w_risk_decoupling import verify
from src.evaluation import m3w_risk_decoupling_readout as readout
from src.world_model import m3w_risk_decoupling as train


def sample(monkeypatch, selected=True):
    monkeypatch.setattr(train.api.parent.previous.prior, 'eligible', lambda a: np.ones(3, bool))
    y = np.array([[.2, 0., 1., 1., 0.], [0., .05, 1., 1., .05], [np.nan]*5])
    p = np.array([[.3, .001, 1., 1., .001]]*3)
    if not selected:
        p[:, [1, 4]] = .5
    args = (np.zeros((3, 2)), np.ones(3), y, None, None, np.array(['a']*3), np.arange(3), {'scale': 1.})
    result = readout.evaluate(args, {a: p.copy() for a in readout.ARMS})
    rows = [dict(context=f'c{c}', source=f's{s}', seed=seed, result=copy.deepcopy(result),
                 scope='fit', held_inference=False, independent_roles_read=False)
            for c in range(6) for seed in (17, 29, 43) for s in range(4)]
    expected = {(r['context'], r['source'], r['seed']) for r in rows}
    return rows, expected


def test_receipt_reconstruction_preserves_unknowns_and_strict_gate(monkeypatch):
    rows, expected = sample(monkeypatch)
    summary = readout.summarize(rows, expected)
    actual = verify(rows, summary, expected)
    assert actual['checks']['cohort_target_checks'] == 864
    assert actual['checks']['policy_checks'] == 216
    assert not actual['fit_eligible_for_later_held_design']
    assert summary['cohort_counts']['origin']['origin']['unknown'] == 72


def test_receipt_reconstruction_rejects_missing_tampered_and_undefined(monkeypatch):
    rows, expected = sample(monkeypatch)
    summary = readout.summarize(rows, expected)
    with pytest.raises(ValueError): verify(rows[:-1], summary, expected)
    broken = copy.deepcopy(summary); broken['easy_pair_wins'] = 12
    with pytest.raises(ValueError): verify(rows, broken, expected)
    rows[0]['result']['cohorts']['estimation']['origin']['easy']['MSE'] = None
    with pytest.raises(ValueError): verify(rows, readout.summarize(rows, expected), expected)


def test_empty_original_cohort_is_undefined_and_cannot_pass(monkeypatch):
    rows, expected = sample(monkeypatch, selected=False)
    summary = readout.summarize(rows, expected)
    result = verify(rows, summary, expected)
    assert summary['mean_view_MSE']['estimation']['origin']['easy'] is None
    assert not result['fit_eligible_for_later_held_design']
