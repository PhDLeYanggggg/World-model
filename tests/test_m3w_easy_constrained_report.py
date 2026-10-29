import copy
import numpy as np
import pytest

from scripts import report_m3w_easy_constrained as report
from tests.test_m3w_local_cost_analogue import inputs


@pytest.fixture(scope='module')
def row():
    run = report.run
    bank, query, frozen = inputs()
    bank['labels'][:, 4] = bank['labels'][:, 1]
    bank['labels'][2] = np.nan
    out = run.api.retrieve(bank, query, frozen, k=4)
    pred = {'origin': frozen[query].copy()}
    full = {'origin': frozen.copy()}
    for arm in run.api.MODES:
        pred[arm] = pred['origin'].copy()
        pred[arm][:, [1, 4]] = out['predictions'][arm]
        full[arm] = frozen.copy()
        full[arm][query] = pred[arm]
    args = (bank['x'], bank['envelope'], bank['labels'], None, None,
            bank['recordings'], np.arange(len(query)), {})
    policy, _ = run.policy.evaluate(args, full, bank['eligible'])
    scores = {a: run.api.base.score(bank['labels'][query], p,
              bank['envelope'][query], 2., np.ones(query.sum(), bool))
              for a, p in pred.items()}
    diagnostics = run.diagnostic(out, np.ones(query.sum(), bool),
                                bank['labels'][query], pred, 2.)
    return dict(context='synthetic', source='a', seed=0, scores=scores,
                diagnostics=diagnostics, policy=policy)


def test_report_checks_complete_arithmetic_and_preserves_unknown_risk(row):
    report.validate_row(row)
    assert row['scores']['origin']['unknown'] == 1
    for matched in row['policy']['matched'].values():
        assert matched['empty']
        assert matched['right']['easy_selected_risk_upper'] is None
    summary = dict(policy_readout=report.run.policy.summarize([row]),
                   equal_view_MSE={a: {t: s[t]['full_cohort_MSE']
                       for t in ('all', 'easy')} for a, s in row['scores'].items()})
    data, text = report.describe([row], summary)
    assert data['donor']['local_supported'] == row['scores']['origin']['rows']
    assert 'not estimable' in text
    assert not summary['policy_readout']['primary_full_roster_budget_supported']


@pytest.mark.parametrize('fault', ['empty_risk', 'cohort', 'partition', 'policy_mass'])
def test_report_rejects_inconsistent_policy_or_diagnostic_receipts(row, fault):
    changed = copy.deepcopy(row)
    if fault == 'empty_risk':
        changed['policy']['matched']['origin']['right']['easy_selected_risk_upper'] = 0.
    elif fault == 'cohort':
        changed['scores']['easy_constrained']['unknown'] += 1
    elif fault == 'partition':
        changed['diagnostics']['strata']['alpha_zero']['rows'] += 1
    else:
        changed['policy']['policies']['origin']['selected_known_harm_mass'] += 1.
    with pytest.raises((ValueError, AssertionError)):
        report.validate_row(changed)
