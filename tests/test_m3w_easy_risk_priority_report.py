import pytest
from scripts.report_m3w_easy_risk_priority_training import training_summary


def evidence():
    terms = ('marginal', 'occurrence', 'conditional', 'supervised')
    rows = []
    for group in range(108):
        for arm in ('uncapped', 'risk_priority'):
            rows.append(dict(group=str(group), arm=arm, step=2000, sample_hash=str(group),
                query_draws=64000, row_draws=128000, unknown_rows_sampled=0,
                first_monitor={k: 1. for k in terms},
                last_monitor={k: .5 if arm == 'uncapped' else .25 for k in terms},
                cap_updates=1000 if arm == 'risk_priority' else None,
                mean_alpha=.5 if arm == 'risk_priority' else 1., min_risk_projection=.6))
    return dict(groups=108, heads=216, fitting_summary=rows, updates_per_head=2000,
                control_parent_states_exact=108)


def test_training_count_and_matched_monitor():
    receipt = evidence(); report = training_summary(receipt)
    assert report['total_updates'] == 432000
    assert report['repaired_lower_final_direct_risk_heads'] == 108
    assert report['cap_active_updates'] == 108000
    receipt['fitting_summary'][1]['sample_hash'] = 'mismatch'
    with pytest.raises(AssertionError):
        training_summary(receipt)


def test_rejects_unknown_sampling_or_missing_parent_reproduction():
    receipt = evidence(); receipt['fitting_summary'][1]['unknown_rows_sampled'] = 1
    with pytest.raises(AssertionError):
        training_summary(receipt)
    receipt = evidence(); receipt['control_parent_states_exact'] = 107
    with pytest.raises(AssertionError):
        training_summary(receipt)
