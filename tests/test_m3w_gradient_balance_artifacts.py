from pathlib import Path
from scripts import report_m3w_gradient_balanced_admission as report
from scripts import run_m3w_gradient_balanced_admission as run
from src.evaluation.m3w_metric_bundle import verify


def test_fresh18_and_cached36_are_distinguished_and_all_groups_reconstruct_exactly():
    value = report.build()
    assert len(value['models']) == 54 and len(value['traces']) == 18
    assert value['training']['fresh_primary_optimizer_updates'] == 36000
    assert len(value['training']['cached_verified_controls']) == 36
    assert len(value['groups']) == 72
    assert all(t[-1]['step'] == 2000 for t in value['traces'].values())
    actual = [(v['context'], v['source'], v['seed']) for v in value['groups']]
    assert len(set(actual)) == 72
    assert all(Path(r['path']).name == f'{v["context"]}_{v["source"]}_seed{v["seed"]}_fit.json'
        for r, v in zip(value['complete']['groups'], value['groups']))


def test_improved_support_cannot_replace_failed_primary_or_complete_risk():
    value = report.build(); s = value['summary']
    assert s['accuracy']['balanced_score']['point'] > 0
    assert s['comparisons']['balanced_minus_admitted']['full_known_pp']['CI95'][1] < 0
    assert s['safety']['balanced_score']['supported_record_views'] == 201
    assert s['safety']['balanced_score']['known_easy_risk_violations'] == 31
    assert s['safety']['balanced_score']['undefined_easy_record_views'] == 72
    assert not s['fitting_advancement_gate'] and not s['independent_confirmation']
    assert not s['deployment_changed'] and not s['submission_ready']


def test_actual_balancing_cap_and_zero_steps_are_not_hidden_as_exact_parity():
    value = report.build(); p = value['pressure']
    assert p['total_steps'] == 36000 and p['zero_scale_steps'] == 28464
    assert p['capped_steps'] == 6163 and p['nonzero_snapshots'] == 163
    assert p['nonzero_weighted_ratio_quantiles'][2] < .2
    assert p['maximum_scale'] == 1e6
    assert value['failure_conditions']['easy_completion_upper_fails_despite_known_budget'] == 135
    assert value['cached_oracle']['positive_record_views'] == 148
    assert value['cached_oracle']['label_aware_not_deployable']


def test_plain_markdown_and_exact_aggregate_only_bundle():
    value = report.build()
    text = (run.PUBLIC / 'readout.md').read_text()
    assert text.startswith('# Capped norm-parity') and text == report.render(value)
    result = verify(run.ROOT, run.PUBLIC)
    assert result['groups'] == 72 and result['bytes_verified']
    assert not result['raw_data_or_weights_included']
