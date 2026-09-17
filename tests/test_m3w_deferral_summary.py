from copy import deepcopy
import json

import pytest

from scripts import summarize_m3w_registered_deferral as module


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path/'parent.json').write_text(json.dumps({'seeds': [17, 29, 43]}))
    registration = {'parent_protocol': 'parent.json', 'families': {'skip': 'a', 'bounded': 'b'},
                    'variants': {v: {'fit_settings': {'steps': 1000}} for v in ('linear_bound1', 'linear_bound10', 'mlp_bound1', 'mlp_bound10')}}
    score = {'selected_error': 1., 'baseline_error': 1., 'improvement_pct': 0., 'degradation_fraction': 0.,
             'mean_positive_harm': 0., 'count': 20, 'bootstrap': {'status': 'not_run_insufficient_physical_scenes'}}
    arm = {'all': score, 'easy': score, 'hard': score, 'secondary_fde': score, 'switch_rate_all_past_supported': 0.}
    summary = {'arms': {a: deepcopy(arm) for a in ['floor', 'uncontrolled', 'independent', 'scene_uniform', 'joint',
                *['deferral_'+v for v in registration['variants']]]}, 'physical_scenes': 1,
               'paired_comparisons': {}}
    result = {'fits': {v: {'training_complete': True, 'steps': 1000} for v in registration['variants']},
              'summaries': {str(i): deepcopy(summary) for i in range(4)}, 'OOF_features_exactly_matched': True,
              'fresh_forecast_scoring_exactly_matches_parent': True}
    report = {'selected_variant': None, 'results': {f: {str(s): deepcopy(result) for s in (17, 29, 43)}
                                                  for f in registration['families']}}
    return registration, report


def test_keeps_all_twenty_four_heads_and_all_parent_controls(tmp_path, monkeypatch):
    registration, report = fixture(tmp_path, monkeypatch)
    rows, controls, _ = module.collect(report, registration)
    assert len(rows) == 24 and len(controls) == 120
    assert not any(r['positive_primary_gain'] for r in rows)
    assert all(r['easy_pass'] for r in rows)


@pytest.mark.parametrize('mutation', ['seed', 'variant', 'budget', 'features', 'forecast', 'policy', 'conditioned_deferrer'])
def test_incomplete_or_unmatched_results_not_summarized(tmp_path, monkeypatch, mutation):
    registration, report = fixture(tmp_path, monkeypatch)
    item = report['results']['skip']['17']
    if mutation == 'seed':
        del report['results']['skip']['43']
    elif mutation == 'variant':
        del item['fits']['mlp_bound1']
    elif mutation == 'budget':
        item['fits']['mlp_bound1']['steps'] = 20
    elif mutation == 'features':
        item['OOF_features_exactly_matched'] = False
    elif mutation == 'forecast':
        item['fresh_forecast_scoring_exactly_matches_parent'] = False
    elif mutation == 'policy':
        del item['summaries']['3']
    else:
        item['summaries']['3']['arms']['deferral_mlp_bound1']['all']['improvement_pct'] = 10.
    with pytest.raises(ValueError):
        module.collect(report, registration)
