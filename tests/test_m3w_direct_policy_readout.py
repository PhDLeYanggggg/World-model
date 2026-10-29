import copy
import numpy as np
import pytest
from src.evaluation import m3w_direct_policy_readout as api
from scripts.report_m3w_shared_train_risk import check_policy


def result():
    y = np.tile([.1, 0., 1., 1., 0.], (4, 1))
    args = (np.zeros((4, 380)), np.ones(4), y, None, None,
            np.array(['a', 'a', 'b', 'b']), np.array([0, 1, 0, 1]), {})
    masks = {a: np.ones(4, bool) for a in api.POLICIES}
    masks['floor'][:] = False; masks['origin'][:] = False
    ranks = {a: np.arange(4, dtype=float) for a in api.POLICIES}
    return api.evaluate(args, masks, ranks)


def test_full_positive_utility_is_separate_from_empty_matched_risk():
    r = result(); c = r['comparisons']['hard_record_minus_origin']
    assert c['full']['lower_pp'] > 0 and c['matched']['lower_pp'] == 0
    assert c['matched_empty'] and c['matched_right']['easy_selected_risk_upper'] is None
    assert c['matched_risk_role'] == 'not_a_safety_gate'
    for p in r['policies'].values(): check_policy(p)
    for p in (c['matched_left'], c['matched_right']): check_policy(p)


def test_summary_requires_both_complete_cohorts_and_retains_floor():
    r = result()
    rows = [dict(context=str(c), source=str(s), seed=k, scope=scope, result=copy.deepcopy(r))
            for scope in ('fit', 'inner_held') for c in range(6) for s in range(4) for k in range(3)]
    out = api.summarize(rows)
    assert out['scopes']['inner_held']['safety']['floor']['undefined_easy_views'] == 72
    assert out['scopes']['inner_held']['safety']['hard_record']['budget_record_views'] == 144
    assert not out['independent_confirmation'] and not out['risk_MSE_gate']
    with pytest.raises(ValueError): api.summarize(rows[:-1])


def test_runner_imports_fixed_api_and_resolves_source_roster():
    from scripts import run_m3w_direct_policy as run
    assert run.np is np and run.readout.PAIRS == api.PAIRS
    assert hasattr(run.LOADER, 'packets') and hasattr(run.ORIGINS, 'models')


def test_prior_array_hash_codec_is_used_at_artifact_boundary(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_direct_policy as run
    from tests.test_m3w_direct_policy import fixture
    packets, predictions = fixture()
    monkeypatch.setattr(run.previous, 'PUBLIC', tmp_path)
    groups = tmp_path/'groups'; groups.mkdir()
    refs = []
    for source, args in packets.items():
        action = run.api.actions(predictions[source], run.api.shared.eligible(args))
        legacy_hash = run.previous.api.base.fingerprint
        assert legacy_hash(action) != run.api.fingerprint(action)
        doc = dict(action_sha256={'origin': legacy_hash(action)},
                   prediction_sha256={'origin': legacy_hash(predictions[source][action])})
        path = groups/f'context_{source}_seed17.json'
        path.write_text(json.dumps(doc))
        refs.append(dict(path=str(path), sha256=run.sha(path)))
    (tmp_path/'complete.json').write_text(json.dumps(dict(groups=refs)))
    run.verify_original('context', 17, packets, predictions)
