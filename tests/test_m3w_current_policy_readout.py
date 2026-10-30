import copy
import json
import pytest
from scripts import run_m3w_current_policy_critic as run
from tests.test_m3w_selected_query_critic_readout import group


def fixture(monkeypatch):
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    class Registration:
        def read_text(self): return json.dumps(dict(contexts=contexts))
    monkeypatch.setattr(run, 'REG', Registration())
    values = []
    for c, sources in contexts.items():
        for s in sources:
            for seed in (17, 29, 43):
                r = copy.deepcopy(group(s, True))
                for arm in ('row', 'set'):
                    r['policies']['fixed_query_'+arm+'_joint'] = copy.deepcopy(r['policies'][arm+'_joint'])
                    for record in r['recordings'].values():
                        record['fixed_query_'+arm+'_joint'] = copy.deepcopy(record[arm+'_joint'])
                r['current_query_contrasts'] = {arm: copy.deepcopy(r['comparisons']['set_joint_minus_parent_joint']) for arm in ('row', 'set')}
                values.append(dict(context=c, source=s, seed=seed, result=r,
                    parent_controls_exact=True, fixed_query_controls_exact=True, paired_training_verified=True))
    return values, dict(seeds=[17, 29, 43], bootstrap_seed=3, bootstrap_resamples=32)


def test_current_query_known_gain_never_bypasses_missing_support_or_risk(monkeypatch):
    values, cfg = fixture(monkeypatch); result = run.summarize(values, cfg)
    assert result['accuracy']['set_joint']['point'] > 0
    assert result['cached_verified_fixed_query_heads'] == 36
    assert result['primary'] == 'current_query_set_joint_minus_fixed_original_query_set_joint'
    assert not result['fitting_advancement_gate'] and not result['fitting_gates']['complete_recording_risk']
    assert not result['deployment_changed'] and not result['submission_ready'] and not result['independent_confirmation']
    assert not result['stage5c_executed'] and not result['smc_enabled']


def test_current_query_reduction_refuses_incomplete_roster(monkeypatch):
    values, cfg = fixture(monkeypatch)
    with pytest.raises(ValueError): run.summarize(values[:-1], cfg)
