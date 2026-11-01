import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import run_m3w_current_bank_deferral as run
from tests.test_m3w_active_bank_readout import fixture


def test_positive_known_utility_does_not_rescue_missing_risk_or_empty_match():
    contexts={f'c{i}':[f's{4*(i%3)+j}' for j in range(4)] for i in range(6)};rows=[]
    for context,sources in contexts.items():
        for source in sources:
            p=fixture(source,missing=True)[0]['new_neural']
            choices={n:np.full(2,6) for n in run.evaluation.NAMES if n!='floor'}
            choices['ade_guarded']=np.full(2,-1)
            result=run.evaluation.readout(p,choices,{n:np.ones(2) for n in choices},1.5,np.zeros((2,2)))
            for seed in (17,29,43):
                rows.append(dict(context=context,source=source,seed=seed,result=copy.deepcopy(result),
                    matched_training=True,original_control_exact=True,guarded_predicted_feasible=True,minimum_gradient_query_visits=4))
    cfg=dict(seeds=[17,29,43],bootstrap_seed=7,bootstrap_resamples=32)
    summary=run.evaluation.summarize(rows,contexts,cfg)
    assert summary['accuracy']['tdef_guarded']['point']>0
    assert summary['effects']['guarded_tdef_minus_ade']['matched']['lower_pp']['point'] is None
    assert summary['safety']['tdef_guarded']['selected_unknown']>0
    assert not summary['fitting_advancement'] and not summary['independent_confirmation']
    assert not summary['stage5c_executed'] and not summary['smc_enabled']
    with pytest.raises(ValueError):run.evaluation.summarize(rows[:-1],contexts,cfg)
    with pytest.raises(ValueError):run.evaluation.summarize(rows[:-1]+rows[:1],contexts,cfg)


def test_registered_science_and_dependencies_remain_frozen_when_present():
    if not run.REG.exists():pytest.skip('Prospective registration not yet written')
    reg=json.loads(run.REG.read_text())
    for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():assert run.sha(run.ROOT/rel)==digest
    assert reg['fits']==36 and reg['parameters_each']==30726 and not reg['independent_roles_opened']


def test_actual_complete_bundle_and_replay_when_present():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full experiment not yet complete')
    complete=json.loads((run.PUBLIC/'complete.json').read_text());replay=json.loads((run.PUBLIC/'verification.json').read_text())
    assert replay['all72_new_score_inference_and_action_scores_exact'] and replay['original_full_and_record_controls_exact']
    assert replay['complete_sha256']==run.sha(run.PUBLIC/'complete.json')
    assert replay['summary_sha256']==complete['summary_sha256']==run.sha(run.PUBLIC/'summary.json')
    assert len(complete['groups'])==72 and not replay['optimization_replayed']
