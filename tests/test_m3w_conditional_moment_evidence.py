import copy
import json
import numpy as np
import pytest
from scripts import run_m3w_conditional_moment as run
from tests.test_m3w_active_bank_readout import fixture


def test_full_roster_required_and_known_gain_cannot_hide_failed_risk():
    contexts={f'c{i}':[f's{4*(i%3)+j}' for j in range(4)] for i in range(6)};rows=[]
    for context,sources in contexts.items():
        for source in sources:
            p=fixture(source,missing=True)[0]['new_neural'];choices={a:np.full(2,6) for a in run.evaluation.NAMES if a!='floor'}
            result=run.evaluation.readout(p,choices,{a:np.ones(2) for a in choices},
                {a:np.ones_like(p['y']) for a in run.api.ARMS},{a:np.full(2,.5) for a in run.api.ARMS},1.5,np.zeros((2,2)))
            for seed in (17,29,43):rows.append(dict(context=context,source=source,seed=seed,result=copy.deepcopy(result),
                matched_training=True,original_control_exact=True,guarded_predicted_feasible=True,minimum_gradient_query_visits=4))
    cfg=dict(seeds=[17,29,43],bootstrap_seed=7,bootstrap_resamples=32)
    summary=run.evaluation.summarize(rows,contexts,cfg)
    assert not summary['fitting_advancement'] and not summary['independent_confirmation']
    assert summary['safety']['identified_guarded']['selected_unknown']>0
    with pytest.raises(ValueError):run.evaluation.summarize(rows[:-1],contexts,cfg)
    with pytest.raises(ValueError):run.evaluation.summarize(rows[:-1]+rows[:1],contexts,cfg)


def test_registered_definitions_when_present():
    if not run.REG.exists():pytest.skip('Pre-outcome registration not written yet')
    reg=json.loads(run.REG.read_text())
    for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():assert run.sha(run.ROOT/rel)==digest
    assert reg['parameters_each']==31381 and not reg['independent_roles_opened']


def test_complete_real_bundle_when_available():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full real experiment not yet complete')
    c=json.loads((run.PUBLIC/'complete.json').read_text());v=json.loads((run.PUBLIC/'verification.json').read_text())
    assert len(c['groups'])==72 and v['all72_new_moment_probability_inference_and_action_scores_exact']
    assert v['summary_sha256']==c['summary_sha256']==run.sha(run.PUBLIC/'summary.json')
    assert v['complete_sha256']==run.sha(run.PUBLIC/'complete.json')
