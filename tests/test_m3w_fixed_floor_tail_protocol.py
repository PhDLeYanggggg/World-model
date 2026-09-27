import inspect
import json
from pathlib import Path
from scripts import run_m3w_european_fixed_floor_tail as run


def test_causal_inference_and_same_query_allocation():
    for f in (run.action_arrays,run.query_keys,run.verify_causal_first):
        source=inspect.getsource(f)
        for token in ('target_eval','baseline_ade','baseline_fde',"data['valid']"):
            assert token not in source
    assert 'data[\'frames\'][ids]' in inspect.getsource(run.action_arrays)
    assert 'data[\'sites\']' in inspect.getsource(run.query_keys)


def test_registered_loss_only_and_no_independent_role_access():
    cfg=json.loads(Path(run.CONFIG).read_text())
    assert cfg['head_count']==2*cfg['groups']==216
    assert cfg['head_training']['steps']==2000 and cfg['head_training']['checkpoint_every']==500
    assert cfg['risk_budget']==.02 and cfg['bootstrap_resamples']>=2000
    assert not any(cfg[k] for k in ('threshold_search','new_forecaster_training','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))
    source=inspect.getsource(run.prepare)
    assert 'parent.costs(c,data,fit)' in source and 'ridge[\'mean\']' in source
    assert 'matched(states[\'mse\'],states[\'tail4\'])' in inspect.getsource(run.train)
