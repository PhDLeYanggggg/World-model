import inspect
import json
from pathlib import Path
from scripts import run_m3w_european_fixed_floor_probe as run
from src.world_model import m3w_fixed_floor_probe as api


def test_inference_has_no_outcome_argument_or_lookup():
    assert list(inspect.signature(api.predict).parameters)==['model','x','envelope']
    for f in (run.contexts,run.score,run.prediction_replay):
        text=inspect.getsource(f)
        for forbidden in ('target_eval','baseline_ade','baseline_fde',"data['valid']"):
            assert forbidden not in text


def test_fixed_floor_does_not_use_probe_calibration():
    text=inspect.getsource(run.contexts)
    assert "decide(dz['excess'],dz['guard'],dz['supported'],0.)" in text
    assert 'calibrate' not in text and 'costs(' not in text


def test_registered_scope_and_unknown_denominator():
    cfg=json.loads(Path(run.CONFIG).read_text())
    assert cfg['groups']==108 and cfg['linear_heads']==216
    assert cfg['risk_budget']==.02 and cfg['bootstrap_resamples']>=2000
    assert all(not cfg[k] for k in ('independent_roles_read','threshold_search','deployment_changed','stage5c_executed','smc_enabled'))
    assert 'take.mean()' in inspect.getsource(run.metric)
    assert 'fit_sites,held_sites' in inspect.getsource(run.fit_all)
