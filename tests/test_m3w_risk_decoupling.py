import copy
import numpy as np
import pytest

from src.world_model import m3w_risk_decoupling as experiment
from src.evaluation import m3w_risk_decoupling_readout as readout
from tests.test_m3w_query_context_risk import anchor, fit
from tests.test_m3w_decoder_risk_factorial import packets


def continuation(path, origin, arm, **kwargs):
    return experiment.fit(packets(), origin=origin, origin_sha256='origin', objective_arm=arm,
                          updates=6, path=path, registration_sha256='test', heartbeat=lambda **_: None, **kwargs)


@pytest.mark.parametrize('arm', experiment.ARMS)
def test_continuation_resume_exact_and_protected(tmp_path, arm):
    origin = fit(tmp_path/'origin.pt.gz', anchor(tmp_path), 'self'); saved = copy.deepcopy(origin)
    direct = continuation(tmp_path/'direct.pt.gz', origin, arm)
    continuation(tmp_path/'resume.pt.gz', origin, arm, stop_at=3)
    resumed = continuation(tmp_path/'resume.pt.gz', origin, arm, resume=True)
    for k in direct:
        if k != 'continuation_seconds': experiment.api.core.exact(direct[k], resumed[k])
    experiment.api.core.exact(origin, saved)
    assert direct['step'] == origin['step']+6
    assert any(not experiment.torch.equal(v, origin['model'][k]) for k, v in direct['model'].items()
               if k.startswith('risk_encoder.'))
    if arm == 'estimation': experiment.api.core.exact(direct['dual'], origin['dual'])
    for source, args in packets().items():
        a = experiment.api.predict(origin, source, args[0], args[1], args[5], args[6])
        b = experiment.api.predict(direct, source, args[0], args[1], args[5], args[6])
        np.testing.assert_array_equal(a[:, [0, 2, 3]], b[:, [0, 2, 3]])
    with pytest.raises(ValueError): continuation(tmp_path/'direct.pt.gz', origin, arm)


def test_matched_draws_and_coupled_first_update_replays_existing_objective(tmp_path):
    origin = fit(tmp_path/'origin.pt.gz', anchor(tmp_path), 'self')
    states = [continuation(tmp_path/(a+'.pt.gz'), origin, a, stop_at=1) for a in experiment.ARMS]
    experiment.matched(*states)
    model = experiment.audit.model_from(origin)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = experiment.torch.optim.AdamW(params, lr=origin['settings']['learning_rate'], weight_decay=.0001)
    optimizer.load_state_dict(copy.deepcopy(origin['optimizer'])); optimizer.zero_grad(set_to_none=True)
    batch = experiment.audit.prepare_probes(packets(), origin, 1)[0]
    parts, _ = experiment.audit.objective_parts(model, origin, batch)
    sum(parts.values()).backward()
    experiment.torch.nn.utils.clip_grad_norm_(params, origin['settings']['gradient_clip'])
    optimizer.step()
    # Summing two per-source dual components separately can introduce float32 reduction noise.
    for k, value in model.state_dict().items():
        experiment.torch.testing.assert_close(value, states[0]['model'][k], atol=1e-7, rtol=1e-6)
    altered = copy.deepcopy(states[1]); altered['samplers']['a']['row_draws'] += 1
    with pytest.raises(AssertionError): experiment.matched(states[0], altered)


def test_unknown_outcomes_remain_in_fixed_cohort_not_loss():
    pred = np.array([[1., .01, 1., 1., .005]]*2)
    truth = np.array([[0., .2, 1., 1., .2], [np.nan]*5])
    result = readout.score(pred, truth, np.ones(2, bool), 1.)
    assert result['rows'] == 2 and result['known'] == result['unknown'] == 1
    assert result['easy']['MSE'] == pytest.approx(.195**2)
    with pytest.raises(ValueError): readout.summarize([], set())
