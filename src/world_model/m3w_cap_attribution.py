"""Fixed output-constraint attribution; no fitting or deployment policy."""
import numpy as np
from src.world_model.m3w_risk_conditioned_residual import risk_features
from src.world_model import m3w_temporal_support as temporal

PROJECTIONS = ('nonnegative', 'frozen_cap', 'envelope', 'envelope_coupled')


def unprojected(model, features, prediction, envelope):
    x, p = np.asarray(features, float), np.asarray(prediction, float)
    risk_features(p, envelope)
    if x.shape != (len(p), len(model['mean'])) or not np.isfinite(x).all():
        raise ValueError('Finite aligned causal features required')
    z = np.column_stack((np.ones(len(x)), np.clip((x-model['mean'])/model['std'], -8, 8)))
    return p[:, 3]+z@np.asarray(model['beta'])*model['rms']


def project(score, prediction, envelope):
    q, p, e = np.asarray(score, float), np.asarray(prediction, float), np.asarray(envelope, float)
    risk_features(p, e)
    if q.shape != e.shape or not np.isfinite(q).all():
        raise ValueError('Finite label-free easy-harm scores required')
    values = (np.maximum(q, 0), np.clip(q, 0, p[:, 1]), np.clip(q, 0, e))
    result = {}
    for name, value in zip(PROJECTIONS[:3], values):
        result[name] = p.copy(); result[name][:, 3] = value
    joint = result['envelope'].copy()
    joint[:, 1] = np.maximum(joint[:, 1], joint[:, 3])
    result['envelope_coupled'] = joint
    return result


def diagnose(score, prediction, envelope, target):
    q, p, e, y = map(lambda x: np.asarray(x, float), (score, prediction, envelope, target))
    outputs = project(q, p, e)
    if y.shape != p.shape or np.isinf(y).any():
        raise ValueError('Aligned possibly missing analysis labels required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1), ~known):
        raise ValueError('Partially missing labels are not allowed')
    use = known & (e > 0)
    if not use.any(): return dict(status='not_estimable')
    yy, ee = y[use], e[use]
    tol = 1e-5*np.maximum(1, ee)
    if ((yy < -tol[:, None]).any() or (yy[:, 3] > yy[:, 1]+tol).any()
            or (yy[:, 1] > ee+tol).any()):
        raise ValueError('Labels must respect nonnegative harm and causal disagreement envelope')
    h = yy[:, 3]; signed = q[use]
    a = {key:val[use, 3] for key,val in outputs.items()}
    mse = lambda s: float(np.mean((s-h)**2))
    delta = (a['frozen_cap']-h)**2-(a['envelope']-h)**2
    identity = (a['frozen_cap']-a['envelope'])*(a['frozen_cap']+a['envelope']-2*h)
    np.testing.assert_allclose(delta, identity, rtol=1e-9, atol=1e-9)
    signed_mse, nonnegative_mse, envelope_mse = mse(signed), mse(a['nonnegative']), mse(a['envelope'])
    tolerance = 1e-8*max(1, signed_mse, nonnegative_mse)
    assert nonnegative_mse <= signed_mse+tolerance
    assert envelope_mse <= nonnegative_mse+tolerance
    np.testing.assert_allclose(delta.mean(), mse(a['frozen_cap'])-envelope_mse, rtol=1e-8, atol=1e-9)
    return dict(status='measured', rows=int(use.sum()), signed_easy_MSE=signed_mse,
        nonnegative_easy_MSE=nonnegative_mse, envelope_easy_MSE=envelope_mse,
        lower_projection_MSE_benefit=signed_mse-nonnegative_mse,
        envelope_projection_MSE_benefit=nonnegative_mse-envelope_mse,
        cap_relaxation_MSE_benefit=float(delta.mean()),
        cap_relaxation_helping_mass=float(np.maximum(delta, 0).mean()),
        cap_relaxation_harming_mass=float(np.maximum(-delta, 0).mean()),
        cap_relaxation_improves_rows=int((delta > 0).sum()),
        cap_relaxation_worsens_rows=int((delta < 0).sum()),
        below_zero_fraction=float((signed < 0).mean()),
        above_frozen_cap_fraction=float((signed > p[use, 1]).mean()),
        above_envelope_fraction=float((signed > ee).mean()),
        uncoupled_order_violation_fraction=float((a['envelope'] > p[use, 1]+1e-10).mean()),
        changed_easy_fraction=float((a['frozen_cap'] != a['envelope']).mean()),
        event_cap_relaxation_contribution=float(np.where(h > 0, delta, 0).mean()),
        zero_event_cap_relaxation_contribution=float(np.where(h == 0, delta, 0).mean()),
        identity_max_error=float(np.max(np.abs(delta-identity))),
        target_used_only_for_analysis=True)


def evaluate(score, prediction, envelope, target):
    return {name:temporal.metrics(p, target, envelope)
            for name,p in project(score, prediction, envelope).items()}
