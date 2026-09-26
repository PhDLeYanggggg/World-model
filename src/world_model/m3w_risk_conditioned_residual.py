"""Fixed risk-score-conditioned cost correction, not a safety certificate."""
import numpy as np
from src.world_model import m3w_context_residual as context

RISK_NAMES = ('predicted_harm_over_envelope', 'predicted_easy_harm_over_harm')
ARMS = ('risk_only', 'risk_context')


def risk_features(prediction, envelope):
    p, env = np.asarray(prediction, float), np.asarray(envelope, float)
    if (p.shape != (len(env), 4) or env.ndim != 1 or not np.isfinite(p).all()
            or not np.isfinite(env).all() or (p < 0).any() or (env < 0).any()
            or (p[:, 1] > env+1e-4).any() or (p[:, 3] > p[:, 1]+1e-5).any()
            or ((env == 0) & (p[:, 1] != 0)).any()):
        raise ValueError('Finite nested causal cost predictions and envelope required')
    a = np.divide(p[:, 1], env, out=np.zeros(len(env)), where=env > 0)
    b = np.divide(p[:, 3], p[:, 1], out=np.zeros(len(env)), where=p[:, 1] > 0)
    return np.clip(np.column_stack((a, b)), 0, 1)


def features(causal_context, prediction, envelope, arm):
    x = np.asarray(causal_context, float)
    if x.shape != (len(envelope), len(context.NAMES)) or np.isinf(x).any() or arm not in ARMS:
        raise ValueError('Fixed causal context and registered arm required')
    risk = risk_features(prediction, envelope)
    return risk if arm == 'risk_only' else np.column_stack((x, risk))


def design(x, cuts):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(cuts) or np.isinf(x).any():
        raise ValueError('Aligned feature schema required')
    bins = np.full(x.shape, 3, int)
    for j, cut in enumerate(cuts):
        if cut is not None:
            use = np.isfinite(x[:, j]); bins[use, j] = np.searchsorted(cut, x[use, j], side='left')
    return np.column_stack((np.ones(len(x)), (bins[:, :, None] == np.arange(4)).reshape(len(x), -1)))


def fit(causal_context, prediction, envelope, target, weights, sites, outer, *, arm, variant, ridge=.1):
    p,y,w,ss = map(np.asarray, (prediction,target,weights,sites))
    x = features(causal_context,p,envelope,arm)
    if (y.shape != p.shape or w.shape != (len(p),) or ss.shape != (len(p),)
            or outer in ss or len(set(ss)) != 3 or variant not in ('oof','in_sample_next','in_sample_prev')
            or not np.isfinite(w).all() or (w < 0).any() or ridge <= 0):
        raise ValueError('Three fitting localities, excluded outer and fixed residual provenance required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1), ~known) or (w[~known] != 0).any():
        raise ValueError('Unknown labels must have zero weight')
    use = known & (w > 0)
    if not use.any(): raise ValueError('Fitting support required')
    ww = w[use].astype(float); ww /= ww.sum(); xx = x[use]
    cuts = [context.weighted_cut(xx[:, j], ww) for j in range(x.shape[1])]
    rms = max(float(np.sqrt(ww @ (y[use, 3]**2))), 1e-8)
    response = (y[use, 3]-p[use, 3])/rms
    z = design(xx, cuts); penalty = np.eye(z.shape[1])*ridge; penalty[0, 0] = 0
    lhs = z.T @ (ww[:, None]*z)+penalty; rhs = z.T @ (ww*response)
    beta = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(lhs@beta,rhs,rtol=1e-9,atol=1e-10)
    return dict(arm=arm,variant=variant,cuts=cuts,rms=rms,coefficients=beta.tolist(),ridge=ridge,
        feature_names=list(RISK_NAMES if arm=='risk_only' else context.NAMES+RISK_NAMES),
        fitting_sites=sorted(set(ss)),known_rows=int(use.sum()),
        fitted_with_in_sample_base_predictions=variant!='oof',independent_calibration=False,
        weighted_fitting_residual_MSE=float(ww@((response-z@beta)**2)))


def predict(model, causal_context, prediction, envelope):
    p = np.asarray(prediction,float)
    x = features(causal_context,p,envelope,model['arm'])
    shift = design(x,model['cuts']) @ np.asarray(model['coefficients']) * model['rms']
    result = p.copy(); result[:,3] = np.clip(p[:,3]+shift,0,p[:,1])
    return result


def producer_diagnostic(context_x, inner, outer, env, y, w, inner_model, outer_model):
    """Fitting-only distribution and exact producer-difference accounting."""
    known = np.isfinite(y).all(1); use = known & (w > 0)
    ww = np.asarray(w[use],float); ww /= ww.sum()
    a,b = risk_features(inner,env),risk_features(outer,env)
    assert inner_model['cuts']==outer_model['cuts'] and inner_model['ridge']==outer_model['ridge']
    z = context.design(context_x[use],inner_model['cuts'],'context_bias')
    penalty = np.eye(z.shape[1])*inner_model['ridge']; penalty[0,0]=0
    delta = outer[use,3]-inner[use,3]
    projected = np.linalg.solve(z.T@(ww[:,None]*z)+penalty,z.T@(ww*delta))
    observed = (np.asarray(inner_model['coefficients'])*inner_model['rms']
                -np.asarray(outer_model['coefficients'])*outer_model['rms'])
    np.testing.assert_allclose(projected,observed,rtol=1e-7,atol=1e-8)
    def summary(p, f):
        return dict(mean_easy_harm=float(ww@p[use,3]),
            mean_residual=float(ww@(y[use,3]-p[use,3])),
            fitting_MSE=float(ww@((y[use,3]-p[use,3])**2)),
            risk_feature_mean=(ww[:,None]*f[use]).sum(0).tolist(),
            risk_feature_std=np.sqrt((ww[:,None]*(f[use]-(ww[:,None]*f[use]).sum(0))**2).sum(0)).tolist())
    sa,sb = summary(inner,a),summary(outer,b)
    central = np.quantile(a[use],[.05,.95],axis=0)
    outside = (b[use] < central[0]) | (b[use] > central[1])
    return dict(inner=sa,outer=sb,target_mean=float(ww@y[use,3]),
        prediction_difference_RMS=float(np.sqrt(ww@(delta**2))),
        coefficient_identity_max_error=float(np.max(np.abs(projected-observed))),
        outer_outside_inner_central90_fraction=(ww[:,None]*outside).sum(0).tolist(),
        outer_held_labels_used=False,causal_mechanism_proven=False,
        central90_is_not_support_or_OOD_guarantee=True)
