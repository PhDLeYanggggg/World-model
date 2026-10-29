"""Fit-only derivative accounting for frozen bounded harm predictors."""
import numpy as np

ARMS = ('isolated_odds', 'full_range')
COHORTS = ('eligible', 'anchor', 'isolated_odds', 'full_range')


def collect(api, state, source, x, envelope, *, chunk=4096):
    """Return predictions and row-local Jacobians; no outcomes enter the model."""
    pr = state['anchor_state']['source_preprocess'][source]
    api.previous.old.parent.forest.causal_inputs(x, envelope, pr)
    api.check_protection(state)
    model = api.initialize(state['anchor_state'], state['arm'])
    model.load_state_dict(state['model'])
    model.eval()
    # The frozen architecture uses row-local layers, so sum derivatives are per row.
    if any(isinstance(m, (api.torch.nn.modules.batchnorm._BatchNorm, api.torch.nn.MultiheadAttention))
           for m in model.modules()):
        raise ValueError('Row-coupled model requires explicit diagonal Jacobians')
    captured = {}
    h1 = model.risk_output.register_forward_hook(lambda _m, _i, out: captured.update(raw=out))
    h2 = model.base.register_forward_hook(lambda _m, _i, out: captured.update(base=out[0]))
    values = {k: [] for k in ('prediction', 'anchor', 'jacobian', 'raw', 'cap', 'anchor_h', 'anchor_q')}
    try:
        for start in range(0, len(x), chunk):
            zz = np.clip((x[start:start+chunk]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
            ee = (envelope[start:start+chunk]/pr['scale']).astype(np.float32)
            env = api.torch.from_numpy(ee)
            pred = model(api.torch.from_numpy(zz), env)
            raw, base = captured['raw'], captured['base']
            jac = api.torch.stack([api.torch.autograd.grad(pred[:, k].sum(), raw,
                retain_graph=k == 1)[0] for k in (1, 4)], dim=1)
            cap = (env-base[:, 0]).clamp_min(0)
            hh = api.torch.minimum(base[:, 1], cap)
            qq = (base[:, 4].double()/base[:, 1].double().clamp_min(api.torch.finfo(base.dtype).tiny)).clamp(0, 1)
            for name, val in dict(prediction=pred, anchor=base, jacobian=jac,
                                  raw=raw, cap=cap, anchor_h=hh, anchor_q=qq).items():
                values[name].append(val.detach().numpy().astype(float))
    finally:
        h1.remove(); h2.remove()
    result = {k: np.concatenate(v) for k, v in values.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite diagnostic output')
    result['prediction'] *= pr['scale']
    result['anchor'] *= pr['scale']
    return result


def flags(values, arm):
    cap, h, q, raw = (values[k] for k in ('cap', 'anchor_h', 'anchor_q', 'raw'))
    shift = np.tanh(raw.astype(np.float32)).astype(float)
    if arm == 'full_range':
        h_lower = cap*shift[:, 0] < -h
        h_upper = cap*shift[:, 0] > cap-h
        q_lower = shift[:, 1] < -q
        q_upper = shift[:, 1] > 1-q
    elif arm == 'isolated_odds':
        h_lower = h_upper = q_lower = q_upper = np.zeros(len(cap), bool)
    else:
        raise ValueError('Unknown registered arm')
    return dict(hard_h_lower=h_lower, hard_h_upper=h_upper, hard_q_lower=q_lower,
        hard_q_upper=q_upper, zero_cap=cap == 0,
        raw_tanh_h_exact_saturation=np.abs(shift[:, 0]) == 1,
        raw_tanh_q_exact_saturation=np.abs(shift[:, 1]) == 1)


def account(values, truth, cohort, scale, arm):
    p, y = values['prediction']/scale, np.asarray(truth, float)/scale
    cohort = np.asarray(cohort, bool)
    if p.shape != y.shape or p.shape[1] != 5 or cohort.shape != (len(y),) or scale <= 0:
        raise ValueError('Aligned rows and positive frozen fitting scale required')
    known = cohort & np.isfinite(y).all(1)
    j = np.abs(values['jacobian']).sum(axis=2)
    cap = values['cap']
    f = flags(values, arm)
    result = dict(rows=int(cohort.sum()), known=int(known.sum()), unknown=int((cohort & ~known).sum()),
        causes={k: int((v & cohort).sum()) for k, v in f.items()})
    for col, idx, name in ((1, 0, 'all'), (4, 1, 'easy')):
        zero = j[:, idx] == 0
        near = j[:, idx] <= 1e-6*np.maximum(cap, np.finfo(float).tiny)
        under = np.maximum(y[known, col]-p[known, col], 0)
        result[name] = dict(zero_jacobian_rows=int((zero & cohort).sum()),
            near_zero_jacobian_rows=int((near & cohort).sum()),
            known_zero_jacobian_rows=int((zero & known).sum()),
            known_positive_harm_zero_jacobian_rows=int((zero[known] & (y[known, col] > 0)).sum()),
            jacobian_l1_sum=float(j[cohort, idx].sum()),
            observed_harm=float(y[known, col].sum()),
            predicted_harm=float(p[known, col].sum()),
            positive_underprediction=float(under.sum()),
            underprediction_zero_jacobian=float(under[zero[known]].sum()),
            underprediction_near_zero_jacobian=float(under[near[known]].sum()),
            underprediction_hard_h_lower=float(under[f['hard_h_lower'][known]].sum()),
            underprediction_hard_q_lower=float(under[f['hard_q_lower'][known]].sum()),
            observed_above_cap_rows=int((y[known, col] > cap[known]+1e-7).sum()))
    validate(result)
    return result


def validate(row):
    if row['rows'] != row['known']+row['unknown']:
        raise ValueError('Missing outcomes must remain explicit')
    for name in ('all', 'easy'):
        d = row[name]
        if not 0 <= d['known_zero_jacobian_rows'] <= row['known'] <= row['rows']:
            raise ValueError('Invalid known-row accounting')
        if not 0 <= d['zero_jacobian_rows'] <= d['near_zero_jacobian_rows'] <= row['rows']:
            raise ValueError('Invalid derivative accounting')
        if not 0 <= d['underprediction_zero_jacobian'] <= d['underprediction_near_zero_jacobian']+1e-8 <= d['positive_underprediction']+2e-8:
            raise ValueError('Invalid underprediction decomposition')


def summarize(rows):
    keys = {(r['context'], r['source'], r['seed']) for r in rows}
    if len(rows) != 72 or len(keys) != 72 or any(r['scope'] != 'fit' for r in rows):
        raise ValueError('Exactly72 fitting-only views required')
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for cohort in COHORTS:
            vv = [r['result'][arm][cohort] for r in rows]
            for v in vv:
                validate(v)
            out = {k: sum(v[k] for v in vv) for k in ('rows', 'known', 'unknown')}
            for key in ('all', 'easy', 'causes'):
                out[key] = {k: sum(v[key][k] for v in vv) for k in vv[0][key]}
            validate(out)
            result[arm][cohort] = out
    return result
