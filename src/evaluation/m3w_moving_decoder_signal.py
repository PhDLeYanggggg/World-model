"""Frozen decoder arithmetic and link sensitivity; not a causal training claim."""
import numpy as np
import torch

from src.world_model import m3w_shared_motion_cost as api

THRESHOLDS = (1e-2, 1e-4, 1e-6)


def probe(state, causal):
    if set(causal) != api.base.base.CAUSAL_KEYS:
        raise ValueError('Only the frozen causal input schema is accepted')
    pr = state['preprocess']
    arm = state['identity']['feature_arm']
    if arm not in api.ARMS or causal['x'].shape[-1] != api.DIMENSION:
        raise ValueError('Original matched feature treatment required')
    model = api.base.base.CostHead(api.DIMENSION, state['settings']['width'],
        state['seed'], pr['bias'], 'context')
    model.load_state_dict(state['model'])
    model.eval()
    double = api.base.base.CostHead(api.DIMENSION, state['settings']['width'],
        state['seed'], pr['bias'], 'context')
    double.load_state_dict(state['model'])
    double = double.double().eval()
    n = len(causal['x'])
    first, second = np.empty((n, 7, 5)), np.empty((n, 7, 5))
    raw = np.empty((n, 3))
    captured = []
    hook = model.action.register_forward_hook(lambda _, __, value:captured.append(value.detach()))
    pending, size = [], 0

    def flush():
        ix = np.concatenate(pending)
        segments = torch.from_numpy(np.repeat(np.arange(len(pending)), [len(v) for v in pending]))
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        if arm == 'local_history':
            z[..., api.COMMON_DIMENSION:] = 0
        scale = causal['scales'][ix]
        envelope = (causal['envelope'][ix]/scale[:, None]).astype(np.float32)
        x, env = torch.from_numpy(z), torch.from_numpy(envelope)
        with torch.no_grad():
            p = model(x, env, segments, len(pending))
            q = double(x.double(), env.double(), segments, len(pending))
        if len(captured) != 1:
            raise ValueError('Exactly one original action decode per causal chunk required')
        raw[ix] = captured.pop().numpy()[:, 6]
        first[ix] = p.numpy().astype(float)*scale[:, None, None]
        second[ix] = q.numpy()*scale[:, None, None]

    try:
        for group in causal['groups']:
            if pending and size+len(group) > 1024:
                flush()
                pending, size = [], 0
            pending.append(group)
            size += len(group)
        if pending:
            flush()
    finally:
        hook.remove()
    np.testing.assert_array_equal(first, api.predict(state, causal))
    if not all(np.isfinite(v).all() for v in (raw, first, second)):
        raise FloatingPointError('Finite frozen precision comparison required')
    return dict(raw_action6=raw, float32=first, same_float32_inputs_float64=second)


def signal(raw_action6, first, second, truth, known, choice):
    raw, p, q, y, mask, action = map(np.asarray,
        (raw_action6, first, second, truth, known, choice))
    n = len(raw)
    if (raw.shape != (n, 3) or p.shape != (n, 7, 5) or q.shape != p.shape or y.shape != p.shape
            or mask.shape != (n,) or mask.dtype != bool or action.shape != (n,)
            or action.dtype.kind not in 'iu' or not np.isin(action, [-1, 6]).all()
            or not all(np.isfinite(v).all() for v in (raw, p, q, y))
            or any((v < 0).any() for v in (p, q, y))):
        raise ValueError('Finite aligned probe, label-only moments and fixed causal choices required')
    logits = np.column_stack((raw[:, :2], np.zeros(n)))
    logits -= logits.max(1, keepdims=True)
    soft = np.exp(logits)
    soft /= soft.sum(1, keepdims=True)
    gate = torch.sigmoid(torch.from_numpy(raw[:, 2])).numpy()
    h_link = soft[:, 1]*(1-soft[:, 1])
    eh_link = gate*(1-gate)
    rows = {}
    for name, take in (('all_neural', np.ones(n, bool)), ('independent_selected', action == 6)):
        use = take & mask
        mass = y[use, 6, 4]
        denominator = float(mass.sum())
        fraction = lambda at:float(y[use & at, 6, 4].sum()/denominator) if denominator > 0 else None
        rows[name] = dict(row_views=int(take.sum()), known_positive_EH_rows=int((use & (y[:, 6, 4] > 0)).sum()),
            known_EH_mass=denominator,
            exact_float32_zero_EH_rows=int((take & (p[:, 6, 4] == 0)).sum()),
            known_EH_mass_fraction_float32_underflow_recovered_by_float64=
                fraction((p[:, 6, 4] == 0) & (q[:, 6, 4] > 0)),
            attenuation=[dict(threshold=t, known_EH_mass_fraction_small_h_link=fraction(h_link < t),
                known_EH_mass_fraction_small_EH_gate_link=fraction(eh_link < t),
                known_EH_mass_fraction_either_small_link=fraction((h_link < t) | (eh_link < t)))
                for t in THRESHOLDS])
    return dict(pools=rows,
        maximum_absolute_float32_float64_moment_difference=float(np.abs(p-q).max()),
        derivative_definition='dH_dHlogit_divided_by_envelope_and_dEH_dgate_logit_divided_by_H',
        logit_link_derivatives_not_parameter_gradients=True,
        neither_a_training_cause_proof_nor_a_safety_gate=True, future_labels_used_for_choice=False)
