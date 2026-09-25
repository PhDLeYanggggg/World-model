"""Label-free port of frozen European policies to model-selection localities."""
import numpy as np

from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_european_source_intervention import causal_cost_features
from src.world_model.m3w_geometric_cost_head import rollout_envelope
from src.world_model.m3w_european_conditional_risk import pointwise_rule
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_incumbent_relative import features, choices, replay
from src.world_model.m3w_producer_conditioned import safe_choice

POLICIES = ('floor_reference', 'incumbent_reference', 'add_only', 'remove_only',
            'ridge_incumbent', 'old_stop', 'previous_matched', 'raw_neural')
INPUTS = ('geometry', 'history', 'origin')


def require_selection(manifest, member, *, purpose):
    if purpose != 'frozen_model_selection_readout':
        raise PermissionError('No fitting, calibration or confirmation access')
    rows = [r for r in manifest['recordings'] if r['source_member'] == member]
    if (len(rows) != 1 or rows[0]['role'] != 'model_selection_reserved'
            or rows[0]['training_access']):
        raise PermissionError('Only the preassigned model-selection role may open')
    return rows[0]


def validate_inputs(inputs):
    if set(inputs) != set(INPUTS):
        raise ValueError('Only geometry, history and origin are inference inputs')
    g, h, o = [np.asarray(inputs[k]) for k in INPUTS]
    if (g.shape != (len(h), 476) or h.shape != (len(h), 8, 2)
            or o.shape != (len(h), 2) or not len(h)
            or not all(np.isfinite(v).all() for v in (g, h, o))
            or not np.array_equal(o, h[:, -1])):
        raise ValueError('Aligned finite complete causal history required')
    return g, h, o


def motion_floor(inputs, utility, risk):
    g, h, o = validate_inputs(inputs)
    cv = baseline_numpy(h, 1)-o[:, None]
    damping = baseline_numpy(h, 3)-o[:, None]
    x, _ = causal_cost_features(g, cv, damping)
    env = rollout_envelope(cv, damping)
    u, r = utility(x, env), risk(x, env)
    u[np.all(cv == damping, axis=(1, 2))] = 0
    moving = np.linalg.norm(np.diff(h, axis=1), axis=2).sum(1) > 0
    bit = pointwise_rule(u[:, 0]-u[:, 1], r, moving, budget=.02, support_available=True)
    return cv, np.where(bit[:, None, None], damping, cv), bit, u, r


def infer(inputs, neural, floor, heads):
    """Callables are frozen checkpoints; there is deliberately no label argument."""
    g, h, _ = validate_inputs(inputs)
    cv, d, bits = floor[:3]
    x, env = matched_features(g, cv, d, neural, bits)
    moving = np.linalg.norm(h[:, -1]-h[:, -2], axis=1) > 0
    ou, rr = [heads['old_stop', task](x, env) for task in ('utility', 'risk')]
    old = safe_choice(ou, rr, moving)
    xx = features(x, old)
    out = dict(old_stop=old, raw_neural=np.ones(len(x), bool))
    for arm in ('floor_reference', 'incumbent_reference', 'ridge_incumbent', 'previous_matched'):
        xx_arm = x if arm == 'previous_matched' else xx
        u, r = [heads[arm, task](xx_arm, env) for task in ('utility', 'risk')]
        role = 'floor_reference' if arm in ('floor_reference', 'previous_matched') else 'incumbent_reference'
        out[arm] = choices(u, r, moving, old, arm=role)
        np.testing.assert_array_equal(out[arm], replay(u, r, moving, old, arm=role))
        out[arm+'__utility'], out[arm+'__risk'] = u, r
        if arm == 'incumbent_reference':
            for key, direction in (('add_only', 'add'), ('remove_only', 'remove')):
                out[key] = choices(u, r, moving, old, arm=role, direction=direction)
                np.testing.assert_array_equal(out[key], replay(u, r, moving, old, arm=role, direction=direction))
    return out, x, env
