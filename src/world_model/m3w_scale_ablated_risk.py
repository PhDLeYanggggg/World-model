"""Retrained removal of explicit absolute scale, not full-policy invariance."""
from src.world_model import m3w_signed_decision as parent

np, torch, base = parent.np, parent.torch, parent.base
ARMS = parent.ARMS
COLUMN = 354
SCHEMA = 'critic_explicit_log_scale_removed_v1'


def prepare(data):
    if set(data['causal']) != base.CAUSAL_KEYS or data['z'].shape[1:] != (7, 428):
        raise ValueError('Original causal schema and original common preprocessing required')
    z = data['z'].clone(); z[..., COLUMN] = 0
    return dict(data, z=z)


def fit(data, warm, *, identity, **kwargs):
    return parent.fit(prepare(data), warm, identity=dict(identity, feature_schema=SCHEMA), **kwargs)


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS or state['identity'].get('feature_schema') != SCHEMA:
        raise ValueError('Only causal inputs and registered scale-ablated critic')
    if state['arm'] != 'kinematic' or state['risk_budget'] != .02:
        raise ValueError('Unchanged architecture and risk budget required')
    pr, extra = state['preprocess'], state['extra_preprocess']
    model = parent.parent.KinematicRiskHead(state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    out = np.empty((len(causal['x']), 7, 2), float)
    pending, size = [], 0

    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        x = causal['x'][ix]
        z = np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0; z[..., COLUMN] = 0
        raw = parent.parent.motion_basis(x)
        e = np.clip((raw-extra['mean'])/extra['std'], -extra['clip'], extra['clip']).astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(np.concatenate((z, e), -1)), torch.from_numpy(seg), len(pending)).numpy()
        out[ix] = p.astype(float)*parent.parent.parent.normalizer(pr)*causal['scales'][ix, None, None]

    for group in causal['groups']:
        if pending and size+len(group) > 1024:
            flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending:
        flush()
    if not np.isfinite(out).all():
        raise FloatingPointError('Finite decision scores required')
    return out


def assert_control(state, control, data):
    for key in ('settings', 'seed', 'preprocess', 'extra_preprocess', 'extra_raw_hash', 'extra_hash',
            'fit_ids', 'source_sites', 'initial_model', 'parameters', 'budget', 'risk_budget', 'warm_step',
            'optimizer_reused', 'target_hash', 'observed_mask_hash', 'proposal_hash', 'objective_arm',
            'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(state[key], control[key])
    original = parent.parent.batch_features(data, 'kinematic').numpy()
    transformed = parent.parent.batch_features(prepare(data), 'kinematic').numpy()
    base.exact(parent.parent.digest(original), control['input_hash'])
    base.exact(parent.parent.digest(transformed), state['input_hash'])
    columns = np.arange(492) != COLUMN
    np.testing.assert_array_equal(original[..., columns], transformed[..., columns])
    if np.any(transformed[..., COLUMN] != 0) or state['identity']['feature_schema'] != SCHEMA:
        raise ValueError('Only explicit absolute-scale input may be removed')
