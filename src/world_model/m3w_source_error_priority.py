"""Source-held error discovery, with unchanged causal inference and risk rules."""
import copy

from src.world_model import m3w_admission_priority as parent

np, torch, base, kin = parent.np, parent.torch, parent.base, parent.parent
ARMS = ('in_source_errors', 'held_source_errors')


def prepare_basis(data):
    c = data['causal']
    if set(c) != base.CAUSAL_KEYS or set(data['preprocess']['source_names']) != set(c['sites']):
        raise ValueError('Own-source causal inputs and preprocessing required')
    raw = kin.motion_basis(c['x'])
    names = sorted(set(c['sites'])); weights = np.zeros(len(raw), float)
    for site in names:
        at = c['sites'] == site; weights[at] = 1/(len(names)*at.sum())
    mean = np.einsum('n,nkd->d', weights, raw.astype(float))/7
    second = np.einsum('n,nkd->d', weights, raw.astype(float)**2)/7
    pr = dict(mean=mean, std=np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6),
        clip=8., source_names=names, names=kin.NAMES, labels_used=False)
    extra = np.clip((raw-pr['mean'])/pr['std'], -8, 8).astype(np.float32)
    return dict(data, extra=torch.from_numpy(extra), extra_preprocess=pr,
        extra_raw_hash=kin.digest(raw), extra_hash=kin.digest(extra))


def scratch(settings, seed):
    model = kin.KinematicRiskHead(settings['width'], seed)
    return dict(settings=settings, seed=seed, step=0, model=copy.deepcopy(model.state_dict()))


def prioritize(data, prediction):
    p = np.asarray(prediction)
    truth = kin.parent.signed(data['truth_by_arm']['bound_aware']).numpy()
    if p.shape != truth.shape or not np.isfinite(p).all():
        raise ValueError('Aligned finite discovery predictions required')
    mask = ((p <= 0) & (truth > 0) & data['observed_future'][:, None, None]
        & data['causal']['allowed'][..., None])
    return dict(data, priority_mask=torch.from_numpy(mask), priority_mask_hash=kin.digest(mask),
        warm_prediction_hash=kin.digest(p))


def teacher_data(data):
    d = prepare_basis(data)
    mask = np.zeros((*d['causal']['allowed'].shape, 2), bool)
    return dict(d, priority_mask=torch.from_numpy(mask), priority_mask_hash=kin.digest(mask),
        warm_prediction_hash='none_scratch_teacher_no_error_priority')


def assert_excluded(state, causal, names):
    kin.parent.parent.assert_excluded(state, causal, names)
    if (set(state['extra_preprocess']['source_names']) != set(state['source_sites'])
            or set(state['extra_preprocess']['source_names']) & set(names)):
        raise ValueError('Held source must not enter extra normalization')


def discovery_pair(sites, predictions, provenance):
    """One equal-budget teacher per row in each arm; never average by outcomes."""
    names = sorted(set(sites))
    if len(names) < 2 or set(predictions) != set(names) or set(provenance) != set(names):
        raise ValueError('At least two sources and one leave-source-out teacher each required')
    shape = next(iter(predictions.values())).shape
    if shape != (len(sites), 7, 2):
        raise ValueError('Complete aligned B discovery predictions required')
    result = {arm: np.empty(shape, np.float64) for arm in ARMS}; mapping = []
    for index, site in enumerate(names):
        excluded_control = names[(index+1) % len(names)]
        for excluded in (site, excluded_control):
            if set(provenance[excluded]) != set(names)-{excluded}:
                raise ValueError('Teacher fit-source provenance must exclude exactly one B source')
            if np.shape(predictions[excluded]) != shape or not np.isfinite(predictions[excluded]).all():
                raise ValueError('Every discovery prediction must be finite and aligned')
        at = np.asarray(sites) == site
        result['held_source_errors'][at] = predictions[site][at]
        result['in_source_errors'][at] = predictions[excluded_control][at]
        mapping.append(dict(rows_source=str(site), held_teacher_excludes=str(site),
            in_source_teacher_excludes=str(excluded_control), rows=int(at.sum())))
    return result, mapping


def assert_student_pair(a, b):
    for key in ('settings', 'seed', 'preprocess', 'extra_preprocess', 'extra_raw_hash', 'extra_hash',
            'fit_ids', 'source_sites', 'initial_model', 'parameters', 'budget', 'risk_budget',
            'warm_step', 'optimizer_reused', 'input_hash', 'target_hash', 'proposal_hash',
            'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])


predict = kin.predict
