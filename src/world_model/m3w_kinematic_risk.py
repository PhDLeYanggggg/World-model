"""Explicit past-motion basis for a matched, warm-started neural risk critic."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_direct_tail_risk as parent

np, torch, nn, F = parent.np, parent.torch, parent.nn, parent.F
base = parent.base
ARMS = ('masked', 'kinematic')
NAMES = tuple([f'past_delta_{i}_{a}' for i in range(7) for a in ('x', 'y')]
    + [f'past_speed_{i}' for i in range(7)] + [f'past_accel_norm_{i}' for i in range(6)]
    + [f'past_turn_cos_{i}' for i in range(6)] + [f'past_turn_sin_{i}' for i in range(6)]
    + ['past_path_length', 'past_net_length', 'past_straightness', 'neighbor_support',
       'neighbor_min_radius', 'neighbor_mean_radius', 'neighbor_mean_closing',
       'neighbor_max_closing', 'closest_approach_time_fraction', 'closest_approach_distance',
       'neighbor_approaching_fraction']
    + [f'{prefix}_{name}' for prefix in ('floor', 'candidate')
       for name in ('path', 'net', 'mean_speed', 'last_speed', 'speed_std', 'absolute_turn', 'straightness')])


def motion_basis(x):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (7, 428) or not np.isfinite(x).all():
        raise ValueError('Only the original finite causal 428-column seven-action schema')
    h = x[..., :24].reshape(*x.shape[:2], 8, 3)[..., :2].astype(float)
    hm = x[..., 24:32]
    n = x[..., 32:224].reshape(*x.shape[:2], 8, 8, 3)[..., :2].astype(float)
    mask = x[..., 224:288].reshape(*x.shape[:2], 8, 8)
    if not (hm == 1).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError('Complete observed ego history and binary neighbor masks required')
    velocity = np.diff(h, axis=-2); speed = np.linalg.norm(velocity, axis=-1)
    accel = np.linalg.norm(np.diff(velocity, axis=-2), axis=-1)
    a, b = velocity[..., :-1, :], velocity[..., 1:, :]
    denominator = speed[..., :-1]*speed[..., 1:]
    cosine = np.divide((a*b).sum(-1), denominator, out=np.zeros_like(denominator), where=denominator > 1e-12)
    sine = np.divide(a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0], denominator,
        out=np.zeros_like(denominator), where=denominator > 1e-12)
    path, net = speed.sum(-1), np.linalg.norm(h[..., -1, :]-h[..., 0, :], axis=-1)
    straight = np.divide(net, path, out=np.zeros_like(net), where=path > 1e-12)
    parts = [velocity.reshape(*x.shape[:2], 14), speed, accel, cosine, sine,
             np.stack((path, net, straight), -1)]
    valid = (mask[..., -1] == 1) & (mask[..., -2] == 1)
    count = valid.sum(-1); divisor = np.maximum(count, 1)
    relative = np.where(valid[..., None], n[..., -1, :]-h[..., None, -1, :], 0)
    v = np.where(valid[..., None], n[..., -1, :]-n[..., -2, :]-velocity[..., None, -1, :], 0)
    radius = np.linalg.norm(relative, axis=-1); dot = (relative*v).sum(-1)
    closing = np.divide(-dot, radius, out=np.zeros_like(dot), where=radius > 1e-12)
    v2 = (v*v).sum(-1)
    t = np.divide(-dot, v2, out=np.full_like(dot, 12.), where=valid & (dot < 0) & (v2 > 1e-12))
    t = np.clip(t, 0, 12)
    distance = np.linalg.norm(relative+t[..., None]*v, axis=-1)
    def minimum(value):
        return np.where(count > 0, np.where(valid, value, np.inf).min(-1), 0)
    parts.append(np.stack((count/8, minimum(radius), (radius*valid).sum(-1)/divisor,
        (closing*valid).sum(-1)/divisor,
        np.where(count > 0, np.where(valid, closing, -np.inf).max(-1), 0),
        minimum(t)/12, minimum(distance), ((closing > 0) & valid).sum(-1)/divisor), -1))
    for start in (306, 330):
        p = x[..., start:start+24].reshape(*x.shape[:2], 12, 2).astype(float)
        step = np.diff(np.concatenate((np.zeros_like(p[..., :1, :]), p), -2), axis=-2)
        speeds = np.linalg.norm(step, axis=-1)
        a, b = step[..., :-1, :], step[..., 1:, :]
        valid_turn = (speeds[..., :-1] > 1e-12) & (speeds[..., 1:] > 1e-12)
        turn = np.where(valid_turn, np.abs(np.arctan2(a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0], (a*b).sum(-1))), 0)
        path, net = speeds.sum(-1), np.linalg.norm(p[..., -1, :], axis=-1)
        parts.append(np.stack((path, net, speeds.mean(-1), speeds[..., -1], speeds.std(-1),
            turn.sum(-1), np.divide(net, path, out=np.zeros_like(net), where=path > 1e-12)), -1))
    result = np.concatenate(parts, -1).astype(np.float32)
    if result.shape != (*x.shape[:2], len(NAMES)) or len(NAMES) != 64 or not np.isfinite(result).all():
        raise ValueError('Finite named kinematic64 basis required')
    return result


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(data, warm):
    c = data['causal']
    if set(c) != base.CAUSAL_KEYS:
        raise ValueError('Causal fields only; future inference fields forbidden')
    for key, value in (('fit_ids', c['ids']), ('preprocess', data['preprocess']),
            ('input_hash', digest(data['z'].numpy())),
            ('target_hash', digest(data['truth_by_arm']['bound_aware'].numpy())),
            ('proposal_mask_hash', digest(c['allowed']))):
        base.exact(warm[key], value)
    raw = motion_basis(c['x'])
    weights = np.zeros(len(raw), float)
    for site in sorted(set(c['sites'])):
        at = c['sites'] == site; weights[at] = 1/(len(set(c['sites']))*at.sum())
    mean = np.einsum('n,nkd->d', weights, raw.astype(float))/7
    second = np.einsum('n,nkd->d', weights, raw.astype(float)**2)/7
    pr = dict(mean=mean, std=np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6), clip=8.,
        source_names=sorted(set(c['sites'])), names=NAMES, labels_used=False)
    extra = np.clip((raw-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
    return dict(data, extra=torch.from_numpy(extra), extra_preprocess=pr,
        extra_raw_hash=digest(raw), extra_hash=digest(extra))


class KinematicRiskHead(parent.DirectRiskHead):
    def __init__(self, width, seed):
        super().__init__(428, width, seed)
        self.kinematics = nn.Linear(64, width, bias=False)
        nn.init.zeros_(self.kinematics.weight)

    def forward(self, x, segments, count):
        if x.shape[1:] != (7, 492) or segments.shape != (len(x),):
            raise ValueError('Original428 plus causal64 and complete query groups required')
        code = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        h = F.silu(self.encoder(torch.cat((x[..., :428], code), -1))+self.kinematics(x[..., 428:]))
        membership = F.one_hot(segments, count).to(h.dtype).T
        total = membership @ h.flatten(1); sizes = membership.T @ membership.sum(1)
        other = ((membership.T @ total).reshape_as(h)-h)/(sizes-1).clamp_min(1)[:, None, None]
        return self.excess(F.silu(h+other))


def initialize(warm):
    model = KinematicRiskHead(warm['settings']['width'], warm['seed'])
    result = model.load_state_dict(warm['model'], strict=False)
    if result.missing_keys != ['kinematics.weight'] or result.unexpected_keys:
        raise ValueError('Only a zero-initialized kinematic branch may be new')
    return model


def batch_features(data, arm):
    if arm not in ARMS:
        raise ValueError('Registered masked/kinematic pair required')
    return torch.cat((data['z'], data['extra'] if arm == 'kinematic' else torch.zeros_like(data['extra'])), -1)


def fit(data, warm, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    z = batch_features(data, arm); c = data['causal']; model = initialize(warm)
    if settings['width'] != warm['settings']['width']:
        raise ValueError('Unchanged common model width required')
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm=arm, settings=settings, identity=identity, seed=warm['seed'], preprocess=data['preprocess'],
        extra_preprocess=data['extra_preprocess'], extra_raw_hash=data['extra_raw_hash'], extra_hash=data['extra_hash'],
        fit_ids=c['ids'].copy(), source_sites=sorted(set(c['sites'])), initial_model=initial,
        parameters=sum(v.numel() for v in model.parameters()), budget=budget, risk_budget=.02,
        warm_model=warm['model'], warm_step=warm['step'], optimizer_reused=False,
        common_input_hash=digest(data['z'].numpy()), target_hash=digest(data['truth_by_arm']['bound_aware'].numpy()),
        proposal_hash=digest(c['allowed']), input_hash=digest(z.numpy()), objective='unchanged_direct_mean_MSE')
    step, elapsed, trace, draws = 0, 0., [], '0'*64
    observed, unknown = 0, 0
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        old = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(old[key], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer']); sampler.restore(old['sampler'])
        torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed, unknown = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else stop_at
    if not 0 <= step <= limit <= budget or limit <= 0:
        raise ValueError('Fixed nondecreasing budget required')
    started = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(),
            torch_rng=torch.get_rng_state(), trace=trace, draw_hash=draws,
            observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown,
            threshold_search=False, calibrated_safety_bound=False))
    if not step:
        save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
            optimizer.zero_grad(set_to_none=True)
            terms = parent.objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), 'mean')
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Nonfinite unchanged neural risk objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, loss=float(terms['total'].detach()), gradient_norm=float(norm),
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='kinematic_risk_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Original complete query coverage required')
    return base.read_checkpoint(path)


def assert_matched(a, b):
    if (a['arm'], b['arm']) != ARMS:
        raise ValueError('Both feature arms required')
    for key in ('settings', 'identity', 'seed', 'preprocess', 'extra_preprocess', 'extra_raw_hash', 'extra_hash',
            'fit_ids', 'source_sites', 'initial_model', 'parameters', 'budget', 'risk_budget', 'warm_model',
            'warm_step', 'optimizer_reused', 'common_input_hash', 'target_hash', 'proposal_hash', 'objective',
            'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Only causal schema; future targets or masks forbidden')
    pr, extra = state['preprocess'], state['extra_preprocess']
    if state['arm'] not in ARMS or state['risk_budget'] != .02:
        raise ValueError('Frozen original risk rule required')
    model = KinematicRiskHead(state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    out = np.empty((len(causal['x']), 7, 2), float)
    pending, size = [], 0
    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        x = causal['x'][ix]
        z = np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32); z[..., 380:] = 0
        raw = motion_basis(x)
        e = np.clip((raw-extra['mean'])/extra['std'], -extra['clip'], extra['clip']).astype(np.float32)
        if state['arm'] == 'masked':
            e[:] = 0
        with torch.no_grad():
            p = model(torch.from_numpy(np.concatenate((z, e), -1)), torch.from_numpy(seg), len(pending)).numpy()
        out[ix] = p.astype(float)*parent.normalizer(pr)*causal['scales'][ix, None, None]
    for group in causal['groups']:
        if pending and size+len(group) > 1024:
            flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending:
        flush()
    if not np.isfinite(out).all():
        raise FloatingPointError('Finite direct budget predictions required')
    return out
