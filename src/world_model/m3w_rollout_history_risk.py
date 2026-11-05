"""Matched causal rollout/history feature basis; original joint risk objective."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_paired_decision_risk as previous

parent, base = previous.parent, previous.base
np, torch, nn, F = parent.np, parent.torch, parent.nn, parent.F
ARMS = ('masked', 'relational')
NAMES = tuple(f'{p}_{v}' for p in ('floor', 'candidate') for v in (
    'start_forward_delta', 'start_side_delta', 'end_forward_departure_per_step',
    'end_side_per_step', 'end_heading_cos', 'end_heading_sin', 'mean_step_heading_cos',
    'mean_step_heading_sin', 'initial_acceleration_cos', 'mean_speed_to_history')) + tuple(
    f'candidate_minus_floor_step{t}_{v}' for t in (1, 3, 6, 12) for v in ('forward', 'side', 'norm'))


def basis(x):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (7, 428) or not np.isfinite(x).all():
        raise ValueError('Finite original causal schema required')
    if not (x[..., 24:32] == 1).all():
        raise ValueError('Complete past-only eight-state history required')
    h = x[..., :24].reshape(*x.shape[:2], 8, 3)[..., :2].astype(float)
    velocity = np.diff(h, axis=-2); v = velocity[..., -1, :]
    speed = np.linalg.norm(v, axis=-1)
    direction = np.divide(v, speed[..., None], out=np.zeros_like(v), where=speed[..., None] > 1e-12)
    mean_speed = np.linalg.norm(velocity, axis=-1).mean(-1)
    acceleration = velocity[..., -1, :]-velocity[..., -2, :]
    def forward(a):
        return (a*direction).sum(-1)
    def side(a):
        return direction[..., 0]*a[..., 1]-direction[..., 1]*a[..., 0]
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 1e-12)
    paths, parts = [], []
    for start in (306, 330):
        p = x[..., start:start+24].reshape(*x.shape[:2], 12, 2).astype(float); paths.append(p)
        step = np.diff(np.concatenate((np.zeros_like(p[..., :1, :]), p), -2), axis=-2)
        length = np.linalg.norm(step, axis=-1); valid = length > 1e-12
        endpoint = p[..., -1, :]; endpoint_norm = np.linalg.norm(endpoint, axis=-1)
        cs = ratio((step*direction[..., None, :]).sum(-1), length)
        sn = ratio(direction[..., None, 0]*step[..., 1]-direction[..., None, 1]*step[..., 0], length)
        initial_acceleration = step[..., 0, :]-v
        parts.append(np.stack((forward(initial_acceleration), side(initial_acceleration),
            forward(endpoint/12-v), side(endpoint)/12, ratio(forward(endpoint), endpoint_norm),
            ratio(side(endpoint), endpoint_norm), cs.sum(-1)/valid.sum(-1).clip(1),
            sn.sum(-1)/valid.sum(-1).clip(1), ratio((initial_acceleration*acceleration).sum(-1),
                np.linalg.norm(initial_acceleration, axis=-1)*np.linalg.norm(acceleration, axis=-1)),
            ratio(length.mean(-1), mean_speed)), -1))
    difference = paths[1]-paths[0]
    for index in (0, 2, 5, 11):
        a = difference[..., index, :]
        parts.append(np.stack((forward(a), side(a), np.linalg.norm(a, axis=-1)), -1))
    out = np.concatenate(parts, -1).astype(np.float32)
    if out.shape != (*x.shape[:2], 32) or not np.isfinite(out).all():
        raise ValueError('Finite causal32 relation basis required')
    return out


def prepare(data, origin):
    parent.validate(data, origin)
    raw = basis(data['causal']['x']); sites = data['causal']['sites']
    w = np.zeros(len(raw), float)
    for site in sorted(set(sites)):
        at = sites == site; w[at] = 1/(len(set(sites))*at.sum())
    mean = np.einsum('n,nkd->d', w, raw.astype(float))/7
    second = np.einsum('n,nkd->d', w, raw.astype(float)**2)/7
    pr = dict(mean=mean, std=np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6), clip=8.,
        names=NAMES, source_names=sorted(set(sites)), labels_used=False)
    rel = np.clip((raw-pr['mean'])/pr['std'], -8, 8).astype(np.float32)
    return dict(data, relation=torch.from_numpy(rel), relation_preprocess=pr,
        relation_raw_hash=parent.kin.digest(raw), relation_hash=parent.kin.digest(rel))


class Head(parent.kin.KinematicRiskHead):
    def __init__(self, width, seed):
        super().__init__(width, seed)
        self.relation = nn.Linear(32, width, bias=False)
        nn.init.zeros_(self.relation.weight)

    def forward(self, x, segments, count):
        if x.shape[1:] != (7, 524) or segments.shape != (len(x),):
            raise ValueError('Causal492 plus relation32 and complete query groups required')
        code = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        h = F.silu(self.encoder(torch.cat((x[..., :428], code), -1))+
            self.kinematics(x[..., 428:492])+self.relation(x[..., 492:]))
        membership = F.one_hot(segments, count).to(h.dtype).T
        total = membership @ h.flatten(1); sizes = membership.T @ membership.sum(1)
        other = ((membership.T @ total).reshape_as(h)-h)/(sizes-1).clamp_min(1)[:, None, None]
        return self.excess(F.silu(h+other))


class Model(nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        self.risk = Head(width, seed); self.decision = copy.deepcopy(self.risk)

    def forward(self, x, segments, count):
        return self.risk(x, segments, count), self.decision(x, segments, count)


def initialize(origin):
    model = Model(origin['settings']['width'], origin['seed'])
    for head in (model.risk, model.decision):
        result = head.load_state_dict(origin['model'], strict=False)
        if result.missing_keys != ['relation.weight'] or result.unexpected_keys:
            raise ValueError('Only the zero relation branch may be new')
    return model


def model_from(state, initial=False):
    model = Model(state['settings']['width'], state['seed'])
    model.load_state_dict(state['initial_model'] if initial else state['model']); model.eval()
    return model


def extended(data, arm):
    if arm not in ARMS:
        raise ValueError('Fixed masked/relational contrast required')
    rel = data['relation'] if arm == 'relational' else torch.zeros_like(data['relation'])
    return dict(data, extra=torch.cat((data['extra'], rel), -1))


def fit(data, origin, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    parent.validate(data, origin)
    c = data['causal']; z = parent.kin.batch_features(extended(data, arm), 'kinematic'); model = initialize(origin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], origin['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm='kinematic', feature_arm=arm, settings=settings, identity=identity, seed=origin['seed'],
        preprocess=data['preprocess'], extra_preprocess=data['extra_preprocess'],
        relation_preprocess=data['relation_preprocess'], relation_hash=data['relation_hash'], relation_raw_hash=data['relation_raw_hash'],
        fit_ids=c['ids'], source_sites=sorted(set(c['sites'])), initial_model=copy.deepcopy(model.state_dict()),
        input_hash=origin['input_hash'], target_hash=origin['target_hash'], proposal_hash=origin['proposal_hash'],
        observed_mask_hash=origin['observed_mask_hash'], parameters=sum(p.numel() for p in model.parameters()),
        risk_budget=.02, budget=budget, checkpoint_selection='fixed_last_iterate_not_V',
        original_bounds_preserved=True, threshold_search=False, calibrated_safety_bound=False,
        objective='original_joint_decision', decision_semantics='sign_only_not_probability')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed, unknown = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        old = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(old[key], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed, unknown = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else stop_at
    if not 0 <= step <= limit <= budget or limit <= 0:
        raise ValueError('Nondecreasing fixed update budget required')
    started = time.monotonic()
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step, seconds=elapsed+time.monotonic()-started,
            trace=trace, draw_hash=draws, observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown))
    if not step:
        save()
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
        optimizer.zero_grad(set_to_none=True)
        terms = previous.previous.objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), 'joint')
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite original joint objective required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='rollout_history_training', feature_arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete registered query coverage required')
    return base.read_checkpoint(path)


def full(state, data, guard=lambda: None, initial=False):
    return parent.full(model_from(state, initial), extended(data, state['feature_arm']), state['sampler']['counts'], guard)


def attribution(state, data, guard=lambda: None):
    c = data['causal']; z = parent.kin.batch_features(extended(data, state['feature_arm']), 'kinematic')
    model = model_from(state); r, d = (np.empty((len(c['ids']), 7, 2), np.float32) for _ in range(2))
    for ix, seg in parent.batches(c):
        guard()
        with torch.no_grad():
            a, b = model(z[ix], torch.from_numpy(seg), int(seg.max())+1)
        r[ix], d[ix] = a.numpy(), b.numpy()
    y = (parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        torch.from_numpy(parent.kin.parent.normalizer(data['preprocess']))).numpy()
    w = parent.query_weights(c, state['sampler']['counts'])
    return previous.previous.attribute(r, d, y, c['allowed'], data['observed_future'], w)


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS or causal['x'].shape[1:] != (7, 428):
        raise ValueError('Exact causal inputs only; future labels forbidden')
    if state['risk_budget'] != .02 or state['feature_arm'] not in ARMS:
        raise ValueError('Original risk and registered model required')
    model = model_from(state); pr, ep, rp = (state[k] for k in ('preprocess', 'extra_preprocess', 'relation_preprocess'))
    r, d = (np.empty((len(causal['x']), 7, 2), float) for _ in range(2))
    for ix, seg in parent.batches(causal):
        raw = causal['x'][ix]
        z = np.clip((raw-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32); z[..., 380:] = 0
        extra = np.clip((parent.kin.motion_basis(raw)-ep['mean'])/ep['std'], -ep['clip'], ep['clip']).astype(np.float32)
        rel = np.clip((basis(raw)-rp['mean'])/rp['std'], -rp['clip'], rp['clip']).astype(np.float32)
        if state['feature_arm'] == 'masked':
            rel[:] = 0
        with torch.no_grad():
            a, b = model(torch.from_numpy(np.concatenate((z, extra, rel), -1)), torch.from_numpy(seg), int(seg.max())+1)
        r[ix] = a.numpy().astype(float)*parent.kin.parent.normalizer(pr)[None, None, :]*causal['scales'][ix, None, None]
        d[ix] = b.numpy()
    if not np.isfinite(r).all() or not np.isfinite(d).all():
        raise FloatingPointError('Finite neural risks and decisions required')
    return r, d


def assert_matched(states):
    if tuple(s['feature_arm'] for s in states) != ARMS:
        raise ValueError('Both matched arms required')
    for key in ('settings', 'identity', 'seed', 'initial_model', 'parameters', 'fit_ids', 'source_sites',
            'preprocess', 'extra_preprocess', 'relation_preprocess', 'relation_hash', 'relation_raw_hash',
            'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash', 'risk_budget', 'step',
            'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(states[0][key], states[1][key])
