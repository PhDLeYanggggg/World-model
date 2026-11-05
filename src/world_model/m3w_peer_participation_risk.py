"""Complete the candidate-identity by peer-participation neural contrast."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_proposal_context_risk as previous

np, torch, nn, F, base = previous.np, previous.torch, previous.nn, previous.F, previous.base
parent = previous.parent
NEW_ARMS = ('same_active', 'proposed_all')
ARMS = ('same_action', 'same_active', 'proposed_action', 'proposed_all')
extended, validate, proposal_from_mask, proposal_mask = previous.extended, previous.validate, previous.proposal_from_mask, previous.proposal_mask


def added_contexts(h, mask, segments, count):
    membership = F.one_hot(segments, count).to(h.dtype).T
    active = mask.sum(1)
    weighted = h*active[:, None, None]
    active_count = membership.T @ (membership @ active)-active
    same_active = ((membership.T @ (membership @ weighted.flatten(1))).reshape_as(h)-weighted)/active_count.clamp_min(1)[:, None, None]
    selected = (h*mask[..., None]).sum(1)+h[:, 0]*(1-active[:, None])
    sizes = membership.T @ membership.sum(1)
    proposed_all = (membership.T @ (membership @ selected)-selected)/(sizes-1).clamp_min(1)[:, None]
    return same_active, proposed_all[:, None, :].expand_as(h)


class Head(previous.Head):
    def __init__(self, width, seed, arm):
        if arm not in ARMS:
            raise ValueError('Registered candidate/participation arm required')
        super().__init__(width, seed, arm if arm in previous.ARMS else 'same_action')
        self.factor_arm = arm

    def forward(self, x, segments, count):
        if self.factor_arm in previous.ARMS:
            return super().forward(x, segments, count)
        if x.shape[1:] != (7, 525) or segments.shape != (len(x),):
            raise ValueError('Causal525 and complete queries required')
        mask = x[..., 524]
        if not ((mask == 0) | (mask == 1)).all() or (mask.sum(1) > 1).any() or mask[:, 0].any():
            raise ValueError('Binary explicit non-fallback proposal mask required')
        code = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        h = F.silu(self.encoder(torch.cat((x[..., :428], code), -1))+
            self.kinematics(x[..., 428:492])+self.relation(x[..., 492:524]))
        same, _ = previous.peer_context(h, mask, segments, count)
        active, complete = added_contexts(h, mask, segments, count)
        context = active if self.factor_arm == 'same_active' else complete
        return self.excess(F.silu(h+same+self.bridge(context)))


class Model(nn.Module):
    def __init__(self, width, seed, arm):
        super().__init__()
        self.risk = Head(width, seed, arm); self.decision = copy.deepcopy(self.risk)

    def forward(self, x, segments, count):
        return self.risk(x, segments, count), self.decision(x, segments, count)


def initialize(warm, arm):
    model = Model(warm['settings']['width'], warm['seed'], arm)
    missing = model.load_state_dict(warm['model'], strict=False)
    if missing.missing_keys != ['risk.bridge.weight', 'decision.bridge.weight'] or missing.unexpected_keys:
        raise ValueError('Only two zero bridges may be new')
    return model


def model_from(state, initial=False):
    model = Model(state['settings']['width'], state['seed'], state['context_arm'])
    model.load_state_dict(state['initial_model'] if initial else state['model']); model.eval()
    return model


def fit(data, warm, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    if arm not in ARMS or settings['width'] != warm['settings']['width']:
        raise ValueError('Matched peer-participation continuation required')
    z = validate(data, warm); c = data['causal']; model = initialize(warm, arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = base.steps_for(c, settings)
    fields = ('seed', 'preprocess', 'extra_preprocess', 'relation_preprocess', 'relation_hash', 'relation_raw_hash',
        'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash', 'fit_ids', 'source_sites')
    binding = dict(**{k:warm[k] for k in fields}, arm='kinematic', feature_arm='relational', context_arm=arm,
        settings=settings, identity=identity, initial_model=copy.deepcopy(model.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()), parent_step=warm['step'],
        parent_model_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in warm['model'].values())).hexdigest(),
        optimizer_reused=False, risk_budget=.02, budget=budget, checkpoint_selection='fixed_last_iterate_not_V',
        objective='unchanged_original_joint', original_bounds_preserved=True, threshold_search=False,
        calibrated_safety_bound=False, decision_semantics='joint_sign_not_probability',
        context_semantics='frozen_proposal_not_final_admission')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed, unknown = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        old = base.read_checkpoint(path)
        for k, v in binding.items():
            base.exact(old[k], v)
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
        terms = parent.previous.previous.objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids), 'joint')
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite original joint loss required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k:float(v.detach()) if isinstance(v, torch.Tensor) else v for k,v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='peer_participation_training', context_arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Every registered query must receive coverage')
    return base.read_checkpoint(path)


def full(state, data, guard=lambda: None, initial=False):
    return parent.parent.full(model_from(state, initial), extended(data), state['sampler']['counts'], guard)


def normalized_predictions(state, data, guard=lambda: None):
    c = data['causal']; z = validate(data, state); model = model_from(state)
    r, d = (np.empty((len(c['ids']), 7, 2), np.float32) for _ in range(2))
    for ix, seg in parent.parent.batches(c):
        guard()
        with torch.no_grad():
            a, b = model(z[ix], torch.from_numpy(seg), int(seg.max())+1)
        r[ix], d[ix] = a.numpy(), b.numpy()
    return r, d


def attribution(state, data, guard=lambda: None):
    c = data['causal']; r, d = normalized_predictions(state, data, guard)
    y = (parent.parent.kin.parent.signed(data['truth_by_arm']['bound_aware'])/
        torch.from_numpy(parent.parent.kin.parent.normalizer(data['preprocess']))).numpy()
    w = parent.parent.query_weights(c, state['sampler']['counts'])
    return parent.previous.previous.attribute(r, d, y, c['allowed'], data['observed_future'], w)


def predict(state, causal, proposal):
    if set(causal) != base.CAUSAL_KEYS or causal['x'].shape[1:] != (7, 428):
        raise ValueError('Exact causal inputs only; future labels forbidden')
    if state['risk_budget'] != .02:
        raise ValueError('Original risk budget required')
    bit = proposal_mask(causal, proposal)
    model = model_from(state); pr, ep, rp = (state[k] for k in ('preprocess', 'extra_preprocess', 'relation_preprocess'))
    r, d = (np.empty((len(causal['x']), 7, 2), float) for _ in range(2))
    for ix, seg in parent.parent.batches(causal):
        raw = causal['x'][ix]
        z = np.clip((raw-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32); z[..., 380:] = 0
        extra = np.clip((parent.parent.kin.motion_basis(raw)-ep['mean'])/ep['std'], -ep['clip'], ep['clip']).astype(np.float32)
        rel = np.clip((parent.basis(raw)-rp['mean'])/rp['std'], -rp['clip'], rp['clip']).astype(np.float32)
        with torch.no_grad():
            a, b = model(torch.from_numpy(np.concatenate((z, extra, rel, bit[ix, :, None]), -1)),
                torch.from_numpy(seg), int(seg.max())+1)
        r[ix] = a.numpy().astype(float)*parent.parent.kin.parent.normalizer(pr)[None, None, :]*causal['scales'][ix, None, None]
        d[ix] = b.numpy()
    if not np.isfinite(r).all() or not np.isfinite(d).all():
        raise FloatingPointError('Finite risk and sign predictions required')
    return r, d


def assert_matched(states):
    if {s['context_arm'] for s in states} != set(ARMS) or len(states) != 4:
        raise ValueError('Complete two-by-two model comparison required')
    for s in states[1:]:
        for k in ('settings', 'seed', 'initial_model', 'parameters', 'fit_ids', 'source_sites',
            'preprocess', 'extra_preprocess', 'relation_preprocess', 'relation_hash', 'relation_raw_hash',
            'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash', 'risk_budget', 'step',
            'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws', 'parent_model_hash'):
            base.exact(states[0][k], s[k])
        base.exact(states[0]['identity']['job'], s['identity']['job'])


def screen(rows, jobs):
    if len(rows) != 4*len(jobs) or len({(r['job'], r['arm']) for r in rows}) != len(rows):
        raise ValueError('All four arms in every context required')
    out = []
    for job in jobs:
        a, b = (next(r for r in rows if r['job'] == job['key'] and r['arm'] == arm) for arm in ('same_action', 'proposed_all'))
        old, new = a['joint_attribution']['total_joint_cost'], b['joint_attribution']['total_joint_cost']
        gain = 1-new/old if old > 0 else None
        harm = b['joint_attribution']['false_safe_cost'] <= a['joint_attribution']['false_safe_cost']
        mse = b['final_fitting']['mean_MSE'] <= a['final_fitting']['mean_MSE']
        out.append(dict(job=job['key'], joint_cost_reduction=gain, false_safe_cost_nonworse=bool(harm),
            risk_MSE_nonworse=bool(mse), passed=bool(gain is not None and gain >= .01 and harm and mse)))
    return dict(contexts=out, followup_readout_allowed=sum(r['passed'] for r in out) >= 2,
        scientific_safety_pass=False, threshold_search=False, primary='proposed_all_minus_same_action')
