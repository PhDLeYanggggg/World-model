"""Risk magnitude and intervention-sign heads; neither certifies safety."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_kinematic_risk as kin
from src.world_model.m3w_signed_decision import signed_cost
from src.world_model.m3w_frozen_readout_refit import query_weights
from src.evaluation.m3w_all_easy_gradient import batches, task_losses

np, torch, nn, F, base = kin.np, kin.torch, kin.nn, kin.F, kin.base
ARMS = ('shared64', 'shared128', 'separate64')


def representation(model, x, segments, count):
    if x.shape[1:] != (7, 492) or segments.shape != (len(x),):
        raise ValueError('Complete causal query groups required')
    code = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
    h = F.silu(model.encoder(torch.cat((x[..., :428], code), -1))+model.kinematics(x[..., 428:]))
    membership = F.one_hot(segments, count).to(h.dtype).T
    total = membership @ h.flatten(1); sizes = membership.T @ membership.sum(1)
    other = ((membership.T @ total).reshape_as(h)-h)/(sizes-1).clamp_min(1)[:, None, None]
    return F.silu(h+other)


class Model(nn.Module):
    def __init__(self, arm, width, seed):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Registered representation contrast required')
        self.arm = arm
        if arm == 'separate64':
            self.risk = kin.KinematicRiskHead(width, seed)
            self.decision = copy.deepcopy(self.risk)
        else:
            w = width*(2 if arm == 'shared128' else 1)
            self.shared = kin.KinematicRiskHead(w, seed)
            self.sign_head = nn.Linear(w, 2)

    def forward(self, x, segments, count):
        if self.arm == 'separate64':
            return self.risk(x, segments, count), self.decision(x, segments, count)
        h = representation(self.shared, x, segments, count)
        return self.shared.excess(h), self.sign_head(h)


def initialize(origin, arm):
    w = origin['settings']['width']; model = Model(arm, w, origin['seed']); state = origin['model']
    if arm == 'separate64':
        model.risk.load_state_dict(state); model.decision.load_state_dict(state)
    else:
        with torch.no_grad():
            # Widening preserves the initial function: extra features start with
            # zero outgoing weights, but their parameters remain trainable.
            for name in ('encoder.weight', 'encoder.bias', 'kinematics.weight'):
                model.shared.state_dict()[name][:w].copy_(state[name])
            model.shared.excess.weight.zero_()
            model.shared.excess.weight[:, :w].copy_(state['excess.weight'])
            model.shared.excess.bias.copy_(state['excess.bias'])
            model.sign_head.load_state_dict(model.shared.excess.state_dict())
    return model


def aggregate(error, allowed, segments, count):
    row = (error.mean(-1)*allowed).sum(1)/allowed.sum(1).clamp_min(1)
    supported = allowed.any(1)
    membership = F.one_hot(segments, count).to(error.dtype).T
    return ((membership @ (row*supported))/(membership @ supported.float()).clamp_min(1)).mean()


def objective(predictions, data, ix, segments, count):
    risk, decision = predictions
    allowed = torch.from_numpy(data['causal']['allowed'][ix])
    known = torch.from_numpy(data['observed_future'][ix])
    y = (kin.parent.signed(data['truth_by_arm']['bound_aware'][ix])/
        torch.from_numpy(kin.parent.normalizer(data['preprocess']))).detach()
    risk_loss = aggregate((risk-y).square(), allowed, segments, count)
    decision_loss = aggregate(torch.where(known[:, None, None], signed_cost(decision, y),
        (decision-y).square()), allowed, segments, count)
    supported = allowed.any(1)
    return dict(total=(risk_loss+decision_loss)/2, risk_MSE=risk_loss, decision_objective=decision_loss,
        observed_supervised_rows=int((supported & known).sum()),
        unknown_bound_supervised_rows=int((supported & ~known).sum()))


def combined(risk, decision):
    """Only its sign is meaningful: accept iff BOTH heads are non-positive."""
    return np.maximum(risk, decision)


def validate(data, origin):
    c = data['causal']; z = kin.batch_features(data, 'kinematic')
    if set(c) != base.CAUSAL_KEYS or origin['arm'] != 'kinematic':
        raise ValueError('Original causal schema required')
    for k, v in (('fit_ids', c['ids']), ('preprocess', data['preprocess']), ('extra_preprocess', data['extra_preprocess']),
            ('input_hash', kin.digest(z.numpy())), ('target_hash', kin.digest(data['truth_by_arm']['bound_aware'].numpy())),
            ('proposal_hash', kin.digest(c['allowed'])), ('observed_mask_hash', kin.digest(data['observed_future']))):
        base.exact(origin[k], v)
    return z


def fit(data, origin, *, arm, settings, identity, path, resume=False, stop_at=None,
        heartbeat=lambda **_: None, guard=lambda: None):
    z = validate(data, origin); c = data['causal']
    if settings['width'] != origin['settings']['width']:
        raise ValueError('Fixed base width required')
    model = initialize(origin, arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], origin['seed'])
    budget = base.steps_for(c, settings)
    binding = dict(arm='kinematic', variant=arm, settings=settings, identity=identity, seed=origin['seed'],
        preprocess=data['preprocess'], extra_preprocess=data['extra_preprocess'], fit_ids=c['ids'],
        source_sites=sorted(set(c['sites'])), initial_model=copy.deepcopy(model.state_dict()),
        input_hash=origin['input_hash'], target_hash=origin['target_hash'], proposal_hash=origin['proposal_hash'],
        observed_mask_hash=origin['observed_mask_hash'], parameters=sum(p.numel() for p in model.parameters()),
        risk_budget=.02, budget=budget, checkpoint_selection='fixed_last_iterate_not_V',
        original_bounds_preserved=True, threshold_search=False, calibrated_safety_bound=False,
        decision_semantics='signed_cost_logit_not_expected_risk_or_calibrated_probability')
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
        raise ValueError('Fixed nondecreasing budget required')
    started = time.monotonic()

    def save():
        guard(); base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, draw_hash=draws,
            observed_supervised_draws=observed, unknown_bound_supervised_draws=unknown))

    if not step:
        save()
    # Atomic completed-step checkpoints are the only recovery states. A signal
    # during a multi-parameter optimizer update must not serialize a partial step.
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
        optimizer.zero_grad(set_to_none=True)
        terms = objective(model(z[ix], segments, len(qids)), data, ix, segments, len(qids))
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Finite multi-task objective required')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        observed += terms['observed_supervised_rows']; unknown += terms['unknown_bound_supervised_rows']
        draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            row = dict(step=step, budget=budget, gradient_norm=float(norm),
                **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
            trace.append(row); heartbeat(state='dual_risk_training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Every fitting query needs full planned coverage')
    return base.read_checkpoint(path)


def model_from(state, initial=False):
    model = Model(state['variant'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['initial_model'] if initial else state['model']); model.eval()
    return model


def exact_state(a, b):
    for k in ('model', 'optimizer', 'sampler', 'torch_rng', 'step', 'trace', 'draw_hash', 'fit_ids', 'source_sites',
            'preprocess', 'extra_preprocess', 'initial_model', 'parameters', 'budget', 'observed_supervised_draws',
            'unknown_bound_supervised_draws'):
        base.exact(a[k], b[k])


def assert_matched(states):
    if tuple(s['variant'] for s in states) != ARMS:
        raise ValueError('All prespecified capacity controls required')
    for s in states[1:]:
        for k in ('settings', 'identity', 'seed', 'preprocess', 'extra_preprocess', 'fit_ids', 'source_sites',
                'input_hash', 'target_hash', 'proposal_hash', 'observed_mask_hash', 'risk_budget', 'budget',
                'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
            base.exact(s[k], states[0][k])


def full(model, data, counts, guard=lambda: None):
    c = data['causal']; z = kin.batch_features(data, 'kinematic'); weights = query_weights(c, counts)
    y = kin.parent.signed(data['truth_by_arm']['bound_aware'])/torch.from_numpy(kin.parent.normalizer(data['preprocess']))
    mse, sign, risk_sign, dec_sign, dec_loss = (np.zeros(2) for _ in range(5))
    false_safe = np.zeros(2, np.int64)
    for ix, seg in batches(c):
        guard()
        with torch.no_grad():
            r, d = model(z[ix], torch.from_numpy(seg), int(seg.max())+1)
            a = torch.from_numpy(c['allowed'][ix]); w = torch.from_numpy(weights[ix])
            mse += task_losses(r, y[ix], a, w).numpy()
            error = torch.where(torch.from_numpy(data['observed_future'][ix])[:, None, None],
                signed_cost(d, y[ix]), (d-y[ix]).square())
            dec_loss += ((error*a[..., None]).sum(1)/a.sum(1).clamp_min(1)[:, None]).double().mul(w[:, None]).sum(0).numpy()
        target = y[ix].numpy(); known = data['observed_future'][ix, None, None] & c['allowed'][ix, :, None]
        values = (r.numpy(), d.numpy(), combined(r.numpy(), d.numpy()))
        for value, out in zip(values, (risk_sign, dec_sign, sign)):
            wrong = ((value <= 0) & (target > 0)) | ((value > 0) & (target < 0))
            out += (weights[ix, None, None]*abs(target)*known*wrong).sum((0, 1))
        false_safe += (known & (values[-1] <= 0) & (target > 0)).sum((0, 1))
    return dict(mean_MSE=float(mse.mean()), task_MSE=mse.tolist(), observed_sign_regret=sign.tolist(),
        risk_only_sign_regret=risk_sign.tolist(), decision_only_sign_regret=dec_sign.tolist(),
        decision_objective=dec_loss.tolist(), false_safe_channels=false_safe.tolist())


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS or causal['x'].shape[-1] != 428:
        raise ValueError('Exact causal inputs only, never future supervision')
    model = model_from(state); pr, ep = state['preprocess'], state['extra_preprocess']
    risks, decisions = (np.empty((len(causal['x']), 7, 2), np.float64) for _ in range(2))
    for ix, seg in batches(causal):
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0
        extra = np.clip((kin.motion_basis(causal['x'][ix])-ep['mean'])/ep['std'], -ep['clip'], ep['clip']).astype(np.float32)
        with torch.no_grad():
            r, d = model(torch.from_numpy(np.concatenate((z, extra), -1)), torch.from_numpy(seg), int(seg.max())+1)
        risks[ix] = r.numpy().astype(float)*kin.parent.normalizer(pr)[None, None, :]*causal['scales'][ix, None, None]
        decisions[ix] = d.numpy()
    if not np.isfinite(risks).all() or not np.isfinite(decisions).all():
        raise FloatingPointError('Finite risk and decision scores required')
    return risks, decisions
