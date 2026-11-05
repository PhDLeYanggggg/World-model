"""Direct budget-excess learning; neither output is a calibrated safety bound."""
import copy
import hashlib
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_proposal_conditioned_risk as parent

base = parent.base
ARMS = ('mean', 'quantile90')


def signed(moments):
    return moments[..., [1, 4]] - .02*moments[..., [2, 3]]


def normalizer(pr):
    rms = np.asarray(pr['rms'])
    if rms.shape != (5,) or not np.isfinite(rms).all() or (rms <= 0).any():
        raise ValueError('Own-fitting positive moment normalizers required')
    return np.sqrt(rms[[1, 4]]**2 + (.02*rms[[2, 3]])**2).astype(np.float32)


class DirectRiskHead(nn.Module):
    def __init__(self, dimension, width, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(dimension+7, width)
        self.excess = nn.Linear(width, 2)
        nn.init.zeros_(self.excess.weight)
        nn.init.zeros_(self.excess.bias)

    def forward(self, x, segments, count):
        if x.ndim != 3 or x.shape[1] != 7 or segments.shape != (len(x),):
            raise ValueError('Complete seven-action causal query batches required')
        code = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        hidden = F.silu(self.encoder(torch.cat((x, code), -1)))
        membership = F.one_hot(segments, count).to(hidden.dtype).T
        total = membership @ hidden.flatten(1)
        sizes = membership.T @ membership.sum(1)
        other = ((membership.T @ total).reshape_as(hidden)-hidden)/(sizes-1).clamp_min(1)[:, None, None]
        return self.excess(F.silu(hidden+other))


def objective(prediction, data, ix, segments, count, arm):
    if arm not in ARMS:
        raise ValueError('Fixed matched mean or quantile90 objective required')
    legal = torch.from_numpy(data['causal']['allowed'][ix])
    truth = signed(data['truth_by_arm']['bound_aware'][ix])/torch.from_numpy(normalizer(data['preprocess']))
    residual = truth.detach()-prediction
    error = residual.square() if arm == 'mean' else torch.maximum(.9*residual, -.1*residual)
    row = (error.mean(-1)*legal).sum(1)/legal.sum(1).clamp_min(1)
    supported = legal.any(1)
    membership = F.one_hot(segments, count).to(prediction.dtype).T
    loss = ((membership @ (row*supported))/(membership @ supported.float()).clamp_min(1)).mean()
    known = torch.from_numpy(data['observed_future'][ix])
    return dict(total=loss, observed_supervised_rows=int((supported & known).sum()),
        unknown_bound_supervised_rows=int((supported & ~known).sum()))


def fit(data, *, arm, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS or set(data['causal']) != base.CAUSAL_KEYS:
        raise ValueError('Registered objective and exact causal input schema required')
    c, pr = data['causal'], data['preprocess']
    model = DirectRiskHead(c['x'].shape[-1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm=arm, seed=seed, preprocess=pr,
        initial_model=initial, parameters=sum(p.numel() for p in model.parameters()), budget=budget,
        source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy(), risk_budget=.02,
        input_hash=hashlib.sha256(data['z'].numpy().tobytes()).hexdigest(),
        target_hash=hashlib.sha256(data['truth_by_arm']['bound_aware'].numpy().tobytes()).hexdigest(),
        proposal_mask_hash=hashlib.sha256(c['allowed'].tobytes()).hexdigest())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, observed_draws, bound_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit same-binding resume required')
        old = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(old[key], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, observed_draws, bound_draws = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('No rollback or zero training budget')
    started = time.monotonic()

    def save():
        guard()
        base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, draw_hash=draws,
            observed_supervised_draws=observed_draws, unknown_bound_supervised_draws=bound_draws,
            calibrated_safety_bound=False, threshold_search=False))

    if not step:
        save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segment = torch.from_numpy(seg)
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model(data['z'][ix], segment, len(qids)), data, ix, segment, len(qids), arm)
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Nonfinite direct risk objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            observed_draws += terms['observed_supervised_rows']; bound_draws += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, gradient_norm=float(norm),
                    **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='direct_tail_training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Incomplete original source-query coverage')
    return base.read_checkpoint(path)


def assert_matched(a, b):
    if (a['arm'], b['arm']) != ARMS:
        raise ValueError('Both prespecified loss arms required')
    for key in ('identity', 'settings', 'seed', 'preprocess', 'initial_model', 'parameters',
            'budget', 'source_sites', 'fit_ids', 'input_hash', 'target_hash', 'proposal_mask_hash',
            'step', 'sampler', 'draw_hash', 'observed_supervised_draws', 'unknown_bound_supervised_draws'):
        base.exact(a[key], b[key])


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS or causal['x'].shape[-1] != 428:
        raise ValueError('Only declared causal schema, not future labels, may enter inference')
    if state['arm'] not in ARMS or state['risk_budget'] != .02:
        raise ValueError('Frozen risk head and original budget required')
    pr = state['preprocess']
    model = DirectRiskHead(428, state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    result = np.empty((len(causal['x']), 7, 2), np.float64)
    pending, size = [], 0

    def flush():
        ix = np.concatenate(pending)
        seg = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0
        with torch.no_grad():
            p = model(torch.from_numpy(z), torch.from_numpy(seg), len(pending)).numpy()
        result[ix] = p.astype(float)*normalizer(pr)[None, None, :]*causal['scales'][ix, None, None]

    for group in causal['groups']:
        if pending and size+len(group) > 1024:
            flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending:
        flush()
    if not np.isfinite(result).all():
        raise FloatingPointError('Finite signed risk outputs required')
    return result


def choose(logits, point_costs, excess, causal):
    proposal, utility = parent.router.choose(logits, causal['allowed'])
    p, e = np.asarray(point_costs), np.asarray(excess)
    if (p.shape != (len(proposal), 7, 5) or e.shape != (len(proposal), 7, 2)
            or not np.isfinite(p).all() or not np.isfinite(e).all() or (p < 0).any()):
        raise ValueError('Frozen support costs and finite signed risks required')
    rows = np.flatnonzero(proposal >= 0)
    costs, risks = p[rows, proposal[rows]], e[rows, proposal[rows]]
    keep = (costs[:, 2] > 0) & (costs[:, 3] > 0) & (risks <= 0).all(1)
    result = np.full(len(proposal), -1, np.int64)
    result[rows[keep]] = proposal[rows[keep]]
    matched = parent.matched_self(proposal, utility, result, causal['groups'], causal['ids'])
    return result, matched
