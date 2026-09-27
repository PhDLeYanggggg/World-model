"""Paired marginal versus explicitly supervised easy-risk factorization."""
import hashlib
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before Torch import')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_query_excess as sampling
from src.world_model.m3w_query_utility import grouped

head = sampling.head
ARMS = ('marginal', 'supervised')
BUDGET = .02


def targets(cv, reference, candidate, easy_cut, scale):
    cv, reference, candidate = map(lambda a: np.asarray(a, float), (cv, reference, candidate))
    known = np.isfinite(cv)
    if (cv.ndim != 1 or reference.shape != cv.shape or candidate.shape != cv.shape
            or not np.array_equal(known, np.isfinite(reference))
            or not np.array_equal(known, np.isfinite(candidate)) or scale <= 0
            or not np.isfinite(scale) or any((a[known] < 0).any() for a in (cv, reference, candidate))):
        raise ValueError('Aligned nonnegative label-only costs and fitting scale required')
    easy = known & (cv > 0) & (cv <= easy_cut)
    result = np.column_stack((easy.astype(float), reference/scale,
                              np.maximum(candidate-reference, 0)/scale))
    result[~known] = np.nan
    return result


class EasyHead(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, x, envelope):
        raw = self.network(x)
        probability = torch.sigmoid(raw[:, 0])
        reference = F.softplus(raw[:, 1])
        harm = envelope*torch.sigmoid(raw[:, 2])
        risk = probability*(harm-BUDGET*reference)
        return torch.stack((probability, reference, harm, risk, raw[:, 0]), 1)


def query_mean(values, segments, count):
    sizes = torch.bincount(segments, minlength=count).to(values.dtype)
    if (sizes == 0).any():
        raise ValueError('Empty fitting query')
    return values.new_zeros(count).index_add(0, segments, values)/sizes


def losses(pred, target, segments, count):
    if (pred.shape != (len(target), 5) or target.shape[1:] != (3,)
            or not torch.isfinite(target).all() or (target < 0).any()
            or not ((target[:, 0] == 0) | (target[:, 0] == 1)).all()):
        raise ValueError('Known fitting labels only; easy occurrence must be binary')
    easy, reference, harm = target.unbind(1)
    truth = easy*(harm-BUDGET*reference)
    error = pred[:, 3]-truth
    marginal = .5*query_mean(error.square(), segments, count).mean()
    marginal = marginal+.5*query_mean(error, segments, count).square().mean()
    occurrence = query_mean(F.binary_cross_entropy_with_logits(pred[:, 4], easy, reduction='none'), segments, count).mean()
    mass = query_mean(easy, segments, count).mean()
    conditional = query_mean(easy*.5*((pred[:, 1]-reference).square()+(pred[:, 2]-harm).square()), segments, count).mean()/mass.clamp_min(1e-12)
    return dict(marginal=marginal, occurrence=occurrence, conditional=conditional,
                supervised=marginal+occurrence+conditional)


def preprocess(x, u, envelope, pr):
    upr = head.descriptor_preprocess(u, pr)
    norm = dict(mean=np.concatenate((pr['mean'], upr['mean'])),
                std=np.concatenate((pr['std'], upr['std'])), clip=float(pr['clip']),
                cost_scale=float(pr['cost_scale']), training_sites=list(pr['training_sites']))
    z, env = inputs(x, u, envelope, norm)
    return z, env, norm


def inputs(x, u, envelope, norm):
    x, u, envelope = map(np.asarray, (x, u, envelope))
    if (x.ndim != 2 or u.shape != (len(x), 6) or envelope.shape != (len(x),)
            or not all(np.isfinite(a).all() for a in (x, u, envelope)) or (envelope < 0).any()):
        raise ValueError('Finite causal features/descriptors/envelope only')
    joined = np.concatenate((x, u), 1).astype(float)
    z = np.clip((joined-norm['mean'])/norm['std'], -norm['clip'], norm['clip']).astype(np.float32)
    return torch.from_numpy(z), torch.from_numpy((envelope/norm['cost_scale']).astype(np.float32))


def initialize(inputs_count, width, seed):
    torch.manual_seed(seed)
    return EasyHead(inputs_count, width)


def fit(x, u, target, sites, recordings, frames, envelope, pr, *, arm, seed,
        settings, identity, path, heartbeat, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Unregistered objective')
    known = np.isfinite(target).all(1)
    if not np.array_equal(known, pr['known']) or set(sites) != set(pr['training_sites']):
        raise ValueError('Only matching fitting-source roles and masks allowed')
    if not (target[known, 0] == 1).any():
        raise ValueError('No fitting easy labels: conditional head unsupported, no training')
    z, env, norm = preprocess(x, u, envelope, pr)
    if (target[known, 2] > env.numpy()[known]+1e-5).any():
        raise ValueError('Observed positive harm exceeds causal envelope')
    groups, source_groups, keys = sampling.query_groups(sites, recordings, frames, known)
    model = initialize(z.shape[1], settings['width'], seed)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    truth = torch.from_numpy(np.where(known[:, None], target, 0).astype(np.float32))
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = sampling.draw_queries(groups, source_groups, 128, torch.Generator().manual_seed(seed+9137))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, sample_hash = 0, 0., [], '0'*64
    query_draws, row_draws = 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Existing head requires resume')
        s = head.read_checkpoint(path)
        for k, v in dict(identity=identity, settings=settings, arm=arm, seed=seed, norm=norm, initial_model=initial).items():
            sampling.exact(s[k], v)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, seconds, trace, sample_hash, query_draws, row_draws = (s[k] for k in
            ('step', 'seconds', 'trace', 'sample_hash', 'query_draws', 'row_draws'))
    start = time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing training budget required')

    def monitor_loss():
        ix, seg, _ = monitor
        with torch.no_grad():
            return {k: float(v) for k, v in losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), 128).items()}

    if not trace:
        trace.append(dict(step=0, monitor=monitor_loss()))

    def save():
        head.save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, seed=seed, norm=norm,
            initial_model=initial, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=seconds+time.monotonic()-start, trace=trace, sample_hash=sample_hash,
            query_draws=query_draws, row_draws=row_draws, unknown_rows_sampled=0,
            fitting_queries=len(keys), probability_calibration_certificate=False))

    model.train()
    while step < limit:
        ix, seg, qids = sampling.draw_queries(groups, source_groups, settings['query_batch_size'], rng)
        optimizer.zero_grad(set_to_none=True)
        objective = losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), settings['query_batch_size'])[arm]
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite easy-risk objective')
        objective.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        sample_hash = hashlib.sha256(bytes.fromhex(sample_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        query_draws += len(qids); row_draws += len(ix)
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(objective.detach()), gradient_norm=float(grad), monitor=monitor_loss())
            trace.append(row); heartbeat(state='training', **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return head.read_checkpoint(path)


def predict(state, x, u, envelope):
    z, env = inputs(x, u, envelope, state['norm'])
    model = initialize(z.shape[1], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(z), 4096):
            values.append(model(z[start:start+4096], env[start:start+4096]).numpy())
    return np.concatenate(values)


def assert_matched(a, b):
    for k in ('settings', 'seed', 'norm', 'initial_model', 'sampler_rng', 'torch_rng',
              'step', 'sample_hash', 'query_draws', 'row_draws', 'fitting_queries'):
        sampling.exact(a[k], b[k])
    assert a['arm'] == 'marginal' and b['arm'] == 'supervised'


def decisions(utility, risks, eligible, recordings, frames, ids, node_limit=256):
    if set(risks) != {'raw', *ARMS}:
        raise ValueError('Raw and both registered risk arms required')
    anchors = {a: eligible & (q <= 0).all(1) for a, q in risks.items()}
    common = np.logical_and.reduce(list(anchors.values()))
    out, stats = dict(common_anchor=common), {}
    for a, q in risks.items():
        if q.shape != (len(eligible), 2) or not np.isfinite(q).all():
            raise ValueError('Finite causal signed risk scores required')
        out[a+'_independent'] = anchors[a]
        for suffix, anchor in (('joint', anchors[a]), ('matched', common)):
            actions, info = grouped(utility, q, eligible, anchor, recordings, frames, ids, node_limit=node_limit)
            out[a+'_'+suffix] = actions['joint_utility']; stats[a+'_'+suffix] = info
    return out, stats


def quality(pred, target, recordings, frames):
    known = np.isfinite(target).all(1)
    groups, _, _ = sampling.query_groups(np.repeat('held', len(target)), recordings, frames, known)
    pi = pred[:, 0].astype(float).clip(1e-7, 1-1e-7)
    easy, reference, harm = target.T
    truth = easy*(harm-BUDGET*reference)
    qm = lambda a: float(np.mean([np.mean(a[g]) for g in groups]))
    rate = qm(easy)
    return dict(easy_rate=rate, Brier=qm((pi-easy)**2),
        log_loss=qm(-easy*np.log(pi)-(1-easy)*np.log1p(-pi)),
        signed_MSE=qm((pred[:, 3]-truth)**2), signed_bias=qm(pred[:, 3]-truth),
        conditional_reference_MSE=qm(easy*(pred[:, 1]-reference)**2)/rate if rate else None,
        conditional_harm_MSE=qm(easy*(pred[:, 2]-harm)**2)/rate if rate else None,
        predicted_easy_probability=qm(pi), evaluable_queries=len(groups))
