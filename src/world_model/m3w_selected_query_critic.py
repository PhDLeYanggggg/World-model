"""Matched risk recalibrators conditioned on a frozen selected-action set."""
import copy
import hashlib
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model.m3w_multiaction_cost import read_checkpoint, save_checkpoint, exact
from src.world_model.m3w_rational_harm import bounded_fraction
from src.world_model.m3w_query_excess import query_groups, draw_queries

ARMS = ('row_local', 'selected_set')
CAUSAL_KEYS = frozenset(('features', 'prediction', 'envelope', 'scales', 'allowed', 'initial',
    'sites', 'recordings', 'frames', 'ids', 'groups', 'source_groups', 'keys'))


def digest(value):
    value = np.asarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode()+np.ascontiguousarray(value).tobytes()).hexdigest()


def feature_rows(features, rows):
    value = features[rows]
    return value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))


def causal_inputs(features, prediction, envelope, scales, allowed, initial,
                  sites, recordings, frames, ids):
    p, env, scales = map(lambda v: np.asarray(v, float), (prediction, envelope, scales))
    allowed, initial, frames, ids = map(np.asarray, (allowed, initial, frames, ids))
    sites, recs = np.asarray(sites, str), np.asarray(recordings, str)
    n = len(p)
    if (p.shape != (n, 7, 5) or features.shape[:2] != (n, 7) or len(features.shape) != 3
            or env.shape != (n, 7) or scales.shape != (n,) or allowed.shape != (n, 7)
            or allowed.dtype != bool or initial.shape != (n,)
            or not np.issubdtype(initial.dtype, np.integer) or ((initial < -1) | (initial >= 7)).any()
            or any(v.shape != (n,) for v in (sites, recs, frames, ids)) or len(np.unique(ids)) != n
            or not all(np.isfinite(v).all() for v in (p, env, scales, frames))
            or (p < 0).any() or (env < 0).any() or (scales <= 0).any()
            or (p[:, :, :2].sum(2) > env+1e-5).any() or (p[:, :, 4] > p[:, :, 1]+1e-7).any()):
        raise ValueError('Aligned past-only seven-action costs/features and frozen choices required')
    at = np.flatnonzero(initial >= 0)
    if not allowed[at, initial[at]].all():
        raise ValueError('The frozen selected set must use real available actions')
    groups, sources, keys = query_groups(sites, recs, frames, np.ones(n, bool))
    groups = [g[np.argsort(ids[g], kind='stable')] for g in groups]
    return dict(features=features, prediction=p, envelope=env, scales=scales, allowed=allowed,
        initial=initial, sites=sites, recordings=recs, frames=frames, ids=ids,
        groups=groups, source_groups=sources, keys=keys)


def hashes(causal):
    if set(causal) != CAUSAL_KEYS:
        raise ValueError('Only the declared causal fields are allowed')
    first = feature_rows(causal['features'], np.arange(min(512, len(causal['ids'])))).numpy()
    h = hashlib.sha256(str((tuple(causal['features'].shape), first.dtype.str)).encode())
    for start in range(0, len(causal['ids']), 512):
        v = feature_rows(causal['features'], np.arange(start, min(start+512, len(causal['ids']))))
        if not torch.isfinite(v).all():
            raise ValueError('Finite frozen normalized causal features required')
        h.update(np.ascontiguousarray(v.numpy()).tobytes())
    return dict(features=h.hexdigest(), **{k: digest(causal[k]) for k in
        ('prediction', 'envelope', 'scales', 'allowed', 'initial', 'sites', 'recordings', 'frames', 'ids')})


def training_data(causal, targets):
    y = np.asarray(targets, float); n = len(causal['ids'])
    if y.shape != (n, 7, 5):
        raise ValueError('Separate five-moment supervision required')
    known = np.isfinite(y).all((1, 2))
    if (not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any()
            or (y[known, :, 1] > causal['envelope'][known]+1e-5).any()
            or (y[known, :, 4] > y[known, :, 1]+1e-7).any()):
        raise ValueError('Unknown labels remain NaN; observed costs must be coherent')
    means = []
    for source in sorted(set(causal['sites'])):
        mask = (causal['sites'] == source) & known
        legal = causal['allowed'][mask]
        value = y[mask]/causal['scales'][mask, None, None]
        if legal.any():
            means.append(np.square(value[legal]).mean(0))
    if not means:
        raise ValueError('No real known fitting action costs')
    rms = np.sqrt(np.mean(means, 0)).clip(1e-6)
    return dict(causal=causal, targets=y, known=known, rms=rms,
        input_hashes=dict(hashes(causal), targets=digest(y)),
        known_supervised_rows=int(known.sum()), unknown_rows=int((~known).sum()))


def selected_context(hidden, initial, segments, count):
    selected = initial >= 0
    chosen = hidden[torch.arange(len(hidden)), initial.clamp_min(0)]*selected[:, None]
    totals = hidden.new_zeros((count, hidden.shape[-1])).index_add(0, segments, chosen)
    sizes = hidden.new_zeros(count).index_add(0, segments, selected.to(hidden.dtype))
    return (totals/sizes.clamp_min(1)[:, None])[segments], sizes[segments]


class QueryCostCritic(nn.Module):
    def __init__(self, dimension, width, seed):
        super().__init__(); torch.manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(dimension+13, width), nn.SiLU())
        self.output = nn.Linear(2*width+2, 2)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, base, envelope, initial, segments, count, *, arm):
        if arm not in ARMS or x.shape[:2] != base.shape[:2] or base.shape[1:] != (7, 5):
            raise ValueError('Registered arm and aligned causal candidate inputs required')
        onehot = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        h = self.encoder(torch.cat((x, torch.log1p(base.clamp_max(2980)).to(x.dtype),
            onehot, torch.log1p(envelope.clamp_max(2980)).to(x.dtype)[:, :, None]), -1))
        context, selected = selected_context(h, initial, segments, count)
        size = torch.bincount(segments, minlength=count).to(x.dtype)[segments]
        metadata = torch.stack((torch.log1p(selected), selected/size), 1)[:, None].expand(-1, 7, -1)
        peer = h if arm == 'row_local' else context[:, None].expand_as(h)
        offsets = self.output(torch.cat((h, peer, metadata), -1)).double()
        b = base.double(); cap = torch.maximum((envelope.double()-b[:, :, 0]).clamp_min(0), b[:, :, 1])
        fraction = b[:, :, 1]/cap.clamp_min(1e-150)
        harm = b[:, :, 1]+cap*(bounded_fraction(fraction, offsets[:, :, 0])-fraction)
        harm = torch.minimum(harm.clamp_min(0), cap)
        easy_fraction = (b[:, :, 4]/b[:, :, 1].clamp_min(1e-150)).clamp(0, 1)
        easy = b[:, :, 4]+(harm*bounded_fraction(easy_fraction, offsets[:, :, 1])-b[:, :, 1]*easy_fraction)
        easy = torch.minimum(easy.clamp_min(0), harm)
        return torch.stack((b[:, :, 0], harm, b[:, :, 2], b[:, :, 3], easy), -1)


def initialize(dimension, width, seed):
    return QueryCostCritic(dimension, width, seed)


def batch(causal, rows, segments, count):
    rows = np.asarray(rows)
    return (feature_rows(causal['features'], rows).float(),
        torch.from_numpy(causal['prediction'][rows]/causal['scales'][rows, None, None]),
        torch.from_numpy(causal['envelope'][rows]/causal['scales'][rows, None]),
        torch.from_numpy(causal['initial'][rows]), torch.as_tensor(segments, dtype=torch.int64), count)


def losses(prediction, data, rows, segments, count):
    c = data['causal']; known = torch.from_numpy(data['known'][rows]); scales = torch.from_numpy(c['scales'][rows])
    target = torch.from_numpy(np.where(data['known'][rows, None, None],
        data['targets'][rows]/c['scales'][rows, None, None], 0.))
    real = torch.from_numpy(c['allowed'][rows]) & known[:, None]
    rms = torch.from_numpy(data['rms'])
    error = ((prediction[:, :, [1, 4]]-target[:, :, [1, 4]])/rms[[1, 4]]).square().mean(2)
    row_error = (error*real).sum(1)/real.sum(1).clamp_min(1)
    valid_row = real.any(1)
    q_error = prediction.new_zeros(count).index_add(0, segments, row_error)
    q_known = prediction.new_zeros(count).index_add(0, segments, valid_row.to(prediction.dtype))
    point = (q_error/q_known.clamp_min(1)).mean()
    selected = torch.from_numpy(c['initial'][rows] >= 0)
    action = torch.from_numpy(c['initial'][rows]).clamp_min(0); ix = torch.arange(len(rows))
    pred, truth = prediction[ix, action], target[ix, action]
    signed_error = pred[:, [1, 4]]-.02*pred[:, [2, 3]]-truth[:, [1, 4]]+.02*truth[:, [2, 3]]
    raw_error = signed_error*scales[:, None]*selected[:, None]
    total = prediction.new_zeros((count, 2)).index_add(0, segments, raw_error)
    mass = prediction.new_zeros(count).index_add(0, segments, scales*selected)
    selected_n = prediction.new_zeros(count).index_add(0, segments, selected.to(prediction.dtype))
    missing_n = prediction.new_zeros(count).index_add(0, segments, (selected & ~known).to(prediction.dtype))
    complete = (selected_n > 0) & (missing_n == 0)
    score_rms = torch.sqrt(rms[[1, 4]].square()+(.02*rms[[2, 3]]).square())
    query = ((total/mass.clamp_min(1e-150)[:, None]/score_rms).square().mean(1)*complete).mean()
    return dict(total=.5*(point+query), pointwise_risk=point, selected_query_score=query,
        unknown_supervised_rows=int((real & ~known[:, None]).sum()),
        complete_selected_queries=int(complete.sum()))


def fit(data, *, arm, seed, settings, path, identity, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Registered critic arm required')
    c = data['causal']; dimension = c['features'].shape[-1]
    model = initialize(dimension, settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'],
        torch.Generator().manual_seed(seed+9137))
    binding = dict(arm=arm, seed=seed, settings=settings, identity=identity, input_hashes=data['input_hashes'],
        rms=data['rms'], initial_model=initial, dimension=dimension,
        parameters=sum(p.numel() for p in model.parameters()), query_keys=c['keys'],
        all_queries_include_unknown_rows=True, original_benefit_references_frozen=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draw_hash, row_draws, unknown_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Existing critic requires explicit resume')
        old = read_checkpoint(path)
        for k, value in binding.items(): exact(old[k], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']); torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, draw_hash, row_draws, unknown_draws = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws', 'unknown_rows_drawn'))
    limit = settings['steps'] if stop_at is None else min(settings['steps'], stop_at)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('Nondecreasing registered update budget required')
    began = time.monotonic()

    def diagnostic():
        ix, seg, _ = monitor; args = batch(c, ix, seg, settings['query_batch_size'])
        with torch.no_grad():
            terms = losses(model(*args, arm=arm), data, ix, args[-2], settings['query_batch_size'])
        return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in terms.items()}

    def save():
        guard()
        save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=seconds+time.monotonic()-began, trace=trace, draw_hash=draw_hash, row_draws=row_draws,
            unknown_rows_drawn=unknown_draws, unknown_rows_with_loss=0,
            known_supervised_rows=data['known_supervised_rows'], unknown_rows=data['unknown_rows']))

    if not trace: trace.append(dict(step=0, monitor=diagnostic()))
    while step < limit:
        ix, seg, qids = draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'], rng)
        args = batch(c, ix, seg, settings['query_batch_size'])
        optimizer.zero_grad(set_to_none=True)
        terms = losses(model(*args, arm=arm), data, ix, args[-2], settings['query_batch_size'])
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']:
            raise FloatingPointError('Nonfinite objective or future-unknown supervised row')
        telemetry = {}
        if step == 0 or (step+1) % settings['heartbeat_every'] == 0 or step+1 == settings['steps']:
            parameters = list(model.parameters()); vectors = []
            for key in ('pointwise_risk', 'selected_query_score'):
                gradients = torch.autograd.grad(terms[key], parameters, retain_graph=True, allow_unused=True)
                vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(parameters, gradients)]).detach().double())
            a, b = vectors; na, nb = float(a.norm()), float(b.norm())
            telemetry = dict(pointwise_gradient_norm=na, selected_query_gradient_norm=nb,
                query_over_pointwise_gradient=nb/na if na else None,
                query_pointwise_cosine=float(a@b)/(na*nb) if na and nb else None,
                complete_selected_queries=terms['complete_selected_queries'])
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; row_draws += len(ix)
        unknown_draws += int((~data['known'][ix]).sum())
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(terms['total'].detach()), gradient_norm=float(norm),
                monitor=diagnostic(), **telemetry)
            trace.append(row); heartbeat(state='critic_training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return read_checkpoint(path)


def assert_matched(left, right):
    for k in ('seed', 'settings', 'input_hashes', 'rms', 'initial_model', 'dimension', 'parameters',
              'query_keys', 'draw_hash', 'row_draws', 'unknown_rows_drawn', 'sampler_rng', 'torch_rng'):
        exact(left[k], right[k])
    if left['unknown_rows_with_loss'] or right['unknown_rows_with_loss']:
        raise AssertionError('Unknown outcomes cannot supervise either arm')


def predict(state, causal, *, chunk_rows=1024):
    if set(causal) != CAUSAL_KEYS:
        raise ValueError('Prediction accepts no labels, future endpoint or outcome mask')
    model = initialize(state['dimension'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    result = causal['prediction'].copy(); pending, size = [], 0

    def flush():
        rows = np.concatenate(pending)
        seg = np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        args = batch(causal, rows, seg, len(pending))
        with torch.no_grad(): p = model(*args, arm=state['arm']).numpy()
        for k in (1, 4): result[rows, :, k] = p[:, :, k]*causal['scales'][rows, None]

    for group in causal['groups']:
        if pending and size+len(group) > chunk_rows:
            flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending: flush()
    return result
