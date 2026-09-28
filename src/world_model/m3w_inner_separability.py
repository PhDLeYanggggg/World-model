"""Single-source gain/risk fitting with a shared bounded moment decoder."""
import hashlib
from pathlib import Path
import platform
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before Torch import')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_query_excess as sampling
from src.world_model.m3w_causal_descriptor_head import read_checkpoint, save_checkpoint
from src.world_model.m3w_fixed_floor_probe import targets
from src.world_model.m3w_easy_component_diagnostic import weights
from src.world_model.m3w_score_support_calibration import support_limit

ARMS = ('affine', 'nonlinear')
BUDGET = .02
exact = sampling.exact


def signed(y):
    stack = torch.stack if isinstance(y, torch.Tensor) else np.stack
    return stack((y[:, 0]-y[:, 1], y[:, 1]-BUDGET*y[:, 2], y[:, 4]-BUDGET*y[:, 3]), 1)


def preprocess(x, envelope, target, sites, recordings, frames, *, training_site):
    x, env, y, sites = map(np.asarray, (x, envelope, target, sites))
    n = len(x); known = np.isfinite(y).all(1)
    if (x.ndim != 2 or y.shape != (n, 5) or sites.shape != (n,) or set(sites) != {training_site}
            or env.shape != (n,) or not np.isfinite(x).all() or not np.isfinite(env).all()
            or (env < 0).any() or not known.any() or not (known | np.isnan(y).all(1)).all()
            or (y[known] < 0).any() or (y[known, :2].sum(1) > env[known]+1e-5).any()
            or (y[known, 3] > y[known, 2]+1e-8).any() or (y[known, 4] > y[known, 1]+1e-8).any()):
        raise ValueError('One training locality, causal features and consistent moment labels required')
    w, _ = weights(sites, recordings, frames, known)
    mean = (w[:, None]*x).sum(0); std = np.sqrt((w[:, None]*(x-mean)**2).sum(0)).clip(1e-6)
    scale = float(w[known]@y[known, 2])
    if scale <= 0 or not np.isfinite(scale): raise ValueError('Positive training reference cost required')
    yn = y[known]/scale; means = w[known]@yn; env_mean = float(w@env)/scale
    fractions = np.maximum([means[0]/max(env_mean, 1e-8), means[1]/max(env_mean, 1e-8),
                            1-(means[0]+means[1])/max(env_mean, 1e-8)], 1e-5)
    fractions /= fractions.sum()
    conditional = np.clip([means[3]/max(means[2], 1e-8), means[4]/max(means[1], 1e-8)], 1e-5, 1-1e-5)
    bias = np.r_[np.log(fractions[:2]/fractions[2]), np.log(np.expm1(means[2])),
                 np.log(conditional/(1-conditional))].astype(np.float32)
    extended = np.column_stack((yn, signed(yn)))
    rms = np.sqrt((w[known, None]*extended**2).sum(0)).clip(.01)
    distance = np.sqrt(np.mean(((x-mean)/std)**2, 1))
    return dict(mean=mean, std=std, scale=scale, bias=bias, rms=rms.astype(np.float32),
        clip=8., support_limit=support_limit(distance, w), training_site=training_site,
        known_rows=int(known.sum()), unknown_rows=int((~known).sum()))


class MomentHead(nn.Module):
    def __init__(self, dims, arm, width, bias):
        super().__init__()
        if arm not in ARMS: raise ValueError('Registered arm required')
        self.encoder = nn.Identity() if arm == 'affine' else nn.Sequential(nn.Linear(dims, width), nn.SiLU())
        self.output = nn.Linear(dims if arm == 'affine' else width, 5)
        with torch.no_grad():
            self.output.weight.zero_(); self.output.bias.copy_(torch.as_tensor(bias))

    def forward(self, z, env):
        raw = self.output(self.encoder(z))
        fractions = torch.softmax(torch.cat((raw[:, :2], torch.zeros_like(raw[:, :1])), 1), 1)
        benefit, harm = env*fractions[:, 0], env*fractions[:, 1]
        reference = F.softplus(raw[:, 2])
        return torch.stack((benefit, harm, reference, reference*torch.sigmoid(raw[:, 3]),
                            harm*torch.sigmoid(raw[:, 4])), 1)


def initialize(pr, arm, width, seed):
    torch.manual_seed(seed)
    return MomentHead(len(pr['mean']), arm, width, pr['bias'])


def losses(prediction, target, segments, count, rms):
    if prediction.shape != target.shape or target.shape[1] != 5 or not torch.isfinite(target).all():
        raise ValueError('Five known moment targets required')
    delta = torch.cat((prediction-target, signed(prediction)-signed(target)), 1)/rms
    sizes = torch.bincount(segments, minlength=count).to(delta.dtype)
    if (sizes == 0).any(): raise ValueError('Nonempty query segments required')
    sums = delta.new_zeros((count, 8)).index_add(0, segments, delta.square())/sizes[:, None]
    moments, decision_scores = sums[:, :5].mean(), sums[:, 5:].mean()
    return dict(moments=moments, decision_scores=decision_scores, total=.5*(moments+decision_scores))


def fit(x, envelope, target, sites, recordings, frames, pr, *, arm, settings, identity,
        seed, path, heartbeat, resume=False, stop_at=None):
    fresh = preprocess(x, envelope, target, sites, recordings, frames, training_site=pr['training_site'])
    exact(fresh, pr)
    known = np.isfinite(target).all(1); groups, source_groups, keys = sampling.query_groups(sites, recordings, frames, known)
    z = torch.from_numpy(np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope/pr['scale']).astype(np.float32))
    truth = torch.from_numpy(np.where(known[:, None], target/pr['scale'], 0).astype(np.float32))
    rms = torch.from_numpy(pr['rms']); model = initialize(pr, arm, settings['width'], seed)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = sampling.draw_queries(groups, source_groups, 128, torch.Generator().manual_seed(seed+9137))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draw_hash, draws = 0, 0., [], '0'*64, 0
    if path.exists():
        if not resume: raise ValueError('Existing training requires resume')
        old = read_checkpoint(path)
        for k, v in dict(identity=identity, settings=settings, arm=arm, preprocess=pr, seed=seed, initial_model=initial).items(): exact(old[k], v)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']); torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, draw_hash, draws = (old[k] for k in ('step', 'seconds', 'trace', 'draw_hash', 'row_draws'))
    limit = settings['steps'] if stop_at is None else min(settings['steps'], stop_at)
    if limit < step or limit <= 0: raise ValueError('Nondecreasing update budget required')
    began = time.monotonic()

    def monitor_loss():
        ix, seg, _ = monitor
        with torch.no_grad():
            return {k: float(v) for k, v in losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), 128, rms).items()}

    def save():
        save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, preprocess=pr, seed=seed,
            initial_model=initial, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step, trace=trace,
            seconds=seconds+time.monotonic()-began, draw_hash=draw_hash, row_draws=draws,
            queries=len(keys), unknown_rows_sampled=0))

    if not trace: trace.append(dict(step=0, monitor=monitor_loss()))
    while step < limit:
        ix, seg, qids = sampling.draw_queries(groups, source_groups, settings['query_batch_size'], rng)
        optimizer.zero_grad(set_to_none=True)
        terms = losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), settings['query_batch_size'], rms)
        if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite training loss')
        terms['total'].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; draws += len(ix)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(terms['total'].detach()), gradient_norm=float(gradient), monitor=monitor_loss())
            trace.append(row); heartbeat(state='training', **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return read_checkpoint(path)


def predict(state, x, envelope, *, initial=False):
    pr = state['preprocess']; x, envelope = map(np.asarray, (x, envelope))
    if (x.ndim != 2 or x.shape[1] != len(pr['mean']) or envelope.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(envelope).all() or (envelope < 0).any()):
        raise ValueError('Causal inputs only')
    model = initialize(pr, state['arm'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['initial_model' if initial else 'model']); model.eval()
    outputs = []; distance = np.sqrt(np.mean(((x-pr['mean'])/pr['std'])**2, 1))
    with torch.no_grad():
        for first in range(0, len(x), 4096):
            z = torch.from_numpy(np.clip((x[first:first+4096]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
            env = torch.from_numpy((envelope[first:first+4096]/pr['scale']).astype(np.float32))
            outputs.append(model(z, env).numpy().astype(float)*pr['scale'])
    return np.concatenate(outputs), distance <= pr['support_limit']


def assert_matched(a, b):
    assert (a['arm'], b['arm']) == ARMS
    for k in ('identity', 'settings', 'preprocess', 'seed', 'step', 'sampler_rng', 'draw_hash', 'row_draws', 'queries'):
        exact(a[k], b[k])


def decisions(predictions, moving, support, recordings, frames, ids):
    moving, support, ids = map(np.asarray, (moving, support, ids))
    if (set(predictions) != set(ARMS) or moving.dtype != bool or support.dtype != bool
            or ids.ndim != 1 or moving.shape != ids.shape or support.shape != ids.shape
            or np.asarray(recordings).shape != ids.shape or np.asarray(frames).shape != ids.shape
            or len(set(ids)) != len(ids)):
        raise ValueError('Paired causal scores and unique row IDs required')
    take, utility = {}, {}
    for arm, p in predictions.items():
        if p.shape != (len(ids), 5) or not np.isfinite(p).all() or (p < 0).any(): raise ValueError('Finite moment predictions required')
        z = signed(p); utility[arm] = z[:, 0]
        take[arm] = moving & support & (z[:, 0] > 0) & (z[:, 1:] <= 0).all(1)
        take[arm+'_matched'] = np.zeros(len(ids), bool)
    groups = {}
    for i, key in enumerate(zip(recordings, frames)): groups.setdefault(key, []).append(i)
    for ix in groups.values():
        at = np.array(ix); count = min(int(take[a][at].sum()) for a in ARMS)
        for arm in ARMS:
            pool = at[take[arm][at]]; order = pool[np.lexsort((ids[pool], -utility[arm][pool]))]
            take[arm+'_matched'][order[:count]] = True
    return take
