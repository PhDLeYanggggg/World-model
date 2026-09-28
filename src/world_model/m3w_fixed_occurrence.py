"""Paired warm-start heads differing only in occurrence-branch trainability."""
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_easy_hurdle as old

np, torch, nn = old.np, old.torch, old.nn
ARMS = ('trainable', 'fixed')


class SplitHead(nn.Module):
    def __init__(self, warm, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError('Unregistered occurrence treatment')
        self.occurrence = old.initialize(len(warm['norm']['mean']), warm['settings']['width'], warm['seed'])
        self.cost = old.initialize(len(warm['norm']['mean']), warm['settings']['width'], warm['seed'])
        self.occurrence.load_state_dict(warm['model']); self.cost.load_state_dict(warm['model'])
        self.occurrence.requires_grad_(arm == 'trainable')

    def forward(self, x, envelope):
        p = self.occurrence(x, envelope); c = self.cost(x, envelope)
        risk = p[:, 0]*(c[:, 2]-old.BUDGET*c[:, 1])
        return torch.stack((p[:, 0], c[:, 1], c[:, 2], risk, p[:, 4]), 1)


def initialize(warm, arm):
    return SplitHead(warm, arm)


def fit(x, envelope, target, sites, recordings, frames, warm, *, arm, settings,
        identity, path, heartbeat, resume=False, stop_at=None):
    x, envelope, target = map(np.asarray, (x, envelope, target))
    if (x.ndim != 2 or envelope.shape != (len(x),) or target.shape != (len(x), 3)
            or not np.isfinite(x).all() or not np.isfinite(envelope).all() or (envelope < 0).any()):
        raise ValueError('Normalized causal inputs and aligned supervised labels required')
    known = np.isfinite(target).all(1)
    if not (known | np.isnan(target).all(1)).all() or not (target[known, 0] == 1).any():
        raise ValueError('Known fitting easy labels and entirely missing unknown rows required')
    if set(sites) != set(warm['norm']['training_sites']): raise ValueError('Fitting-source mismatch')
    if (target[known, 2] > envelope[known]+1e-5).any(): raise ValueError('Causal harm envelope exceeded')
    groups, sources, keys = old.sampling.query_groups(sites, recordings, frames, known)
    z = torch.from_numpy(x.astype(np.float32)); env = torch.from_numpy(envelope.astype(np.float32))
    truth = torch.from_numpy(np.where(known[:, None], target, 0).astype(np.float32))
    warm = {k: warm[k] for k in ('model', 'settings', 'seed', 'norm')}
    model = initialize(warm, arm); initial = {k: v.clone() for k, v in model.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=.0001)
    seed = warm['seed']; rng = torch.Generator().manual_seed(seed+7919)
    monitor = old.sampling.draw_queries(groups, sources, 128, torch.Generator().manual_seed(seed+9137))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, sample_hash, query_draws, row_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume: raise ValueError('Existing warm-start fit requires resume')
        s = old.head.read_checkpoint(path)
        for k, v in dict(identity=identity, settings=settings, arm=arm, warm_start=warm, initial_model=initial).items():
            old.sampling.exact(s[k], v)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, seconds, trace, sample_hash, query_draws, row_draws = (s[k] for k in
            ('step', 'seconds', 'trace', 'sample_hash', 'query_draws', 'row_draws'))
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0: raise ValueError('Nondecreasing training budget required')
    started = time.monotonic()

    def monitor_loss():
        ix, seg, _ = monitor
        with torch.no_grad():
            return {k: float(v) for k, v in old.losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), 128).items()}

    def save():
        if arm == 'fixed':
            for k, v in model.occurrence.state_dict().items():
                old.sampling.exact(v, initial['occurrence.'+k])
        old.head.save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, warm_start=warm,
            initial_model=initial, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=seconds+time.monotonic()-started, trace=trace, sample_hash=sample_hash,
            query_draws=query_draws, row_draws=row_draws, fitting_queries=len(keys), unknown_rows_sampled=0,
            occurrence_frozen_exact=True if arm == 'fixed' else None,
            probability_calibration_certificate=False))

    if not trace: trace.append(dict(step=0, monitor=monitor_loss()))
    model.train()
    while step < limit:
        ix, seg, qids = old.sampling.draw_queries(groups, sources, settings['query_batch_size'], rng)
        optimizer.zero_grad(set_to_none=True)
        terms = old.losses(model(z[ix], env[ix]), truth[ix], torch.from_numpy(seg), settings['query_batch_size'])
        loss = terms['supervised']
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite supervised fitting loss')
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(params, settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1
        sample_hash = hashlib.sha256(bytes.fromhex(sample_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        query_draws += len(qids); row_draws += len(ix)
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), monitor=monitor_loss())
            trace.append(row); heartbeat(state='training', **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return old.head.read_checkpoint(path)


def predict(state, x, envelope):
    model = initialize(state['warm_start'], state['arm']); model.load_state_dict(state['model']); model.eval()
    x, envelope = map(lambda a: torch.from_numpy(np.asarray(a, np.float32)), (x, envelope))
    rows = []
    with torch.no_grad():
        for i in range(0, len(x), 4096): rows.append(model(x[i:i+4096], envelope[i:i+4096]).numpy())
    return np.concatenate(rows)


def assert_matched(a, b):
    assert (a['arm'], b['arm']) == ARMS
    for k in ('warm_start', 'initial_model', 'settings', 'sampler_rng', 'torch_rng',
              'step', 'sample_hash', 'query_draws', 'row_draws', 'fitting_queries'):
        old.sampling.exact(a[k], b[k])
