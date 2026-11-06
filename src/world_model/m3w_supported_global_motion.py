"""Matched data-support by admitted global-past-motion cost experiment."""
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_role_separated_cost as parent

ARMS = ('own_local', 'own_global', 'augmented_local', 'augmented_global')
SETTINGS = dict(parent.SETTINGS, context_width=48)
base = parent.storage.storage.base


def lookup(ids, context_ids, context):
    """Exact agent/action namespace join; no outcome is accepted."""
    ids, keys, values = np.asarray(ids), np.asarray(context_ids), np.asarray(context)
    if (ids.ndim != 1 or keys.ndim != 1 or values.shape != (len(keys), 48)
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(keys.dtype, np.integer) or len(np.unique(keys)) != len(keys)
            or not np.isfinite(values).all() or (ids < 0).any()
            or not np.isin(ids % 7, np.arange(1, 7)).all()):
        raise ValueError('Finite causal48 and unique original agent IDs required')
    order = np.argsort(keys)
    at = np.searchsorted(keys[order], ids // 7)
    if (at == len(keys)).any() or not np.array_equal(keys[order[at]], ids // 7):
        raise ValueError('Missing exact causal context; no nearest-row join')
    return values[order[at]].astype(np.float32)


def prepare(data, held):
    p = parent.prepare(data, held)
    c = np.asarray(data['context'])
    fit = data['sites'] != held
    known = fit & data['observed']
    if c.shape != (len(data['ids']), 48) or not np.isfinite(c).all():
        raise ValueError('Already-admitted global past motion, not targets, required')
    w = data['weights'][known]; w = w / w.sum()
    mean = np.einsum('n,nd->d', w, c[known])
    std = np.sqrt(np.einsum('n,nd->d', w, (c[known] - mean)**2)).clip(1e-6)
    return dict(p, context=torch.from_numpy(np.clip((c[fit]-mean)/std, -8, 8).astype(np.float32)),
        binding=dict(parent=p['binding'], context_mean=mean, context_std=std,
            fit_context_hash=parent.storage.digest(c[fit]), train_sources=p['binding']['train_sources'],
            held_source=held, context_normalization_known_fitting_only=True))


class Head(parent.Head):
    def __init__(self, cfg, preprocess):
        super().__init__(cfg, preprocess)
        self.global_conditional = nn.Linear(48, cfg['width1'], bias=False)
        self.global_reference = nn.Linear(48, cfg['reference_width'], bias=False)
        self.global_observation = nn.Linear(48, cfg['observation_width'], bias=False)
        for layer in (self.global_conditional, self.global_reference, self.global_observation):
            nn.init.zeros_(layer.weight)

    def forward(self, x, envelope, context, arm):
        if (arm not in ARMS or x.shape != (len(envelope), 120)
                or context.shape != (len(envelope), 48)):
            raise ValueError('Fixed arm, causal120 and causal48 required')
        c = context if arm.endswith('_global') else torch.zeros_like(context)
        raw = self.conditional[2:](F.silu(self.conditional[0](x) + self.global_conditional(c)))
        fraction = torch.softmax(torch.cat((raw[:, :2], torch.zeros_like(raw[:, :1])), 1), 1)
        ref = self.reference[2](F.silu(self.reference[0](x[:, :50]) + self.global_reference(c)))
        reference = F.softplus(ref[:, 0]); harm = envelope * fraction[:, 1]
        conditional = torch.stack((torch.minimum(envelope*fraction[:, 0], reference), harm,
            reference, reference*torch.sigmoid(ref[:, 1]), harm*torch.sigmoid(raw[:, 2])), 1)
        logit = self.observation[2](F.silu(self.observation[0](x[:, :50]) + self.global_observation(c)))[:, 0]
        q = torch.sigmoid(logit)
        unknown = torch.stack((torch.zeros_like(envelope), envelope, torch.zeros_like(envelope),
            torch.zeros_like(envelope), envelope), 1)
        return dict(conditional=conditional, observation_logit=logit, observation_probability=q,
            moments=q[:, None]*conditional+(1-q[:, None])*unknown)


def fit(data, held, identity, path, *, arm, seed, resume=False, stop_at=None,
        guard=lambda: None, heartbeat=lambda **_: None):
    if arm not in ARMS or seed not in (17, 29, 43):
        raise ValueError('Registered factorial arms and seeds required')
    p = prepare(data, held); cfg = SETTINGS
    torch.manual_seed(seed); model = Head(cfg, p['binding']['parent'])
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = torch.Generator().manual_seed(seed); n = len(p['x'])
    binding = dict(identity=identity, settings=cfg, arm=arm, seed=seed, preprocess=p['binding'],
        initial_model=initial, parameters=sum(v.numel() for v in model.parameters()), fit_rows=n)
    step, elapsed, trace, visits = 0, 0., [], np.zeros(n, np.int32); path = Path(path)
    if path.exists():
        if not resume: raise ValueError('Explicit same-definition resume required')
        old = base.read_checkpoint(path)
        for k, v in binding.items(): base.exact(old[k], v)
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        rng.set_state(old['generator']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, visits = (old[k] for k in ('step', 'seconds', 'trace', 'visits'))
    limit = cfg['steps'] if stop_at is None else stop_at
    if not step <= limit <= cfg['steps']: raise ValueError('Fixed nondecreasing budget required')
    start = time.monotonic()
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            generator=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-start, trace=trace, visits=visits, complete=step == cfg['steps']))
    try:
        while step < limit:
            guard(); ix = torch.multinomial(p['probability'], cfg['batch_size'], replacement=True, generator=rng)
            opt.zero_grad(set_to_none=True)
            terms = parent.objective(model(p['x'][ix], p['envelope'][ix], p['context'][ix], arm),
                p['y'][ix], p['known'][ix], p['rms'], 'role_separated')
            if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite cost/support loss')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1; np.add.at(visits, ix.numpy(), 1)
            if step == 1 or step % cfg['log_every'] == 0:
                row = dict(step=step, gradient_norm=float(norm), **{k: float(v.detach()) for k, v in terms.items()},
                    distinct_fitting_rows_drawn=int((visits > 0).sum()))
                trace.append(row); heartbeat(state='global_motion_training', arm=arm, **row)
            if step % cfg['checkpoint_steps'] == 0 or step == limit: save()
    except BaseException:
        save(); raise
    return base.read_checkpoint(path)


def predict(state, x, envelope, context):
    x, env, c = np.asarray(x), np.asarray(envelope), np.asarray(context)
    if (x.shape != (len(env), 120) or c.shape != (len(env), 48)
            or not all(np.isfinite(v).all() for v in (x, env, c)) or (env < 0).any()):
        raise ValueError('Finite causal-only aligned inputs required')
    pr = state['preprocess']; old = pr['parent']
    z = np.clip((x-old['mean'])/old['std'], -8, 8).astype(np.float32)
    c = np.clip((c-pr['context_mean'])/pr['context_std'], -8, 8).astype(np.float32)
    model = Head(state['settings'], old); model.load_state_dict(state['model']); model.eval()
    out = {k: [] for k in ('conditional', 'moments', 'observation_probability')}
    with torch.no_grad():
        for start in range(0, len(x), 512):
            result = model(torch.from_numpy(z[start:start+512]),
                torch.from_numpy(env[start:start+512].astype(np.float32)), torch.from_numpy(c[start:start+512]), state['arm'])
            for k in out: out[k].append(result[k].numpy())
    return {k: np.concatenate(v).astype(float) for k, v in out.items()}


def exact(a, b):
    parent.exact(a, b)
