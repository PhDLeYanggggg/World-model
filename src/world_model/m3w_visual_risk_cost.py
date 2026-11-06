"""Matched past-image cost heads with action-invariant reference/support inputs."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_role_separated_cost as parent

ARMS = ('no_image', 'image_bounds', 'quality_geometry', 'motion_geometry')
SETTINGS = dict(parent.SETTINGS, context_width=56)
base = parent.storage.storage.base
EDGE_NAMES = ('left_margin', 'right_margin', 'top_margin', 'bottom_margin',
    'minimum_current_margin', 'causal_CV_outside_fraction',
    'candidate_outside_fraction', 'candidate_minimum_margin')


def edge_features(history, rollout, native_size):
    h, p = np.asarray(history, float), np.asarray(rollout, float); size = np.asarray(native_size, float)
    if (h.shape != (len(h), 8, 2) or p.shape != (len(h), 12, 2) or size.shape != (2,)
            or not all(np.isfinite(x).all() for x in (h, p, size)) or (size <= 0).any()):
        raise ValueError('Causal history/rollout and native image dimensions required')
    scale = np.maximum(np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1), 1.)
    now = h[:, -1]; margin = np.c_[now[:, 0], size[0]-now[:, 0], now[:, 1], size[1]-now[:, 1]]
    cv = now[:, None]+np.arange(1, 13)[None, :, None]*(h[:, -1]-h[:, -2])[:, None]
    candidate = now[:, None]+p
    def outside(x):
        return ((x < 0)|(x >= size)).any(-1).mean(1)
    candidate_margin = np.minimum(candidate, size-candidate).min((1, 2))
    return np.c_[margin/scale[:, None], margin.min(1)/scale, outside(cv),
        outside(candidate), candidate_margin/scale].astype(np.float32)


def prepare(data):
    x, y, w, known, env, c = (np.asarray(data[k]) for k in ('x', 'moments', 'weights', 'observed', 'envelope', 'context'))
    n = len(x)
    if (x.shape != (n, 120) or y.shape != (n, 5) or c.shape != (n, 56) or w.shape != (n,)
            or known.shape != (n,) or known.dtype != bool or env.shape != (n,)
            or not all(np.isfinite(v).all() for v in (x, y, w, env, c))
            or (y < 0).any() or (w < 0).any() or (env < 0).any() or w.sum() <= 0
            or not known.any() or known.all() or len(set(data['ids'])) != n):
        raise ValueError('Complete A-only causal bank with separate observed/bound-only roles required')
    kw = w[known]/w[known].sum(); ww = w/w.sum()
    mean = np.einsum('n,nd->d', kw, x[known]); std = np.sqrt(np.einsum('n,nd->d', kw, (x[known]-mean)**2)).clip(1e-6)
    cm = np.einsum('n,nd->d', kw, c[known]); cs = np.sqrt(np.einsum('n,nd->d', kw, (c[known]-cm)**2)).clip(1e-6)
    rms = np.sqrt(np.einsum('n,nd->d', kw, y[known]**2)).clip(.05).astype(np.float32)
    prior = np.einsum('n,nd->d', kw, y[known]); radius = float(kw@env[known])
    fractions = np.maximum([prior[0], prior[1], max(radius-prior[0]-prior[1], 1e-5)], 1e-5)
    reference = max(float(prior[2]), 1e-5)
    gates = np.clip([prior[3]/reference, prior[4]/max(prior[1], 1e-5)], 1e-5, 1-1e-5)
    bias = np.r_[np.log(fractions[:2]/fractions[2]), reference+np.log(-np.expm1(-reference)), np.log(gates/(1-gates))].astype(np.float32)
    pr = dict(mean=mean, std=std, rms=rms, bias=bias, observation_prior=float(ww@known),
        context_mean=cm, context_std=cs, train_ids=data['ids'], train_sources=sorted(set(data['sites'])),
        fit_input_hash=parent.storage.digest(x), fit_context_hash=parent.storage.digest(c),
        fit_moment_hash=parent.storage.digest(y), fit_weight_hash=parent.storage.digest(ww),
        fit_role_hash=parent.storage.digest(known), fitting_only_normalization=True)
    return dict(x=torch.from_numpy(np.clip((x-mean)/std, -8, 8).astype(np.float32)),
        context=torch.from_numpy(np.clip((c-cm)/cs, -8, 8).astype(np.float32)),
        y=torch.from_numpy(y.astype(np.float32)), known=torch.from_numpy(known),
        envelope=torch.from_numpy(env.astype(np.float32)), probability=torch.from_numpy(ww),
        rms=torch.from_numpy(rms), binding=pr)


class Head(parent.Head):
    def __init__(self, cfg, preprocess):
        super().__init__(cfg, preprocess)
        self.visual_conditional = nn.Linear(56, cfg['width1'], bias=False)
        self.visual_reference = nn.Linear(46, cfg['reference_width'], bias=False)
        self.visual_observation = nn.Linear(46, cfg['observation_width'], bias=False)
        for layer in (self.visual_conditional, self.visual_reference, self.visual_observation):
            nn.init.zeros_(layer.weight)

    def forward(self, x, envelope, context):
        if x.shape != (len(envelope), 120) or context.shape != (len(envelope), 56):
            raise ValueError('Causal120 plus declared image56 only')
        observation = torch.cat((context[:, :40], context[:, 48:54]), 1)
        raw = self.conditional[2:](F.silu(self.conditional[0](x)+self.visual_conditional(context)))
        fraction = torch.softmax(torch.cat((raw[:, :2], torch.zeros_like(raw[:, :1])), 1), 1)
        ref = self.reference[2](F.silu(self.reference[0](x[:, :50])+self.visual_reference(observation)))
        reference = F.softplus(ref[:, 0]); harm = envelope*fraction[:, 1]
        conditional = torch.stack((torch.minimum(envelope*fraction[:, 0], reference), harm,
            reference, reference*torch.sigmoid(ref[:, 1]), harm*torch.sigmoid(raw[:, 2])), 1)
        logit = self.observation[2](F.silu(self.observation[0](x[:, :50])+self.visual_observation(observation)))[:, 0]
        q = torch.sigmoid(logit)
        unknown = torch.stack((torch.zeros_like(envelope), envelope, torch.zeros_like(envelope),
            torch.zeros_like(envelope), envelope), 1)
        return dict(conditional=conditional, observation_logit=logit, observation_probability=q,
            moments=q[:, None]*conditional+(1-q[:, None])*unknown)


def fit(data, identity, path, *, arm, seed, resume=False, stop_at=None,
        guard=lambda: None, heartbeat=lambda **_: None):
    if arm not in ARMS or seed not in (17, 29, 43):
        raise ValueError('Fixed image arms and three risk-head seeds required')
    p = prepare(data); cfg = SETTINGS; torch.manual_seed(seed); model = Head(cfg, p['binding'])
    initial = copy.deepcopy(model.state_dict()); opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = torch.Generator().manual_seed(seed); n = len(p['x']); path = Path(path)
    binding = dict(identity=identity, settings=cfg, arm=arm, seed=seed, preprocess=p['binding'],
        initial_model=initial, parameters=sum(v.numel() for v in model.parameters()), fit_rows=n)
    step, elapsed, trace, visits = 0, 0., [], np.zeros(n, np.int32)
    if path.exists():
        if not resume: raise ValueError('Explicit same-definition resume required')
        old = base.read_checkpoint(path)
        for k, v in binding.items(): base.exact(old[k], v)
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        rng.set_state(old['generator']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, visits = (old[k] for k in ('step', 'seconds', 'trace', 'visits'))
    limit = cfg['steps'] if stop_at is None else stop_at
    if not step <= limit <= cfg['steps']: raise ValueError('Fixed final training budget required')
    started = time.monotonic()
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            generator=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace, visits=visits, complete=step == cfg['steps']))
    try:
        while step < limit:
            guard(); ix = torch.multinomial(p['probability'], cfg['batch_size'], replacement=True, generator=rng)
            opt.zero_grad(set_to_none=True)
            terms = parent.objective(model(p['x'][ix], p['envelope'][ix], p['context'][ix]),
                p['y'][ix], p['known'][ix], p['rms'], 'role_separated')
            if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite image-risk loss')
            terms['total'].backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1; np.add.at(visits, ix.numpy(), 1)
            if step == 1 or step % cfg['log_every'] == 0:
                row = dict(step=step, gradient_norm=float(norm), **{k: float(v.detach()) for k, v in terms.items()},
                    distinct_fitting_rows_drawn=int((visits > 0).sum()))
                trace.append(row); heartbeat(state='visual_risk_training', arm=arm, **row)
            if step % cfg['checkpoint_steps'] == 0 or step == limit: save()
    except BaseException:
        save(); raise
    return base.read_checkpoint(path)


def predict(state, x, envelope, context):
    x, env, c = np.asarray(x), np.asarray(envelope), np.asarray(context)
    if (x.shape != (len(env), 120) or c.shape != (len(env), 56)
            or not all(np.isfinite(v).all() for v in (x, env, c)) or (env < 0).any()):
        raise ValueError('Finite past inputs and rollout diagnostics required')
    pr = state['preprocess']; model = Head(state['settings'], pr); model.load_state_dict(state['model']); model.eval()
    z = np.clip((x-pr['mean'])/pr['std'], -8, 8).astype(np.float32)
    c = np.clip((c-pr['context_mean'])/pr['context_std'], -8, 8).astype(np.float32)
    output = {k: [] for k in ('conditional', 'moments', 'observation_probability')}
    with torch.no_grad():
        for start in range(0, len(x), 512):
            result = model(torch.from_numpy(z[start:start+512]), torch.from_numpy(env[start:start+512].astype(np.float32)),
                torch.from_numpy(c[start:start+512]))
            for k in output: output[k].append(result[k].numpy())
    return {k: np.concatenate(v).astype(float) for k, v in output.items()}


def exact(a, b):
    parent.exact(a, b)
