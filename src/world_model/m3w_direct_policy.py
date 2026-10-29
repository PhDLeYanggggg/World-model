"""Direct intervention learning; empirical constraints are not calibration."""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before importing Torch')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')

import numpy as np
import torch
from src.world_model import m3w_query_context_risk as frozen
from src.world_model import m3w_shared_train_risk as shared
from src.evaluation.m3w_selected_tail_audit import actions

ARMS = ('soft_source', 'soft_record', 'hard_record')
core = frozen.core
fingerprint = frozen.fingerprint


class Policy(torch.nn.Module):
    def __init__(self, dimension, width, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(dimension, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, anchor):
        return self.net(z).squeeze(-1) + 2*anchor.to(z.dtype)-1


def inputs(x, envelope, prediction, preprocess, eligible):
    if (x.ndim != 2 or prediction.shape != (len(x), 5) or eligible.shape != (len(x),)
            or eligible.dtype != bool or not np.isfinite(prediction).all()
            or (prediction < 0).any()):
        raise ValueError('Finite causal inputs and frozen moments required')
    frozen.parent.previous.old.parent.forest.causal_inputs(x, envelope, preprocess)
    z = np.clip((x-preprocess['mean'])/preprocess['std'],
                -preprocess['clip'], preprocess['clip'])
    p = np.clip(prediction/preprocess['scale'], 0, preprocess['clip'])
    return np.column_stack((z, p)).astype(np.float32), actions(prediction, eligible)


def targets(y, envelope):
    """Supervised loss only. Unknown selection receives its completion worst case."""
    if y.shape != (len(envelope), 5) or not np.isfinite(envelope).all() or (envelope < 0).any():
        raise ValueError('Aligned nonnegative envelopes and target moments required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known] < 0).any():
        raise ValueError('Unknown targets must remain entirely unknown')
    clean = np.where(known[:, None], y, 0.)
    missing = np.where(known, 0., envelope)
    gain = clean[:, 0]-clean[:, 1]-missing
    risk = np.column_stack((clean[:, 1]-.02*clean[:, 2]+missing,
                            clean[:, 4]-.02*clean[:, 3]+missing))
    return gain, risk, clean[:, [2, 3]], known


def decision(logit, hard, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive fixed-schedule temperature required')
    soft = torch.sigmoid(logit/temperature)
    if hard:
        return (logit > 0).to(soft.dtype) + (soft-soft.detach())
    return soft


def reduce_risk(record_risk, data, arm):
    if arm not in ARMS:
        raise ValueError('Registered arm required')
    if arm == 'soft_source':
        out = record_risk.new_zeros((len(data['sources']), 2))
        out.index_add_(0, data['source_index'], record_risk*data['within_source'][:, None])
        return out/data['source_scales']
    return record_risk/data['record_scales']


def prepare(packets, predictions):
    sources = sorted(packets)
    zs, anchors, gains, risks, groups, record_keys, source_ids = [], [], [], [], [], [], []
    sizes, scales, unknowns, hashes, normalizers, empty = [], [], [], {}, {}, []
    offset = 0
    for source_id, source in enumerate(sources):
        args = packets[source]
        x, env, y, _, _, recs, _, pr = args
        eligible = shared.eligible(args)
        z, anchor = inputs(x, env, predictions[source], pr, eligible)
        gain, risk, ref, known = targets(y/pr['scale'], env/pr['scale'])
        hashes[source] = {k: fingerprint(v) for k, v in zip(
            ('x', 'envelope', 'labels', 'series', 'sites', 'recordings', 'frames'), args[:-1])}
        hashes[source]['prediction'] = fingerprint(predictions[source])
        normalizers[source] = pr
        zs.append(z); anchors.append(anchor); gains.append(gain); risks.append(risk)
        for rec in sorted(set(recs)):
            ix = np.flatnonzero((recs == rec) & eligible)
            key = json.dumps([source, str(rec)], separators=(',', ':'))
            if not len(ix):
                empty.append(key)
                continue
            groups.append(torch.from_numpy(ix+offset)); record_keys.append(key)
            source_ids.append(source_id); sizes.append(len(ix)); unknowns.append(int((~known[ix]).sum()))
            # Fixed positive TRAIN scaling changes gradient units, not budget signs.
            causal_scale = max(float(np.mean(env[ix]/pr['scale'])), 1e-3)
            scales.append(np.where(ref[ix].mean(0) > 1e-6, ref[ix].mean(0), causal_scale))
        offset += len(x)
    si = np.asarray(source_ids, dtype=np.int64); sizes = np.asarray(sizes, float)
    if set(si) != set(range(len(sources))):
        raise ValueError('Every source needs an eligible training recording')
    within = sizes/np.bincount(si, weights=sizes)[si]
    record_scales = np.asarray(scales, dtype=np.float32)
    source_scales = np.array([(within[si == s, None]*record_scales[si == s]).sum(0)
                              for s in range(len(sources))], dtype=np.float32)
    return dict(z=torch.from_numpy(np.concatenate(zs)), anchor=torch.from_numpy(np.concatenate(anchors)),
        gain=torch.tensor(np.concatenate(gains), dtype=torch.float32),
        risk=torch.tensor(np.concatenate(risks), dtype=torch.float32), groups=groups,
        source_index=torch.from_numpy(si), within_source=torch.tensor(within, dtype=torch.float32),
        record_scales=torch.from_numpy(record_scales), source_scales=torch.from_numpy(source_scales),
        weights=torch.tensor(within/len(sources), dtype=torch.float32), sources=sources,
        record_keys=record_keys, sizes=sizes.tolist(), unknowns=unknowns,
        input_hashes=hashes, source_preprocess=normalizers, empty_training_records=empty)


def fit(data, *, arm, settings, seed, identity, path, heartbeat=lambda **_: None,
        resume=False, stop_at=None, guard=lambda: None):
    if arm not in ARMS:
        raise ValueError('Registered direct-policy arm required')
    model = Policy(data['z'].shape[1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed+7919)
    dual = torch.zeros((len(data['sources']) if arm == 'soft_source' else len(data['groups']), 2))
    binding = dict(arm=arm, settings=settings, seed=seed, identity=identity, initial_model=initial,
        input_hashes=data['input_hashes'], source_preprocess=data['source_preprocess'],
        record_keys=data['record_keys'], empty_training_records=data['empty_training_records'],
        record_sizes=data['sizes'], record_unknown_counts=data['unknowns'],
        record_scales=data['record_scales'], source_scales=data['source_scales'],
        dimension=data['z'].shape[1], parameters=sum(p.numel() for p in model.parameters()))
    step, seconds, trace, draw_hash, row_draws = 0, 0., [], '0'*64, 0
    path = Path(path)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        state = core.read_checkpoint(path)
        for key, value in binding.items(): core.exact(state[key], value)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draw_hash, row_draws, dual = (state[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws', 'dual'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Nondecreasing fixed update budget required')
    began = time.monotonic()

    def save():
        guard(); path.parent.mkdir(parents=True, exist_ok=True)
        core.save_checkpoint(path, dict(**binding, step=step, seconds=seconds+time.monotonic()-began,
            trace=trace, draw_hash=draw_hash, row_draws=row_draws, dual=dual,
            sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            new_forecaster=False, held_used_for_training=False))

    if step == 0: save()
    count = settings['rows_per_record']
    while step < limit:
        ix = torch.stack([g[torch.randint(len(g), (count,), generator=generator)] for g in data['groups']])
        flat = ix.flatten()
        logits = model(data['z'][flat], data['anchor'][flat]).reshape(ix.shape)
        temperature = 1.-.75*min(step/max(1, settings['steps']-1), 1.)
        action = decision(logits, arm == 'hard_record', temperature)
        gain = (action*data['gain'][ix]).mean(1)
        raw_risk = (action[..., None]*data['risk'][ix]).mean(1)
        risk = reduce_risk(raw_risk, data, arm)
        weights = (torch.full((len(data['sources']),), 1./len(data['sources']))
                   if arm == 'soft_source' else data['weights'])
        loss = -(data['weights']*gain).sum()+(weights[:, None]*dual.detach()*risk).sum()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite direct-policy objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        dual = (dual+settings['dual_learning_rate']*risk.detach()).clamp(0, settings['dual_maximum'])
        step += 1; row_draws += len(flat)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+flat.numpy().tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
            row = dict(step=step, loss=float(loss.detach()), gain_lower=float((data['weights']*gain).sum().detach()),
                gradient_norm=float(norm), action_mean=float(action.mean().detach()),
                raw_record_risk=raw_risk.detach().tolist(), scaled_risk=risk.detach().tolist(),
                dual=dual.tolist(), soft_hard_gap=float((torch.sigmoid(logits.detach()/temperature)-
                                                         (logits.detach() > 0).to(logits.dtype)).abs().mean()))
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return core.read_checkpoint(path)


def assert_matched(states):
    if tuple(s['arm'] for s in states) != ARMS:
        raise ValueError('All registered arms required')
    for state in states[1:]:
        for key in ('settings', 'seed', 'identity', 'initial_model', 'input_hashes', 'source_preprocess',
                    'record_keys', 'record_sizes', 'record_scales', 'sampler_rng', 'draw_hash', 'row_draws', 'step'):
            core.exact(states[0][key], state[key])


def predict(state, source, x, envelope, prediction, eligible):
    z, anchor = inputs(x, envelope, prediction, state['source_preprocess'][source], eligible)
    model = Policy(state['dimension'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    scores = []
    with torch.no_grad():
        for lo in range(0, len(z), 4096):
            scores.append(model(torch.from_numpy(z[lo:lo+4096]),
                                torch.from_numpy(anchor[lo:lo+4096])).numpy())
    score = np.concatenate(scores).astype(float)
    return score, eligible & (score > 0)
