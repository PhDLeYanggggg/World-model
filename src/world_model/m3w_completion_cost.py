"""Paired known-only versus finite-completion endpoint supervision.

The endpoints are risk proxies, never imputed future observations. Inference
retains the original causal feature interface and unchanged decision budget.
"""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from src.world_model import m3w_multiaction_cost as base
from src.world_model import m3w_scene_context_cost as scene

MODES = ('known_only', 'completion_bounds')
BANKS = ('physical_only', 'physical_neural')
BOUND_COLUMNS = ('benefit_lower', 'harm_upper', 'reference_lower',
                 'easy_reference_lower', 'easy_harm_upper')
OBJECTIVE = 'same_asymmetric_five_proxy_MSE_and_within_query_ranking'


def completion_endpoints(observed, envelope):
    """Training-only bounds; the supplied observed-label array is not changed."""
    y, env = np.asarray(observed), np.asarray(envelope)
    if (y.ndim != 3 or y.shape[1:] != (7, 5) or env.shape != y.shape[:2]
            or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Seven aligned action labels and finite causal envelopes required')
    known = np.isfinite(y).all((1, 2))
    if (not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any()
            or np.any(y[known, :, 0]+y[known, :, 1] > env[known]+1e-5)):
        raise ValueError('Known legal moments or wholly unknown rows required')
    out = np.where(known[:, None, None], y, 0.).copy()
    out[~known, :, 1] = env[~known]
    out[~known, :, 4] = env[~known]
    return out, known


def training_data(packets, parent):
    """Use the frozen physical-fitting normalizer and a common past-only sampler."""
    data = parent.training_data(packets)
    names = sorted(packets)
    observed = np.concatenate([packets[n]['y'] for n in names])
    envelope = np.concatenate([packets[n]['envelope'] for n in names])
    scales = np.concatenate([packets[n]['scales'] for n in names])
    sites = np.concatenate([packets[n]['sites'] for n in names])
    bounds, known = completion_endpoints(observed, envelope)
    causal = data['allowed'].numpy().any(1)
    groups = [np.flatnonzero(causal & (sites == name)) for name in names]
    if any(not len(g) for g in groups):
        raise ValueError('Every fitting source needs causally eligible query rows')
    data['groups'] = groups
    data['known'] = torch.from_numpy(known)
    data['bound_truth'] = torch.from_numpy((bounds/scales[:, None, None]).astype(np.float32))
    data['input_hashes'] = dict(data['input_hashes'],
        completion_endpoints=parent.fingerprint(bounds), known_label_support=parent.fingerprint(known),
        common_past_only_groups=parent.fingerprint(np.concatenate(groups)))
    data['draw_pool'] = dict(rows=sum(map(len, groups)),
        known_rows=sum(int(known[g].sum()) for g in groups),
        unknown_rows=sum(int((~known[g]).sum()) for g in groups),
        causal_eligibility_only=True, future_support_used_for_group_membership=False)
    return data


def supervision(data, mode):
    if mode not in MODES:
        raise ValueError('One registered supervision mode required')
    if mode == 'known_only':
        return data['truth'], data['allowed'] & data['known'][:, None]
    return data['bound_truth'], data['allowed']


def fit(data, *, mode, bank, settings, seed, identity, path,
        heartbeat=lambda **_: None, guard=lambda: None, resume=False, stop_at=None):
    if mode not in MODES or bank not in BANKS:
        raise ValueError('Registered paired supervision and candidate bank required')
    pr = data['preprocess']; actions = torch.tensor(base.ARMS[bank])
    model = scene.SceneCostHead(data['z'].shape[-1], settings['width'], seed, pr['bias'])
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    target, real = supervision(data, mode)
    binding = dict(mode=mode, arm=bank, settings=settings, seed=seed, identity=identity,
        preprocess=pr, input_hashes=data['input_hashes'], initial_model=initial,
        dimension=data['z'].shape[-1], parameters=sum(p.numel() for p in model.parameters()),
        objective=OBJECTIVE, target_columns=list(BOUND_COLUMNS) if mode == 'completion_bounds' else
        ['known_benefit', 'known_harm', 'known_reference', 'known_easy_reference', 'known_easy_harm'],
        unknown_endpoints_are_observed_costs=False, future_support_inference_input=False,
        draw_pool=data['draw_pool'])
    step, elapsed, trace, draw_hash, row_draws = 0, 0., [], '0'*64, 0
    unknown_draws, unknown_loss, known_loss = 0, 0, 0
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError('Explicit checkpoint resume required')
        state = base.read_checkpoint(path)
        for key, value in binding.items():
            try:
                base.exact(state[key], value)
            except (AssertionError, ValueError) as error:
                raise ValueError('Checkpoint binding changed: '+key) from error
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, elapsed, trace, draw_hash, row_draws = (state[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws'))
        unknown_draws, unknown_loss, known_loss = (state[k] for k in
            ('unknown_rows_drawn', 'unknown_rows_with_loss', 'known_rows_with_loss'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit == 0:
        raise ValueError('Fixed nondecreasing training budget required')
    started = time.monotonic(); rms = torch.from_numpy(pr['rms'])

    def save():
        guard()
        base.save_checkpoint(path, dict(**binding, step=step,
            seconds=elapsed+time.monotonic()-started, trace=trace,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), draw_hash=draw_hash,
            row_draws=row_draws, unknown_rows_drawn=unknown_draws,
            unknown_rows_with_loss=unknown_loss, known_rows_with_loss=known_loss,
            inner_held_used_for_training=False, inferred_future_observation_count=0))

    if step == 0:
        save()
    while step < limit:
        ix = base.draw_batch(data['groups'], settings['batch_size'], rng)
        prediction = model(data['z'][ix][:, actions], data['envelope'][ix][:, actions], actions)
        active = real[ix][:, actions]
        terms = base.continuous_loss(prediction, target[ix][:, actions], active, rms,
            underharm_weight=settings['underharm_weight'], ranking_weight=settings['ranking_weight'])
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Nonfinite paired completion-cost objective')
        optimizer.zero_grad(set_to_none=True); terms['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; row_draws += len(ix)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+ix.tobytes()).hexdigest()
        known = data['known'][ix]; used = active.any(1)
        unknown_draws += int((~known).sum())
        unknown_loss += int((used & ~known).sum()); known_loss += int((used & known).sum())
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(terms['total'].detach()), regression=float(terms['regression'].detach()),
                ranking=float(terms['ranking'].detach()), gradient_norm=float(norm),
                unknown_rows_drawn=unknown_draws, unknown_rows_with_loss=unknown_loss)
            trace.append(row); heartbeat(state='training', mode=mode, bank=bank, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return base.read_checkpoint(path)


def predict(state, x, static_scene, envelope, scales):
    if state.get('mode') not in MODES or state.get('future_support_inference_input'):
        raise ValueError('Registered causal endpoint head required')
    return scene.predict(state, x, static_scene, envelope, scales)


def exact_training(left, right):
    for key in left:
        if key != 'seconds':
            base.exact(left[key], right[key])


def exact_paired_design(left, right):
    for key in ('initial_model', 'preprocess', 'input_hashes', 'draw_hash', 'row_draws',
                'sampler_rng', 'torch_rng', 'parameters', 'unknown_rows_drawn', 'draw_pool'):
        base.exact(left[key], right[key])
