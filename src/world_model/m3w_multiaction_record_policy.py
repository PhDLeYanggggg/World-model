"""Exact fitting-record decision training over frozen causal cost predictions."""
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_multiaction_cost import ARMS, read_checkpoint, save_checkpoint, exact


class RecordActionPolicy(nn.Module):
    def __init__(self, width, seed):
        super().__init__(); torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(19, width), nn.SiLU(), nn.Linear(width, 1))
        with torch.no_grad():
            self.network[-1].weight.zero_(); self.network[-1].bias.zero_()

    def forward(self, features, anchor, action_ids):
        if features.shape[-1] != 12 or anchor.shape != features.shape[:2]:
            raise ValueError('Twelve causal descriptors/moments and original action anchor required')
        code = torch.nn.functional.one_hot(action_ids, 7).to(features.dtype).expand(len(features), -1, -1)
        return self.network(torch.cat((features, code), -1)).squeeze(-1)+2*anchor.to(features.dtype)-1


def categorical(logits, allowed, temperature):
    if logits.shape != allowed.shape or allowed.dtype != torch.bool or temperature <= 0:
        raise ValueError('Aligned causal availability and positive fixed temperature required')
    raw = torch.cat((torch.zeros_like(logits[:, :1]), logits.masked_fill(~allowed, -1e4)), 1)
    soft = torch.softmax(raw/temperature, 1)
    winner = raw.argmax(1)
    hard = torch.nn.functional.one_hot(winner, raw.shape[1]).to(raw.dtype)
    straight_through = hard+(soft-soft.detach())
    return straight_through[:, 1:], winner-1


def supervised_terms(labels, envelope):
    y, env = np.asarray(labels, float), np.asarray(envelope, float)
    if y.shape != (*env.shape, 5) or not np.isfinite(env).all() or (env < 0).any():
        raise ValueError('Aligned training moments and complete causal envelopes required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any():
        raise ValueError('Unknown rows stay unknown, not zero-error labels')
    clean = np.where(known[:, None, None], y, 0.)
    missing = np.where(known[:, None], 0., env)
    gain = clean[..., 0]-clean[..., 1]-missing
    risk = np.stack((clean[..., 1]-.02*clean[..., 2]+missing,
                     clean[..., 4]-.02*clean[..., 3]+missing), -1)
    return gain, risk, clean[..., 3], known


def prepare(features, anchor, allowed, labels, envelope, scales, sites, recordings, *, arm, identity):
    features, anchor, allowed, scales, sites, records = map(np.asarray,
        (features, anchor, allowed, scales, sites, recordings))
    if (arm not in ARMS or features.shape != (*allowed.shape, 12) or allowed.shape[1] != 7
            or anchor.shape != allowed.shape or anchor.dtype != bool or allowed.dtype != bool
            or not np.isfinite(features).all() or scales.shape != (len(features),)
            or (scales <= 0).any() or not np.isfinite(scales).all()):
        raise ValueError('Finite fitting-only causal action features and legal bank required')
    gain, risk, easy_reference, known = supervised_terms(labels, envelope)
    actions = np.asarray(ARMS[arm]); available = allowed[:, actions]
    gain = gain[:, actions]/scales[:, None]
    risk = risk[:, actions]/scales[:, None, None]
    easy_reference = easy_reference[:, actions]/scales[:, None]
    groups, keys, weights, risk_scales, support_min = [], [], [], [], []
    names = sorted(set(sites))
    for source in names:
        source_count = int((sites == source).sum())
        for record in sorted(set(records[sites == source].astype(str))):
            ix = np.flatnonzero((sites == source) & (records.astype(str) == record))
            candidates = np.where(available[ix], easy_reference[ix], 0.)
            positive = candidates[candidates > 0]
            if not len(positive):
                raise ValueError('No real known easy-reference support in required fitting recording')
            # One genuine easy-row equivalent; never an identical-floor pseudo-action.
            minimum = float(positive.min())/len(ix)
            reference = np.where(known[ix], labels[ix, 0, 2]/scales[ix], 0.)
            easy = np.where(known[ix], labels[ix, 0, 3]/scales[ix], 0.)
            causal = float(np.mean(envelope[ix]/scales[ix, None]))
            risk_scales.append([max(float(reference.mean()), causal*1e-3, 1e-6),
                                max(float(easy.mean()), minimum, 1e-6)])
            groups.append(torch.from_numpy(ix)); keys.append((str(source), str(record)))
            weights.append(len(ix)/(len(names)*source_count)); support_min.append(minimum)
    row_record = np.empty(len(features), np.int64)
    for index, group in enumerate(groups):
        row_record[group.numpy()] = index
    return dict(features=torch.from_numpy(np.clip(features[:, actions], -8, 8).astype(np.float32)),
        anchor=torch.from_numpy(anchor[:, actions]), allowed=torch.from_numpy(available),
        gain=torch.from_numpy(gain.astype(np.float32)), risk=torch.from_numpy(risk.astype(np.float32)),
        easy_reference=torch.from_numpy(easy_reference.astype(np.float32)), action_ids=torch.from_numpy(actions),
        row_record=torch.from_numpy(row_record), groups=groups, keys=keys,
        weights=torch.tensor(weights, dtype=torch.float32), sizes=torch.tensor([len(g) for g in groups]),
        risk_scales=torch.tensor(risk_scales, dtype=torch.float32),
        support_min=torch.tensor(support_min, dtype=torch.float32), identity=identity)


def full_pass(model, data, dual, temperature, chunk_size, *, backward=False):
    count = len(data['keys']); gain = torch.zeros(count, dtype=torch.float64)
    risk = torch.zeros((count, 2), dtype=torch.float64)
    support = torch.zeros(count, dtype=torch.float64); loss_value = 0.; actions_total = 0
    row_weight = data['weights']/data['sizes']
    for start in range(0, len(data['features']), chunk_size):
        end = start+chunk_size; rec = data['row_record'][start:end]
        with torch.set_grad_enabled(backward):
            logit = model(data['features'][start:end], data['anchor'][start:end], data['action_ids'])
            action, _ = categorical(logit, data['allowed'][start:end], temperature)
            gain_term = (action*data['gain'][start:end]).sum(1)
            risk_term = (action[..., None]*data['risk'][start:end]).sum(1)
            support_term = (action*data['easy_reference'][start:end]).sum(1)
            scaled_risk = risk_term/data['risk_scales'][rec]
            scaled_support = (data['support_min'][rec]-support_term)/data['risk_scales'][rec, 1]
            objective = (-gain_term+(dual[rec, :2].detach()*scaled_risk).sum(1)
                         +dual[rec, 2].detach()*scaled_support)
            loss = (objective*row_weight[rec]).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite exact whole-record decision objective')
            if backward:
                loss.backward()
        gain.index_add_(0, rec, gain_term.detach().double())
        risk.index_add_(0, rec, risk_term.detach().double())
        support.index_add_(0, rec, support_term.detach().double())
        loss_value += float(loss.detach()); actions_total += int(action.detach().sum())
    sizes = data['sizes'].double()
    risk = risk/sizes[:, None]; support = support/sizes
    excess = torch.cat((risk/data['risk_scales'],
                        ((data['support_min']-support)/data['risk_scales'][:, 1])[:, None]), 1)
    return dict(loss=loss_value, gain=gain/sizes, risk=risk, support=support,
                excess=excess.float(), actions=actions_total)


def fit(data, *, arm, settings, seed, path, heartbeat=lambda **_: None, guard=lambda: None,
        resume=False, stop_at=None):
    model = RecordActionPolicy(settings['width'], seed); initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    dual = torch.zeros((len(data['keys']), 3)); step, elapsed, trace = 0, 0., []
    binding = dict(arm=arm, settings=settings, seed=seed, identity=data['identity'], initial_model=initial,
        record_keys=data['keys'], record_sizes=data['sizes'], record_risk_scales=data['risk_scales'],
        minimum_real_easy_support=data['support_min'], parameters=sum(p.numel() for p in model.parameters()))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        state = read_checkpoint(path)
        for key, value in binding.items():
            exact(state[key], value)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        step, elapsed, trace, dual = (state[k] for k in ('step', 'seconds', 'trace', 'dual'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing full-gradient update budget required')
    started = time.monotonic()

    def save():
        guard()
        save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state(),
            dual=dual, trace=trace, full_gradient_rows=step*len(data['features']),
            independent_roles_opened=False, inner_held_used=False))

    if step == 0:
        save()
    while step < limit:
        temperature = 1.-.9*step/max(1, settings['steps']-1)
        opt.zero_grad(set_to_none=True)
        stats = full_pass(model, data, dual, temperature, settings['chunk_size'], backward=True)
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        dual = (dual+settings['dual_learning_rate']*stats['excess']).clamp(0, settings['dual_maximum'])
        step += 1
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, objective=stats['loss'], gradient_norm=float(norm),
                mean_gain_lower=float((data['weights']*stats['gain']).sum()),
                selected_rows=stats['actions'], positive_exact_constraint_count=int((stats['excess'] > 0).sum()),
                exact_excess=stats['excess'].tolist(), dual=dual.tolist())
            trace.append(row); heartbeat(state='training', arm=arm,
                **{k:v for k,v in row.items() if k not in ('exact_excess', 'dual')})
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return read_checkpoint(path)


def predict(state, features, anchor, allowed, *, arm, chunk_size=8192):
    if arm != state['arm']:
        raise ValueError('Frozen action bank required')
    model = RecordActionPolicy(state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval(); actions = torch.tensor(ARMS[arm])
    choice, rank = [], []
    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            end = start+chunk_size
            f = torch.from_numpy(np.clip(features[start:end, actions], -8, 8).astype(np.float32))
            a = torch.from_numpy(anchor[start:end, actions]); ok = torch.from_numpy(allowed[start:end, actions])
            logits = model(f, a, actions); _, selected = categorical(logits, ok, .1)
            ix = selected.numpy(); active = ix >= 0
            global_choice = np.full(len(ix), -1, np.int64); global_choice[active] = actions.numpy()[ix[active]]
            chosen_rank = np.zeros(len(ix)); chosen_rank[active] = logits.numpy()[active, ix[active]]
            choice.append(global_choice); rank.append(chosen_rank)
    return np.concatenate(choice), np.concatenate(rank)
