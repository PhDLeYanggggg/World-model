"""Uniform record constraint weights; unchanged source-weighted utility."""
import copy
from pathlib import Path
import time

from src.world_model import m3w_cost_sensitive_policy as previous

base, torch = previous.base, previous.torch


def constraint_factor(weights, balanced=True):
    if (weights.ndim != 1 or not len(weights) or not torch.isfinite(weights).all()
            or not (weights > 0).all() or abs(float(weights.sum())-1.) > 1e-6):
        raise ValueError('Positive normalized record utility weights required')
    return torch.full_like(weights, 1./len(weights))/weights if balanced else torch.ones_like(weights)


def full_pass(model, data, loc, dual, temperature, chunk_size, backward=False, balanced=True):
    # The old row loss already includes utility weight w_j. q_j/w_j changes
    # only its constraint coefficient, leaving the gain term untouched.
    effective = dual * constraint_factor(data['weights'], balanced)[:, None]
    return previous.full_pass(model, data, loc, effective, temperature, chunk_size, backward)


def fit(data, *, settings, seed, identity, path, heartbeat=lambda **_:None,
        resume=False, stop_at=None, guard=lambda:None, balanced=True):
    loc = previous.whole.layout(data)
    model = base.Policy(data['z'].shape[1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    dual = torch.zeros((len(data['groups']),2))
    factor = constraint_factor(data['weights'], balanced)
    binding = dict(arm='record_balanced' if balanced else 'neutral_replay', settings=settings,
        seed=seed, identity=identity, initial_model=initial, input_hashes=data['input_hashes'],
        source_preprocess=data['source_preprocess'], record_keys=data['record_keys'],
        record_sizes=data['sizes'], record_scales=data['record_scales'],
        utility_weights=data['weights'], constraint_factor=factor,
        empty_training_records=data['empty_training_records'], dimension=data['z'].shape[1],
        parameters=sum(p.numel() for p in model.parameters()))
    step, seconds, trace = 0, 0., []
    path = Path(path)
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        state = base.core.read_checkpoint(path)
        for k,v in binding.items(): base.core.exact(state[k],v)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,dual = (state[k] for k in ('step','seconds','trace','dual'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing update budget required')
    began = time.monotonic()

    def save():
        guard(); path.parent.mkdir(parents=True,exist_ok=True)
        base.core.save_checkpoint(path, dict(**binding, step=step, optimizer_steps=step,
            seconds=seconds+time.monotonic()-began, trace=trace, dual=dual,
            model=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            full_gradient_rows=step*len(loc['index']),new_forecaster=False,held_used_for_training=False))

    if step == 0: save()
    while step < limit:
        temperature = 1.-.75*step/max(1,settings['steps']-1)
        optimizer.zero_grad(set_to_none=True)
        stats = full_pass(model,data,loc,dual,temperature,settings['chunk_size'],True,balanced)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
        dual = (dual+settings['dual_learning_rate']*(stats['risk']/data['record_scales']).float()).clamp(0,settings['dual_maximum'])
        step += 1
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step,surrogate_objective=stats['surrogate_objective'],
                hard_objective=stats['hard_objective'],gradient_norm=float(norm),
                full_gain_lower=float((data['weights']*stats['gain']).sum()),
                action_fraction=float(stats['action'].mean()),dual=dual.tolist(),
                exact_record_risk=stats['risk'].tolist())
            trace.append(row); heartbeat(state='training',**row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return base.core.read_checkpoint(path)
