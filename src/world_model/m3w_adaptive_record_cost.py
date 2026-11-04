"""Current-selection training mask; unchanged causal network and excess loss."""
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.world_model import m3w_record_excess_cost as frozen
from src.world_model.m3w_harm_refinement import digest, select_actions

base, parent = frozen.base, frozen.parent
training_packet = frozen.training_packet
exact_scientific_state = frozen.exact_scientific_state
ARMS = ('frozen_teacher', 'adaptive_selection')


def objective(prediction, data, ix, arm):
    if arm not in ARMS:
        raise ValueError('Fixed teacher versus detached current selection required')
    if arm == 'frozen_teacher':
        actions = data['teacher_actions'][ix]
    else:
        raw = prediction.detach().cpu().numpy().astype(float) * data['causal']['scales'][ix, None, None]
        actions = select_actions(raw, data['causal']['allowed'][ix], arm='physical_neural')[0]
    # A loss-only minibatch view preserves repeated sampled rows separately.
    loss_view = dict(teacher_actions=actions, record_codes=data['record_codes'][ix],
        record_source_codes=data['record_source_codes'], causal=dict(scales=data['causal']['scales'][ix]),
        truth_by_arm=dict(bound_aware=data['truth_by_arm']['bound_aware'][ix]), preprocess=data['preprocess'])
    terms = frozen.record_excess(prediction, loss_view, np.arange(len(ix)))
    terms['selected_training_rows'] = terms.pop('teacher_selected_rows')
    terms['training_action_counts'] = np.bincount(actions+1, minlength=8).tolist()
    return terms, actions


def fit(data, *, loss_arm, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if loss_arm not in ARMS:
        raise ValueError('Registered selection-mask objective arm required')
    c, pr = data['causal'], data['preprocess']
    model = base.base.CostHead(c['x'].shape[-1], settings['width'], seed, pr['bias'], 'context')
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, arm='context', target_arm='bound_aware', loss_arm=loss_arm,
        loss_weight=1., positive_harm_budget=.02, seed=seed, preprocess=pr, initial_model=initial,
        parameters=sum(p.numel() for p in model.parameters()), budget=budget,
        source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy(),
        teacher_sha256=digest(data['teacher_actions']), record_codes_sha256=digest(data['record_codes']),
        targets_sha256=digest(data['truth_by_arm']['bound_aware'].numpy()),
        standardized_features_sha256=digest(data['z'].numpy()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws, selection_draws, observed_draws, bound_draws = 0, 0., [], '0'*64, '0'*64, 0, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged adaptive resume required')
        old = base.base.read_checkpoint(path)
        for key, value in binding.items():
            base.base.exact(old[key], value)
        model.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler'])
        torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, selection_draws, observed_draws, bound_draws = (
            old[k] for k in ('step', 'seconds', 'trace', 'draw_hash', 'selection_draw_hash',
                            'observed_supervised_draws', 'unknown_bound_supervised_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('No zero-update or rolled-back result')
    start = time.monotonic()

    def save():
        guard()
        base.base.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=opt.state_dict(),
            sampler=sampler.state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=elapsed+time.monotonic()-start, trace=trace, draw_hash=draws,
            selection_draw_hash=selection_draws, observed_supervised_draws=observed_draws,
            unknown_bound_supervised_draws=bound_draws, unknown_rows_mislabeled_as_observed=0))

    if step == 0:
        save()
    try:
        while step < limit:
            guard()
            ix, seg, qids = sampler.draw()
            segment = torch.from_numpy(seg)
            envelope = torch.from_numpy((c['envelope'][ix]/c['scales'][ix, None]).astype(np.float32))
            opt.zero_grad(set_to_none=True)
            prediction = model(data['z'][ix], envelope, segment, len(qids))
            terms = base.objective(prediction, data, ix, segment, len(qids), 'bound_aware')
            extra, actions = objective(prediction, data, ix, loss_arm)
            total = terms['total'] + extra['record_excess']
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite adaptive cost objective')
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            step += 1
            observed_draws += terms['observed_supervised_rows']
            bound_draws += terms['unknown_bound_supervised_rows']
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            selection_draws = hashlib.sha256(bytes.fromhex(selection_draws)+actions.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, **frozen.trace_losses(terms, extra, total),
                    gradient_norm=float(norm), minimum_query_visits=int(sampler.counts.min()),
                    never_drawn_queries=int((sampler.counts == 0).sum()), selection_draw_hash=selection_draws)
                trace.append(row)
                heartbeat(state='adaptive_record_cost_training', loss_arm=loss_arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save()
        raise
    result = base.base.read_checkpoint(path)
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Complete every fitting query coverage budget')
    return result
