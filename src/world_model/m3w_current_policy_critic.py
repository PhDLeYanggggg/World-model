"""Current-decoder query supervision; identical critic and all-action objective."""
from collections import Counter
from contextlib import contextmanager
import platform
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import numpy as np
import torch
from src.world_model import m3w_query_unit_critic as unit
from src.world_model import m3w_critic_allocation as decoder

TARGETS = ('frozen_original', 'current_decoded')


def selected_actions(prediction, causal, rows, segments, count, *, seconds,
                     heartbeat=lambda **_: None, guard=lambda: None):
    if set(causal) != unit.parent.CAUSAL_KEYS:
        raise ValueError('Current selection accepts only declared causal inputs')
    rows = np.asarray(rows); seg = np.asarray(segments.cpu(), np.int64)
    if prediction.shape != (len(rows), 7, 5) or seg.shape != rows.shape or set(seg) != set(range(count)):
        raise ValueError('Complete current-query minibatch required')
    guard(); raw = causal['prediction'][rows].copy(); pred = prediction.detach().cpu().numpy()
    for k in (1, 4): raw[:, :, k] = pred[:, :, k]*causal['scales'][rows, None]
    choice = np.full(len(rows), -1, np.int64); reasons = Counter(); signal_time = time.monotonic()
    for group in range(count):
        ix = np.flatnonzero(seg == group); q = raw[ix]; ok = causal['allowed'][rows[ix]]
        anchor, _ = decoder.base.select_actions(q, ok, arm='physical_neural')
        chosen, info = decoder.base.solve_frame(q, ok, anchor, exact_count=int((anchor >= 0).sum()), seconds=seconds)
        if info['reason'].startswith('solver_status_'):
            raise RuntimeError('Incomplete training-query optimization refuses: '+info['reason'])
        choice[ix] = chosen; reasons[info['reason']] += 1
        if time.monotonic()-signal_time >= 30:
            guard(); heartbeat(state='current_training_query_selection', queries=group+1, total_queries=count)
            signal_time = time.monotonic()
    return choice, dict(query_count=count, solve_frame_function_calls=count,
        reason_counts=dict(reasons), labels_used=False, choices_detached=True)


@contextmanager
def installed(normalizer, target, *, seconds, heartbeat=lambda **_: None, guard=lambda: None):
    if target not in TARGETS: raise ValueError('Registered fixed/current query target required')
    with unit.installed(normalizer, 'selected_query'):
        frozen = unit.parent.losses
        if target == 'frozen_original':
            yield
            return
        def objective(prediction, data, rows, segments, count):
            terms = frozen(prediction, data, rows, segments, count)
            actions, info = selected_actions(prediction, data['causal'], rows, segments, count,
                seconds=seconds, heartbeat=heartbeat, guard=guard)
            supervision = dict(causal=dict(scales=data['causal']['scales'][rows], initial=actions),
                targets=data['targets'][rows], known=data['known'][rows])
            error, complete = unit.query_error(prediction, supervision, np.arange(len(rows)), segments, count)
            scale = torch.as_tensor(normalizer['rms'], dtype=prediction.dtype, device=prediction.device)
            rms = torch.from_numpy(data['rms'])
            original_scale = torch.sqrt(rms[[1, 4]].square()+(.02*rms[[2, 3]]).square())
            value = ((error/scale).square().mean(1)*complete).mean()
            old_units = ((error/original_scale).square().mean(1)*complete).mean()
            return dict(terms, total=.5*(terms['pointwise_risk']+value), selected_query_score=value,
                selected_query_score_original_units=old_units,
                frozen_selected_query_score_original_units=terms['selected_query_score_original_units'],
                frozen_selected_query_score=terms['selected_query_score'],
                frozen_complete_selected_queries=terms['complete_selected_queries'],
                complete_selected_queries=int(complete.sum()),
                current_selected_rows=int((actions >= 0).sum()),
                current_selected_queries_with_unknown_labels=sum(
                    bool(((actions[np.asarray(segments.cpu()) == k] >= 0)
                        & ~data['known'][rows[np.asarray(segments.cpu()) == k]]).any()) for k in range(count)),
                current_selection=info, selected_actions_detached=True)
        unit.parent.losses = objective
        try: yield
        finally: unit.parent.losses = frozen


def fit(data, *, target, identity, seconds=2., heartbeat=lambda **_: None, guard=lambda: None, **kwargs):
    normalizer = unit.query_normalizer(data)
    with installed(normalizer, target, seconds=seconds, heartbeat=heartbeat, guard=guard):
        return unit.parent.fit(data, identity=dict(identity, query_unit_normalizer=normalizer,
            query_units_mode='selected_query', query_score_target=target, training_selector_seconds=seconds),
            heartbeat=heartbeat, guard=guard, **kwargs)
