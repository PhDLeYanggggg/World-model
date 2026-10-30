"""One-factor selected-query score units; unchanged point loss and architecture."""
from contextlib import contextmanager
import platform
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import numpy as np
import torch
from src.world_model import m3w_selected_query_critic as parent

ORIGINAL_LOSS = parent.losses


def query_normalizer(data):
    c = data['causal']; second = {}; counts = {}; unknown = empty = complete = 0
    for key, group in zip(c['keys'], c['groups']):
        source = key[0]; ix = group[c['initial'][group] >= 0]
        if not len(ix): empty += 1; continue
        if not data['known'][ix].all(): unknown += 1; continue
        y = data['targets'][ix, c['initial'][ix]]
        signed = (y[:, [1, 4]]-.02*y[:, [2, 3]]).sum(0)/c['scales'][ix].sum()
        if source not in second: second[source] = np.zeros(2); counts[source] = 0
        second[source] += signed**2; counts[source] += 1; complete += 1
    if not second: raise ValueError('No nonempty complete selected-query fitting targets')
    value = np.sqrt(np.mean([second[k]/counts[k] for k in sorted(second)], 0))
    return dict(rms=np.maximum(value, 1e-6).tolist(), rms_before_floor=value.tolist(),
        sources_with_selected_support=len(second), complete_selected_queries=complete,
        unknown_selected_queries=unknown, empty_selected_queries=empty,
        source_complete_query_counts=counts, fitting_labels_only=True, inference_feature=False,
        normalization='equal_source_conditional_selected_query_signed_target_RMS', floor=1e-6)


def query_error(prediction, data, rows, segments, count):
    c = data['causal']; known = torch.from_numpy(data['known'][rows])
    scales = torch.from_numpy(c['scales'][rows])
    target = torch.from_numpy(np.where(data['known'][rows, None, None],
        data['targets'][rows]/c['scales'][rows, None, None], 0.))
    selected = torch.from_numpy(c['initial'][rows] >= 0)
    action = torch.from_numpy(c['initial'][rows]).clamp_min(0); ix = torch.arange(len(rows))
    pred, truth = prediction[ix, action], target[ix, action]
    signed_error = pred[:, [1, 4]]-.02*pred[:, [2, 3]]-truth[:, [1, 4]]+.02*truth[:, [2, 3]]
    raw_error = signed_error*scales[:, None]*selected[:, None]
    total = prediction.new_zeros((count, 2)).index_add(0, segments, raw_error)
    mass = prediction.new_zeros(count).index_add(0, segments, scales*selected)
    selected_n = prediction.new_zeros(count).index_add(0, segments, selected.to(prediction.dtype))
    missing_n = prediction.new_zeros(count).index_add(0, segments, (selected & ~known).to(prediction.dtype))
    return total/mass.clamp_min(1e-150)[:, None], (selected_n > 0) & (missing_n == 0)


@contextmanager
def installed(normalizer, mode):
    if mode not in ('all_action', 'selected_query'): raise ValueError('Registered units required')
    previous = parent.losses
    def objective(prediction, data, rows, segments, count):
        terms = ORIGINAL_LOSS(prediction, data, rows, segments, count)
        error, complete = query_error(prediction, data, rows, segments, count)
        rms = torch.from_numpy(data['rms'])
        old_scale = torch.sqrt(rms[[1, 4]].square()+(.02*rms[[2, 3]]).square())
        original = ((error/old_scale).square().mean(1)*complete).mean()
        if not torch.equal(original, terms['selected_query_score']):
            raise ValueError('Unmodified query arithmetic must reproduce exactly')
        if mode == 'all_action': return terms
        scale = torch.as_tensor(normalizer['rms'], dtype=prediction.dtype, device=prediction.device)
        value = ((error/scale).square().mean(1)*complete).mean()
        return dict(terms, total=.5*(terms['pointwise_risk']+value), selected_query_score=value,
            selected_query_score_original_units=original)
    parent.losses = objective
    try: yield
    finally: parent.losses = previous


def fit(data, *, normalization, identity, **kwargs):
    normalizer = query_normalizer(data)
    with installed(normalizer, normalization):
        return parent.fit(data, identity=dict(identity, query_unit_normalizer=normalizer,
            query_units_mode=normalization), **kwargs)
