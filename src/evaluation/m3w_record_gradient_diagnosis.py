"""Frozen TRAIN-state diagnostics; no outcome labels become inference features."""
import numpy as np
import torch
from src.world_model import m3w_multiaction_record_policy as policy
from src.world_model.m3w_record_support_units import installed


def vector(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()]).double().numpy()


def cosine(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator > 0 else None


def gradient_components(state, data):
    model = policy.RecordActionPolicy(state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    temperature = .1
    gradients, stats = {}, None
    with installed():
        for name, channels in (('utility', ()), ('all', (0,)), ('easy', (1,)),
                               ('support', (2,)), ('total', (0, 1, 2))):
            dual = torch.zeros_like(state['dual'])
            if channels:
                dual[:, list(channels)] = state['dual'][:, list(channels)]
            model.zero_grad(set_to_none=True)
            current = policy.full_pass(model, data, dual, temperature,
                state['settings']['chunk_size'], backward=True)
            gradients[name] = vector(model)
            if name == 'total':
                stats = current
    utility = gradients['utility']
    components = {name: gradients[name]-utility for name in ('all', 'easy', 'support')}
    reconstructed = utility+sum(components.values())
    error = np.linalg.norm(reconstructed-gradients['total'])
    scale = max(np.linalg.norm(gradients['total']), 1.)
    if error > 2e-4*scale:
        raise ValueError('Linear frozen-gradient decomposition does not reproduce')
    excess = stats['excess'].numpy(); dual = state['dual'].numpy()
    return dict(selected_rows=stats['actions'], exact_record_views=len(excess),
        record_weight_quantiles=np.quantile(data['weights'].numpy(), [0, .5, 1]).tolist(),
        gradient_norm={name:float(np.linalg.norm(v)) for name,v in
                       dict(utility=utility, **components, total=gradients['total']).items()},
        support_cosine_with_risk=cosine(components['support'], components['all']+components['easy']),
        utility_cosine_with_risk=cosine(utility, components['all']+components['easy']),
        combined_gradient_clip_multiplier=min(1., state['settings']['gradient_clip']/
            max(float(np.linalg.norm(gradients['total'])), 1e-30)),
        decomposition_relative_error=float(error/scale),
        constraints={name:dict(violations=int((excess[:, k] > 0).sum()),
            positive_with_dual_below_one=int(((excess[:, k] > 0) & (dual[:, k] < 1)).sum()),
            dual_at_cap=int((dual[:, k] >= state['settings']['dual_maximum']).sum()),
            negative_excess_below_minus_one=int((excess[:, k] < -1).sum()),
            positive_dual_with_negative_excess_below_minus_one=int(((dual[:, k] > 0) & (excess[:, k] < -1)).sum()),
            excess_quantiles=np.quantile(excess[:, k], [0, .25, .5, .75, 1]).tolist(),
            dual_quantiles=np.quantile(dual[:, k], [0, .5, 1]).tolist())
            for k,name in enumerate(('all', 'easy', 'support'))},
        any_constraint_violations=int((excess > 0).any(1).sum()),
        full_supported=int((excess <= 0).all(1).sum()),
        outcome_labels_used_only_for_frozen_training_objective=True,
        no_optimizer_step=True, temperature=temperature)


def reference_consistency(prediction, labels, selected):
    prediction, labels, selected = map(np.asarray, (prediction, labels, selected))
    known = np.isfinite(labels).all((1, 2))
    for k in (2, 3):
        if not np.array_equal(labels[known, :, k], np.repeat(labels[known, :1, k], 7, axis=1)):
            raise ValueError('Reference target must be candidate-invariant')
    result = {}
    for k, name in ((2, 'reference'), (3, 'easy_reference')):
        p = prediction[..., k]
        spread = np.ptp(p, axis=1)/np.maximum(p.mean(1), 1e-20)
        result[name] = dict(target_candidate_invariant=True,
            predicted_relative_spread_quantiles=np.quantile(spread, [0, .25, .5, .75, 1]).tolist(),
            rows_over_one_percent_spread=int((spread > .01).sum()), rows=len(p))
    active = known & (selected >= 0)
    if active.any():
        ix = selected[active]
        actual = labels[active, ix].sum(0); predicted = prediction[active, ix].sum(0)
        result['known_selected'] = dict(rows=int(active.sum()), predicted=predicted.tolist(), actual=actual.tolist(),
            predicted_over_actual=[float(p/a) if a > 0 else None for p,a in zip(predicted, actual)])
    return result
