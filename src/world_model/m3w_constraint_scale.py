"""TRAIN-fixed positive constraint scaling and frozen-model gradient diagnosis."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')

import numpy as np
import torch

from src.world_model import m3w_task_risk_training as old


def train_budget_scales(target, sites, recordings, frames, reference_scale):
    target = np.asarray(target)
    if (target.ndim != 2 or target.shape[1] != 5 or not np.isfinite(reference_scale)
            or reference_scale <= 0):
        raise ValueError('Five TRAIN costs and positive frozen scale required')
    known = np.isfinite(target).all(1)
    if (not (known | np.isnan(target).all(1)).all() or (target[known] < 0).any()):
        raise ValueError('Unknown labels remain unknown; no partial imputation')
    groups, sources, _ = old.core.sampling.query_groups(sites, recordings, frames, known)
    query = np.array([(target[g][:, [2, 3]]/reference_scale).mean(0) for g in groups])
    result = .02*np.array([query[g].mean(0) for g in sources]).mean(0)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError('No TRAIN reference support; scaling is undefined')
    return result.astype(np.float32)


def scaled_excess(excess, scales):
    if (excess.shape != (2,) or scales.shape != (2,) or scales.requires_grad
            or not torch.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError('Fixed positive TRAIN scales required')
    return excess/scales


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)]).detach()


def cosine(a, b):
    denominator = float(a.norm()*b.norm())
    return float(a@b)/denominator if denominator > 0 else None


def frozen_monitor(state, args, eligible):
    x, envelope, target, series, sites, recordings, frames, pr = args
    old.core.exact(pr, state['preprocess'])
    temporal, _ = old.parent.temporal.validate_series(series)
    for k, a in dict(x=x, envelope=envelope, primary=target, temporal=temporal,
                     sites=sites, recordings=recordings, frames=frames).items():
        if old.parent.forest.fingerprint(np.asarray(a)) != state['input_hashes'][k]:
            raise ValueError('TRAIN input differs from frozen fit')
    if old.parent.forest.fingerprint(eligible) != state['eligibility_sha256']:
        raise ValueError('Causal eligibility differs')
    known = np.isfinite(target).all(1)
    groups, sources, _ = old.core.sampling.query_groups(sites, recordings, frames, known)
    ix, seg, qids = old.core.sampling.draw_queries(
        groups, sources, 128, torch.Generator().manual_seed(state['seed']+9137))
    z = torch.from_numpy(np.clip((x[ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope[ix]/pr['scale']).astype(np.float32))
    truth = torch.from_numpy((target[ix]/pr['scale']).astype(np.float32))
    mask = torch.from_numpy(eligible[ix]); rms = torch.from_numpy(pr['rms'])
    segments = torch.from_numpy(seg)
    scales = torch.from_numpy(train_budget_scales(target, sites, recordings, frames, pr['scale']))
    model = old.parent.initialize(pr, state['settings'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    p, _ = model(z, env)
    primary = old.core.losses(p, truth, segments, 128, rms)['total']
    task = old.decision.terms(p, truth, segments, 128, mask, rms,
                              old.temperature(state['step'], state['settings'], state['objective_config']))
    replay = dict(primary_total=primary, decision_regret=task['regret'],
                  soft_all_excess=task['risk_excess'][0], soft_easy_excess=task['risk_excess'][1],
                  soft_intervention=task['action_mean'])
    saved = state['trace'][-1]['monitor']
    for k, value in replay.items():
        np.testing.assert_allclose(float(value.detach()), saved[k], rtol=2e-6, atol=2e-7)
    pieces = dict(primary=primary, regret=task['regret'],
                  all_excess=task['risk_excess'][0], easy_excess=task['risk_excess'][1])
    params = list(model.parameters())
    gradients = {k: gradient_vector(v, params) for k, v in pieces.items()}
    data_gradient = gradients['primary']+gradients['regret']
    dual = state['dual']
    weighted = dual[0]*gradients['all_excess']+dual[1]*gradients['easy_excess']
    raw_easy_norm = float((dual[1]*gradients['easy_excess']).norm())
    data_norm = float(data_gradient.norm())
    sign = old.core.signed(p)
    hard = mask & (sign[:, 0] > 0) & (sign[:, 1:] <= 0).all(1)
    def mean(v): return float(old.decision.query_mean(v, segments, 128)[0].detach())
    hard_excess = old.decision.query_mean(hard[:, None]*torch.stack(
        (truth[:, 1]-.02*truth[:, 2], truth[:, 4]-.02*truth[:, 3]), 1), segments, 128)
    # Only input-independent gradient summaries leave the private TRAIN packet.
    normed = scaled_excess(task['risk_excess'], scales)
    return dict(identity=state['identity'], arm=state['arm'], step=state['step'],
        monitor_queries=128, monitor_row_occurrences=len(ix),
        query_ids_hash=old.parent.forest.fingerprint(qids),
        monitor_replayed=True, TRAIN_budget_scales=scales.tolist(),
        raw_excess=task['risk_excess'].detach().tolist(), scaled_excess=normed.detach().tolist(),
        hard_excess=hard_excess.detach().tolist(),
        gradient_norms={k:float(g.norm()) for k,g in gradients.items()},
        weighted_risk_gradient_norm=float(weighted.norm()),
        weighted_easy_gradient_over_data_gradient=raw_easy_norm/data_norm if data_norm else None,
        weighted_risk_gradient_over_data_gradient=float(weighted.norm())/data_norm if data_norm else None,
        data_risk_gradient_cosine=cosine(data_gradient, weighted),
        regret_easy_gradient_cosine=cosine(gradients['regret'], gradients['easy_excess']),
        dual=dual.tolist(), soft_action_mean=mean(task['action']), hard_action_mean=mean(hard.float()),
        soft_half_hard_disagreement=mean(((task['action'] >= .5) != hard).float()),
        hard_selected_occurrences=int(hard.sum()),
        hard_selected_soft_mean=float(task['action'][hard].mean().detach()) if hard.any() else None,
        soft_action_is_calibrated_probability=False, new_training=False, independent_roles_read=False)
