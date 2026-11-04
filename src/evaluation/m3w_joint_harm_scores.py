"""Same-row, query-balanced event and moment diagnostics, not risk certificates."""
import numpy as np


def scores(prediction, truth, known, causal, choice, prior):
    p, y = prediction['moments'], truth
    event = np.zeros(y.shape[:2], int)
    event[y[..., 0] > 0] = 1
    event[y[..., 1] > 0] = 2
    event[y[..., 4] > 0] = 3
    probability = prediction['probability']
    result = {}
    for region in ('all_legal', 'fixed_proposal'):
        mask = causal['allowed'] & known[:, None]
        if region == 'fixed_proposal':
            selected = np.zeros_like(mask)
            ix = np.flatnonzero(choice >= 0); selected[ix, choice[ix]] = True
            mask &= selected
        weights = np.zeros(mask.shape, float)
        supported = 0
        for group in causal['groups']:
            rows = group[mask[group].any(1)]
            if not len(rows): continue
            supported += 1
            weights[rows] = mask[rows]/mask[rows].sum(1)[:, None]/len(rows)
        if not supported:
            result[region] = dict(status='not_estimable_no_known_legal_actions', queries=0)
            continue
        weights /= supported
        envelope = causal['envelope'].clip(1e-12)
        reference_scale = causal['scales'][:, None]
        row = dict(status='estimated_available_labels_only', queries=supported, known_actions=int(mask.sum()),
            H_fraction_MSE=float(np.sum(weights*((p[..., 1]-y[..., 1])/envelope)**2)),
            EH_fraction_MSE=float(np.sum(weights*((p[..., 4]-y[..., 4])/envelope)**2)),
            R_scaled_MSE=float(np.sum(weights*((p[..., 2]-y[..., 2])/reference_scale)**2)))
        for name, target, prob, constant in (
            ('H', event >= 2, probability[..., 2:].sum(-1), prior[:, 2:].sum(-1)[None]),
            ('EH', event == 3, probability[..., 3], prior[:, 3][None])):
            row[name+'_Brier'] = float(np.sum(weights*(prob-target)**2))
            row[name+'_fitting_prior_Brier'] = float(np.sum(weights*(constant-target)**2))
            row[name+'_positive_rate'] = float(np.sum(weights*target))
            clipped = prob.clip(1e-7, 1-1e-7)
            row[name+'_log_loss'] = float(-np.sum(weights*(target*np.log(clipped)+(1-target)*np.log1p(-clipped))))
        result[region] = row
    return result


def fitting_prior(learning):
    c = learning['causal']; y = learning['truth_by_arm']['bound_aware'].numpy()
    event = np.zeros(y.shape[:2], int)
    event[y[..., 0] > 0] = 1; event[y[..., 1] > 0] = 2; event[y[..., 4] > 0] = 3
    legal = c['allowed'] & learning['observed_future'][:, None]
    numerator, denominator = np.zeros((7, 4)), np.zeros(7)
    for source in c['source_groups']:
        for qi in source:
            group = c['groups'][qi]
            for action in range(7):
                rows = group[legal[group, action]]
                if not len(rows): continue
                weight = 1/len(source)
                numerator[action] += weight*np.bincount(event[rows, action], minlength=4)/len(rows)
                denominator[action] += weight
    prior = np.zeros((7, 4)); prior[:, 0] = 1
    take = denominator > 0; prior[take] = numerator[take]/denominator[take, None]
    return prior
