"""Own-fitting sensitivity to neighbor slots, not a deployment policy."""
from src.world_model import m3w_paired_decision_risk as api
from src.evaluation.m3w_decision_semantics import diagnose

np, torch = api.np, api.torch


def permute_neighbors(x, shift):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (7, 428) or not np.isfinite(x).all():
        raise ValueError('Finite causal 428-column action schema required')
    if not isinstance(shift, int) or not 0 <= shift < 8:
        raise ValueError('Fixed cyclic neighbor-slot permutation required')
    result = x.copy()
    result[..., 32:224] = np.roll(x[..., 32:224].reshape(*x.shape[:2], 8, 8, 3),
        shift, axis=-3).reshape(*x.shape[:2], 192)
    result[..., 224:288] = np.roll(x[..., 224:288].reshape(*x.shape[:2], 8, 8),
        shift, axis=-2).reshape(*x.shape[:2], 64)
    return result


def features(raw, state):
    p, ep = state['preprocess'], state['extra_preprocess']
    z = np.clip((raw-p['mean'])/p['std'], -p['clip'], p['clip']).astype(np.float32)
    z[..., 380:] = 0
    e = np.clip((api.parent.kin.motion_basis(raw)-ep['mean'])/ep['std'], -ep['clip'], ep['clip']).astype(np.float32)
    return torch.from_numpy(np.concatenate((z, e), -1))


def measured(risk, decision, target, legal, observed, weights):
    n = legal.sum(1).clip(1)
    w = weights[:, None]/n[:, None]*legal
    y = target.numpy()
    existing = diagnose(risk, decision, y, legal, observed, weights)
    return dict(mean_MSE=float((((risk-y)**2).mean(-1)*w).sum()),
        joint_cost=existing['total_joint_cost'], false_safe=existing['false_safe_count'],
        false_reject=existing['false_reject_count'], admitted=existing['accepted_known_proposals'])


def audit(state, data, guard=lambda: None):
    c = data['causal']; z = api.parent.validate(data, state); model = api.model_from(state)
    count = len(c['ids'])
    risks, decisions = (np.empty((8, count, 7, 2), np.float32) for _ in range(2))
    clip_counts = np.zeros(8, np.int64)
    for ix, seg in api.parent.batches(c):
        guard()
        for shift in range(8):
            raw = permute_neighbors(c['x'][ix], shift)
            v = features(raw, state)
            if shift == 0:
                torch.testing.assert_close(v, z[ix], rtol=0, atol=0)
            clip_counts[shift] += int((v[..., 32:288].abs() >= state['preprocess']['clip']).any(-1).sum())
            with torch.no_grad():
                r, d = model(v, torch.from_numpy(seg), int(seg.max())+1)
            risks[shift, ix], decisions[shift, ix] = r.numpy(), d.numpy()
    target = (api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        torch.from_numpy(api.parent.kin.parent.normalizer(data['preprocess']))).detach()
    weights = api.parent.query_weights(c, state['sampler']['counts'])
    known = c['allowed'] & data['observed_future'][:, None]
    decisions_original = np.maximum(risks[0], decisions[0]).max(-1) <= 0
    rows = []
    for shift in range(8):
        selected = np.maximum(risks[shift], decisions[shift]).max(-1) <= 0
        metrics = measured(risks[shift], decisions[shift], target, c['allowed'], data['observed_future'], weights)
        rows.append(dict(shift=shift, **metrics,
            changed_known_decisions=int(((selected != decisions_original) & known).sum()),
            changed_known_fraction=float(((selected != decisions_original) & known).sum()/known.sum()) if known.any() else None,
            clipped_neighbor_action_rows=int(clip_counts[shift])))
    average = measured(risks.mean(0), decisions.mean(0), target, c['allowed'], data['observed_future'], weights)
    return dict(context=state['identity']['job'], checkpoint_variant=state.get('paired_arm', state.get('variant')),
        fitting_queries=len(c['groups']), rows=count, known_allowed_actions=int(known.sum()),
        shifts=rows, cyclic_mean=average,
        mean_change_over_nonidentity=float(np.mean([r['changed_known_fraction'] for r in rows[1:]])) if known.any() else None,
        averaging_better_MSE_nonworse_cost=average['mean_MSE'] < rows[0]['mean_MSE'] and average['joint_cost'] <= rows[0]['joint_cost'],
        future_inputs=False, target_mask_only_for_loss=True, fresh_fit=False)
