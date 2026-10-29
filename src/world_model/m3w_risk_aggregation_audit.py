"""Frozen TRAIN-monitor audit of query-mean versus agent-mass feasibility."""
import numpy as np

from src.world_model import m3w_constraint_scale as scale

torch, old = scale.torch, scale.old


def compare_aggregation(target, action, segments, count):
    target, action, segments = map(np.asarray, (target, action, segments))
    if (target.ndim != 2 or target.shape[1] != 5 or len(target) == 0
            or action.shape != (len(target),) or segments.shape != action.shape
            or not np.issubdtype(segments.dtype, np.integer) or count <= 0
            or not np.isfinite(target).all() or (target < 0).any()
            or not np.isfinite(action).all() or ((action < 0) | (action > 1)).any()
            or (segments < 0).any() or (segments >= count).any()):
        raise ValueError('Known aligned TRAIN costs and bounded actions required')
    sizes = np.bincount(segments, minlength=count)
    if (sizes == 0).any():
        raise ValueError('Missing query cannot be dropped')
    result = {}
    for name, harm, ref in [('all', 1, 2), ('easy', 4, 3)]:
        numerator = action * target[:, harm]
        denominator = action * target[:, ref]
        excess = numerator - .02 * denominator
        query_excess = float(np.mean(np.bincount(segments, weights=excess) / sizes))
        mass_excess = float(excess.mean())
        risk = float(numerator.sum()/denominator.sum()) if denominator.sum() > 0 else None
        result[name] = dict(query_mean_excess=query_excess, agent_mass_mean_excess=mass_excess,
            agent_mass_risk=risk, query_pass_mass_fail=query_excess <= 0 < mass_excess,
            mass_pass_query_fail=mass_excess <= 0 < query_excess,
            reference_support=risk is not None)
    return dict(constraints=result, query_count=count, row_occurrences=len(target),
        query_size_min=int(sizes.min()), query_size_max=int(sizes.max()),
        nonzero_action_occurrences=int((action > 0).sum()),
        empty_action=not bool((action > 0).any()))


def frozen_monitor(state, args, eligible):
    x, envelope, target, series, sites, recordings, frames, pr = args
    old.core.exact(pr, state['preprocess'])
    temporal, _ = old.parent.temporal.validate_series(series)
    for name, value in dict(x=x, envelope=envelope, primary=target, temporal=temporal,
                           sites=sites, recordings=recordings, frames=frames).items():
        if old.parent.forest.fingerprint(np.asarray(value)) != state['input_hashes'][name]:
            raise ValueError('TRAIN input hash mismatch')
    if old.parent.forest.fingerprint(eligible) != state['eligibility_sha256']:
        raise ValueError('Causal eligibility changed')
    known = np.isfinite(target).all(1)
    groups, sources, _ = old.core.sampling.query_groups(sites, recordings, frames, known)
    ix, seg, qids = old.core.sampling.draw_queries(
        groups, sources, 128, torch.Generator().manual_seed(state['seed']+9137))
    z = torch.from_numpy(np.clip((x[ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope[ix]/pr['scale']).astype(np.float32))
    truth = torch.from_numpy((target[ix]/pr['scale']).astype(np.float32))
    mask, rms, segments = torch.from_numpy(eligible[ix]), torch.from_numpy(pr['rms']), torch.from_numpy(seg)
    model = old.parent.initialize(pr, state['settings'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    with torch.no_grad():
        prediction, _ = model(z, env)
        task = old.decision.terms(prediction, truth, segments, 128, mask, rms,
            old.temperature(state['step'], state['settings'], state['objective_config']))
        saved = state['trace'][-1]['monitor']
        np.testing.assert_allclose(task['risk_excess'].numpy(),
            [saved['soft_all_excess'], saved['soft_easy_excess']], rtol=2e-6, atol=2e-7)
        signs = old.core.signed(prediction)
        hard = mask & (signs[:, 0] > 0) & (signs[:, 1:] <= 0).all(1)
        results = {name:compare_aggregation(truth.numpy().astype(float), action.numpy(), seg, 128)
                   for name, action in [('soft', task['action']), ('hard', hard)]}
    return dict(identity=state['identity'], monitor_replayed=True, monitor_queries=128,
        query_ids_hash=old.parent.forest.fingerprint(qids), aggregations=results,
        TRAIN_known_only=True, full_TRAIN_estimate=False, validation_read=False,
        independent_roles_read=False, new_training=False, deployment_changed=False)
