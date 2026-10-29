"""Complete TRAIN cost-head diagnostics; actions never receive outcome labels."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before numerical imports')

import hashlib
import numpy as np

from src.world_model import m3w_task_risk_training as train
from src.world_model.m3w_false_safe_diagnostic import moving_from_features
from src.world_model.m3w_risk_aggregation_audit import compare_aggregation
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds


def bind_inputs(state, args):
    x, envelope, target, series, sites, recordings, frames, pr = args
    train.core.exact(pr, state['preprocess'])
    temporal, _ = train.parent.temporal.validate_series(series)
    for name, value in dict(x=x, envelope=envelope, primary=target, temporal=temporal,
                           sites=sites, recordings=recordings, frames=frames).items():
        if train.parent.forest.fingerprint(np.asarray(value)) != state['input_hashes'][name]:
            raise ValueError('TRAIN packet differs from frozen fit')
    _, support = train.parent.forest.causal_inputs(x, envelope, pr)
    eligible = moving_from_features(x) & support
    if train.parent.forest.fingerprint(eligible) != state['eligibility_sha256']:
        raise ValueError('Causal input eligibility differs')
    return eligible


def sampling_coverage(state, args):
    _, _, target, _, sites, recordings, frames, _ = args
    known = np.isfinite(target).all(1)
    groups, sources, keys = train.core.sampling.query_groups(sites, recordings, frames, known)
    counts = np.zeros(len(groups), np.int64)
    rng = train.torch.Generator().manual_seed(state['seed']+7919)
    draw_hash = '0'*64; row_draws = 0
    for _ in range(state['step']):
        ix, seg, qids = train.core.sampling.draw_queries(groups, sources, state['settings']['query_batch_size'], rng)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        np.add.at(counts, qids, 1); row_draws += len(ix)
    train.core.exact(rng.get_state(), state['sampler_rng'])
    if draw_hash != state['draw_hash'] or row_draws != state['row_draws']:
        raise ValueError('Sampler replay does not match final fit')
    seen = np.zeros(len(target), bool)
    for group, count in zip(groups, counts): seen[group] = count > 0
    easy_harm = known & (target[:, 4] > 0)
    mass = float(target[easy_harm, 4].sum())
    return dict(queries=len(keys), unseen_queries=int((counts == 0).sum()),
        known_rows=int(known.sum()), unseen_known_rows=int((known & ~seen).sum()),
        easy_positive_harm_rows=int(easy_harm.sum()), unseen_easy_positive_harm_rows=int((easy_harm & ~seen).sum()),
        unseen_easy_harm_mass_fraction=float(target[easy_harm & ~seen, 4].sum()/mass) if mass > 0 else None,
        query_draw_quantiles=np.quantile(counts, [0,.5,.95,1]).tolist(),
        sampler_exact=True, draw_hash=draw_hash, row_draws=row_draws)


def cohort(prediction, target, action, envelope):
    result = completion_bounds(target, action, envelope)
    known = np.isfinite(target).all(1)
    selected = known & action
    predicted = prediction[selected].sum(0); actual = target[selected].sum(0)
    values = np.sort(target[known, 4])
    tail = values[-max(1, int(np.ceil(.01*len(values)))):] if len(values) else values
    mass = float(values.sum())
    result.update(predicted_selected_moments=predicted.tolist(), observed_selected_moments=actual.tolist(),
        predicted_observed_easy_harm_ratio=float(predicted[4]/actual[4]) if actual[4] > 0 else None,
        known_all_risk=float(actual[1]/actual[2]) if actual[2] > 0 else None,
        known_easy_risk=float(actual[4]/actual[3]) if actual[3] > 0 else None,
        positive_easy_harm_rows=int((target[known, 4] > 0).sum()),
        top_one_percent_rows_easy_harm_share=float(tail.sum()/mass) if mass > 0 else None,
        all_known_predicted_moments=prediction[known].sum(0).tolist(),
        all_known_observed_moments=target[known].sum(0).tolist())
    return result


def census(state, args):
    eligible = bind_inputs(state, args)
    x, envelope, target, _, sites, recordings, frames, pr = args
    prediction, _, support = train.predict(state, x, envelope)
    again, _, support_again = train.predict(state, x, envelope)
    np.testing.assert_array_equal(prediction, again); np.testing.assert_array_equal(support, support_again)
    signed = train.core.signed(prediction)
    action = eligible & (signed[:,0] > 0) & (signed[:,1:] <= 0).all(1)
    known = np.isfinite(target).all(1)
    groups, _, keys = train.core.sampling.query_groups(sites, recordings, frames, known)
    ix = np.concatenate(groups); seg = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    soft = train.decision.soft_action(train.torch.from_numpy(prediction[ix]/pr['scale']),
        train.torch.from_numpy(eligible[ix]), train.torch.from_numpy(pr['rms']), .1).numpy()
    aggregates = {name:compare_aggregation(target[ix]/pr['scale'], a, seg, len(keys))
                  for name,a in [('soft',soft), ('hard',action[ix])]}
    recordings_result = {}
    for rec in np.unique(recordings):
        at = recordings == rec
        recordings_result[str(rec)] = cohort(prediction[at], target[at], action[at], envelope[at])
    return dict(full=cohort(prediction,target,action,envelope), recordings=recordings_result,
        aggregations=aggregates, input_hashes=state['input_hashes'],
        prediction_hash=train.parent.forest.fingerprint(prediction),
        action_hash=train.parent.forest.fingerprint(action), inference_replay_exact=True,
        inference_outcome_filter=False, full_TRAIN_rows=len(x), known_TRAIN_queries=len(keys),
        new_training=False, independent_roles_read=False)


def summarize(rows):
    summary = {}
    for mode in ('quadratic','task_utility','unit_control','budget_scaled'):
        subset = [r for r in rows if r['mode'] == mode]
        full = [r['census']['full'] for r in subset]
        dev = [r['cached_development_policy'] for r in subset]
        ratios = [r['predicted_observed_easy_harm_ratio'] for r in full]
        ratios = [v for v in ratios if v is not None]
        summary[mode] = dict(views=len(subset),
            TRAIN_known_easy_violations=sum(r['known_easy_risk'] is not None and r['known_easy_risk'] > .02+1e-12 for r in full),
            TRAIN_known_all_violations=sum(r['known_all_risk'] is not None and r['known_all_risk'] > .02+1e-12 for r in full),
            TRAIN_upper_easy_violations=sum(r['easy_selected_risk_upper'] is not None and r['easy_selected_risk_upper'] > .02+1e-12 for r in full),
            TRAIN_supported_views=sum(r['finite_completion_supported'] for r in full),
            TRAIN_undefined_easy_views=sum(r['known_easy_risk'] is None for r in full),
            TRAIN_underpredicted_views=sum(v < 1 for v in ratios),
            TRAIN_median_predicted_actual_easy_harm=float(np.median(ratios)) if ratios else None,
            development_known_easy_violations=sum(r['known_easy_positive_risk'] is not None and r['known_easy_positive_risk'] > .02+1e-12 for r in dev),
            TRAIN_known_easy_pass_development_fail=sum(
                t['known_easy_risk'] is not None and t['known_easy_risk'] <= .02+1e-12
                and d['known_easy_positive_risk'] is not None and d['known_easy_positive_risk'] > .02+1e-12
                for t,d in zip(full,dev)),
            TRAIN_positive_soft_query_easy_excess=sum(r['census']['aggregations']['soft']['constraints']['easy']['query_mean_excess'] > 0 for r in subset),
            TRAIN_positive_hard_query_easy_excess=sum(r['census']['aggregations']['hard']['constraints']['easy']['query_mean_excess'] > 0 for r in subset))
    return summary
