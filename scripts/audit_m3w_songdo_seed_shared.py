"""Tighten the original Songdo completion envelope without numeric label access."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_recipient_attribution as prior
from src.evaluation.m3w_completion_contrasts import estimate
from src.evaluation.m3w_seed_shared_geometry import shared_geometry

np, primary, storage = prior.np, prior.primary, prior.storage
OUT = prior.OUT.parent/'songdo_shared_seed_completion_v1'
SEEDS = primary.protocol.SEEDS


def build_paths(z, cv, union):
    result = {}
    for arm in (*prior.CONTROLS, 'joint'):
        paths = []
        for seed in SEEDS:
            tag = f'seed{seed}_'
            ix, path = z[tag+arm+'_rows'], z[tag+arm+'_paths']
            lo, hi = z[tag+'bounds']
            if len(set(ix)) != 2 or not np.all((lo <= ix) & (ix < hi)):
                raise ValueError('Unexpected frozen proposals')
            dest = cv[union].copy()
            dest[np.searchsorted(union, ix)] = path
            paths.append(dest)
        result[arm] = np.stack(paths)
    return result


def score_flight(data, name, archived):
    meta, cache, _, bank = data.observed(name)
    stem = Path(name).stem
    packet = primary.checked(primary.PACKETS/f'{stem}.npz', data.proposals[name]['packet_sha256'])
    mask = primary.checked(cache/f'{stem}_label_index.npz', meta['label_index_sha'])
    with (np.load(bank, allow_pickle=False) as b, np.load(mask, allow_pickle=False) as m,
          np.load(packet, allow_pickle=False) as z):
        union = np.unique(np.concatenate([z[f'seed{s}_{a}_rows'] for s in SEEDS
                                         for a in (*prior.CONTROLS, 'joint')]))
        cv, off = b['cv'], b['offsets']
        valid = m['future_mask']
        if valid.shape != cv.shape[:2] or off[-1] != len(cv):
            raise ValueError('Original label mask or weights no longer align')
        missing = ~valid[union]
        weights = np.repeat(1./np.diff(off), np.diff(off))[union]
        predictions = build_paths(z, cv, union)
        geometric = {a: shared_geometry(predictions[a], predictions['joint'], missing, weights)
                     for a in prior.CONTROLS}

    saved = {a: [archived['scores'][str(s)]['matched']['matched_contrasts'][a] for s in SEEDS]
             for a in prior.CONTROLS}
    diag = np.concatenate([geometric[a]['per_seed_diagonal'] for a in prior.CONTROLS])
    old = np.concatenate([[r['known_pp']-r['lower_pp'] for r in saved[a]] for a in prior.CONTROLS])
    j = int(diag.argmax())
    scale = old[j]/diag[j] if diag[j] > 0. else 0.
    if scale < 0.:
        raise ValueError('Original normalization must be positive')
    np.testing.assert_allclose(scale*diag, old, atol=1e-12, rtol=0.)
    result = {}
    for a, g in geometric.items():
        observed = np.mean([r['known_pp'] for r in saved[a]])
        lower = np.mean([r['lower_pp'] for r in saved[a]])
        radius = min(observed-lower, scale*g['shared'])
        result[a] = dict(observed=float(observed), original_lower=float(lower), original_upper=float(2.*observed-lower),
                         shared_lower=float(observed-radius), shared_upper=float(observed+radius),
                         witness_low=float(observed+scale*g['witness_low']),
                         witness_high=float(observed+scale*g['witness_high']),
                         affected_steps=g['affected_steps'], tighter_steps=g['tighter_steps'])
    return result


def run():
    data, groups, hashes, binding, orig = prior.check_scope()
    scores = {}
    for n, name in enumerate(sorted(groups), 1):
        receipt = prior.published.RUN/'confirmation'/f'{Path(name).stem}.json'
        scores[name] = score_flight(data, name, prior.verified_flight(receipt, hashes[name]))
        if n % 40 == 0:
            print(f'shared seed geometry: {n}/160', flush=True)
    members = [[scores[k] for k in sorted(groups) if groups[k] == g] for g in sorted(set(groups.values()))]
    draws = np.random.default_rng(1439).integers(14, size=(3000, 14)).tolist()
    out = {}
    for a in prior.CONTROLS:
        first = members[0][0][a]
        points = {k: [float(np.mean([v[a][k] for v in g])) for g in members] for k in first if not k.endswith('_steps')}
        metrics = {k: estimate(v, draws) for k, v in points.items()}
        for a0, b0 in [('observed', 'known_pp'), ('original_lower', 'lower_pp')]:
            r = orig['matched_joint_minus'][a][b0]
            np.testing.assert_allclose(points[a0], r['component_points'], atol=1e-12, rtol=0.)
            np.testing.assert_allclose([metrics[a0]['point'], *metrics[a0]['ci95']], [r['point'], *r['ci95']],
                                       atol=1e-12, rtol=0.)
        out[a] = dict(endpoints=metrics, source_means=points,
                      **{k: sum(v[a][k] for v in scores.values()) for k in first if k.endswith('_steps')})
    row = dict(result_source='fresh_shared_seed_completion_of_frozen_predictions', binding=binding,
               comparisons=out, flights=len(scores), components=len(members), seeds=SEEDS, bootstrap_draws=len(draws),
               forecast_label_mask_read=True, numeric_future_coordinates_read=False, calibration_outcomes_read=False,
               new_cohort_opened=False, original_2pct_unchanged=True, training=False, deployment_changed=False)
    storage.write_json(OUT/'readout.json', row)
    for a, v in out.items():
        print(a, v['endpoints'], v['tighter_steps'], flush=True)


if __name__ == '__main__':
    run()
