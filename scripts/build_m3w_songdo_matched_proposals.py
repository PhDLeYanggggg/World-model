"""Prepare matched Songdo policy proposals without reading future labels."""

import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_first_day_vehicle_bank as bank
from scripts import run_m3w_vehicle_pair_learning as learn

first, pair, base, old = bank.first, learn.pair, learn.base, learn.old
np, torch, pack = bank.np, bank.torch, bank.pack
OUT = first.PUBLIC.parent / 'songdo_matched_proposals_v1'
LOCAL = first.PRIVATE.parent / OUT.name
ARMS = ('independent', 'additive', 'same_family', 'joint')
CHUNK = 4096


def setup():
    p = first.PUBLIC / 'input_receipt.json'
    q = first.PUBLIC / 'vehicle_bank.json'
    receipt, forecasts = first.read(p), first.read(q)
    lineage = first.PUBLIC / 'lineage.json'
    if (receipt['split_assigned'] or receipt['forecasts_scored'] or receipt['future_coordinates_parsed']
            or forecasts['split_assigned'] or forecasts['forecasts_scored'] or forecasts['future_index_loaded']
            or forecasts['future_coordinates_loaded'] or forecasts['binding']['prefix'] != pack.sha(p)
            or forecasts['binding']['lineage'] != pack.sha(lineage)
            or set(receipt['sessions']) != set(forecasts['sessions']) or len(receipt['sessions']) != 200):
        raise ValueError('Require the frozen split-neutral first-day bank')
    key, *_ = learn.register()
    trained = learn.OUT / 'freeze.json'
    f = first.read(trained)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require the fixed OpenDD pair heads')
    ensembles = {seed: {a: [] for a in ('additive', 'joint')} for seed in base.SEEDS}
    weights = {}
    for name, c in sorted(f['cells'].items()):
        for arm in ('additive', 'joint'):
            ref = c['weights'][arm]
            ident = ref['identity']
            if (ident['registration'] != key or ident['seed'] != c['seed'] or ident['role'] != arm
                    or ref['step'] != learn.STEPS or ident['fit_sites'] != c['fit_sites']
                    or len(ident['excluded']) != 2 or ident['name'] != f'{name}_{arm}'):
                raise ValueError('Invalid head provenance')
            net = base.load_net(learn.LOCAL / f'{name}_{arm}.pt', ref,
                                learn.model.PairValue(arm == 'joint'), ident['excluded'])
            ensembles[c['seed']][arm].append(net)
            weights[ident['name']] = ref['sha256']
    if any(len(n) != 21 for e in ensembles.values() for n in e.values()):
        raise ValueError('Missing seed-specific head ensemble')
    binding = dict(prefix_receipt=pack.sha(p), candidate_bank=pack.sha(q), lineage=pack.sha(lineage),
        implementation=pack.sha(Path(__file__)), donor_freeze=pack.sha(trained), donor_weights=weights,
        protocol=pack.sha(OUT / 'protocol.md'), selection='one past-defined query per flight and seed')
    return receipt, forecasts, ensembles, binding


def ranked_pairs(row, family, i, j, additive, joint):
    if not len(i) or not (family[i] == family[j]).any() or (row[i] == row[j]).any():
        raise ValueError('A legal matched control requires two distinct actors')
    a = learn.model.merit(additive)
    summed = a[i]+a[j]
    v = learn.model.merit(.5*(additive[i]+additive[j]))
    w = learn.model.merit(joint)
    result = {}
    for arm, score in zip(ARMS, (summed, v, np.where(family[i]==family[j], w, -np.inf), w), strict=True):
        k = int(score.argmax())
        if not np.isfinite(score[k]):
            raise ValueError('Nonfinite selected candidate score')
        result[arm] = (np.array([i[k], j[k]]), float(score[k]))
    return result


@torch.inference_mode()
def logits(ensembles, x, i, j, relative):
    a = sum((n.head(n.actor(x)).numpy().astype(np.float64) for n in ensembles['additive']),
            start=np.zeros((len(x), 4)))/len(ensembles['additive'])
    embed = [n.actor(x) for n in ensembles['joint']]
    result = np.empty((len(i), 4), dtype=np.float64)
    for lo in range(0, len(i), CHUNK):
        ix, jx, r = i[lo:lo+CHUNK], j[lo:lo+CHUNK], relative[lo:lo+CHUNK]
        block = np.zeros((len(ix), 4))
        for n, z in zip(ensembles['joint'], embed, strict=True):
            u, v = z[ix], z[jx]
            block += n.head(torch.cat((.5*(u+v), (u-v).abs(), r), dim=1)).numpy()
        result[lo:lo+len(ix)] = block/len(embed)
    return a, result


def propose(history, offsets, bank_data, seed, ensembles):
    cv, ols = bank_data['cv'], bank_data['ols4']
    raw = np.stack([bank_data[k] for k in ('ols4', 'median4', f'self_{seed}')], axis=1)
    paths = old.matched.same_radius(cv, raw, offsets)
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    slots = pair.fixed_pairs(offsets, np.repeat('flight', len(history)), peak)
    if not slots:
        return None
    lo, hi = slots['flight']
    q = slice(lo, hi)
    h, f, g, path = history[q], cv[q], ols[q], paths[q]
    x, units, _, _ = old.bridge.features(h, f, g, path)
    m = dict(cv=f, paths=path, x=x, units=units, peak=peak[q],
             speed=np.diff(h[:, -4:], axis=1).mean(axis=1))
    row, family, i, j = learn.choices(m['peak'], 0, len(h))
    rel = learn.relative(m, row[i], row[j], family[i], family[j])
    a, z = logits(ensembles, torch.from_numpy(x[row, family]), i, j, torch.from_numpy(rel))
    decisions = ranked_pairs(row, family, i, j, a, z)
    result = dict(bounds=np.array([lo, hi]), query_index=int(np.searchsorted(offsets, lo)),
                  legal_pairs=len(i), radius=float(m['peak'][row[0], family[0]]))
    for arm, (v, score) in decisions.items():
        r, k = row[v], family[v]
        np.testing.assert_allclose(m['peak'][r, k], result['radius'], atol=1e-9, rtol=1e-9)
        result[f'{arm}_rows'] = r+lo
        result[f'{arm}_families'] = k
        result[f'{arm}_paths'] = path[r, k]
        result[f'{arm}_rank_score'] = score
    return result


def run():
    receipt, banks, heads, binding = setup()
    fingerprint = pack.hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    records = {}
    for name, info in receipt['sessions'].items():
        stem = Path(name).stem
        cache = pack.CACHE if info['cache_role'] == 'previous' else first.PRIVATE
        prefix, paths = cache / f'{stem}_prefix.npz', bank.CACHE / f'{stem}.npz'
        if pack.sha(prefix) != info['prefix_sha'] or pack.sha(paths) != banks['sessions'][name]['sha256']:
            raise ValueError(f'Frozen inputs changed: {name}')
        p, r = LOCAL / f'{stem}.npz', LOCAL / f'{stem}.json'
        if not r.exists():
            result, summary = {}, {}
            with np.load(prefix, allow_pickle=False) as x, np.load(paths, allow_pickle=False) as f:
                h, offsets = x['history'], x['offsets']
                np.testing.assert_array_equal(f['offsets'], offsets)
                for seed in base.SEEDS:
                    v = propose(h, offsets, f, seed, heads[seed])
                    summary[str(seed)] = None
                    if v is not None:
                        result.update({f'seed{seed}_{k}': a for k, a in v.items()})
                        j = v['joint_rows']*3+v['joint_families']
                        summary[str(seed)] = dict(query=int(v['query_index']), agents=int(np.diff(v['bounds'])[0]),
                            pairs=v['legal_pairs'], radius=v['radius'], joint_differs={
                                a: not np.array_equal(j, v[f'{a}_rows']*3+v[f'{a}_families']) for a in ARMS[:-1]})
            pack.write_json(r, dict(binding=fingerprint, packet_sha256=pack.save_npz(p, result), seeds=summary))
        row = first.read(r)
        if row['binding'] != fingerprint or row['packet_sha256'] != pack.sha(p):
            raise ValueError('Frozen proposal changed')
        records[name] = dict(site=info['site'], **row)
        first.heartbeat('matched_proposals', completed=len(records), session=name)
    summary = {}
    for seed in base.SEEDS:
        eligible = [r['seeds'][str(seed)] for r in records.values() if r['seeds'][str(seed)] is not None]
        summary[str(seed)] = dict(eligible_flights=len(eligible), exact_unique_pairs=sum(x['pairs'] for x in eligible),
            max_pairs_in_flight=max(x['pairs'] for x in eligible),
            joint_differs={a: sum(x['joint_differs'][a] for x in eligible) for a in ARMS[:-1]})
    pack.write_json(OUT / 'receipt.json', dict(binding=binding, result_source='fresh_run', arms=ARMS,
        sessions=records, structural_summary=summary, split_assigned=False, future_index_loaded=False,
        future_coordinates_loaded=False, forecasts_scored=False, policy_selected=False,
        calibration_thresholds_fitted=False, independent_confirmation=False, original_2pct_unchanged=True))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
