"""Freeze stricter Athens and Swiss movement controls without outcome access."""

from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import freeze_m3w_external_recipient_factorial as frozen
from src.evaluation import m3w_external_step_budget as budget

np, store = frozen.np, frozen.store
OUT = frozen.OUT.with_name('external_step_budget_v1')
LOCAL = frozen.LOCAL.with_name(OUT.name)
CONTROLS = ('independent', 'additive', 'same_family')
DATASETS = ('athens', 'swiss')


def geometry(cv, offsets, pair):
    q, rows, profiles = (pair[k] for k in ('queries', 'rows', 'profiles'))
    summary = dict(kept=len(q), collapsed=pair['collapsed'], variants={})
    if not len(q):
        return summary
    for k, path in pair['path'].items():
        limits = []
        for j, i in enumerate(q):
            lo, hi = offsets[i:i+2]
            a, b = ((rows[j, arm]-lo, None, path[j, arm]) for arm in range(2))
            limits.append(frozen.distance_envelope(cv[lo:hi], a, b))
        v = np.asarray(limits)
        summary['variants'][k] = dict(ade_envelope=float(v[:, 0].mean()), fde_envelope=float(v[:, 1].mean()),
            relative_envelope=float(v[:, 2].mean()), distinct_paths=int(v[:, 3].sum()), recipient_changes=int(v[:, 4].sum()))
    path = np.linalg.norm(pair['path']['original']-cv[rows], axis=-1)
    summary['retained_integrated_fraction'] = float(np.mean(4.*profiles.sum(1)/path.sum((1, 2, 3))))
    summary['retained_peak_fraction'] = float(np.mean(profiles.max(1)/path[:, 0, 0].max(1)))
    return summary


def rehearse(cv, offsets, pair):
    if not len(pair['queries']):
        return 0
    # Score one actual roster per seed packet with fabricated targets only.
    tiny = {k: pair[k][:1] for k in ('queries', 'rows')}
    tiny['path'] = {k: x[:1] for k, x in pair['path'].items()}
    fake = cv.copy()
    fake[..., 0] += .05+(.002*(np.arange(len(cv)) % 11))[:, None]
    valid = np.ones(cv.shape[:2], dtype=bool)
    valid[::5, -3:] = False
    valid[::19] = False
    data = budget.score_pair(cv, offsets, tiny, fake, valid)
    for v in data.values():
        if (v['observed_pp'] is None or not v['lower_pp']-1e-10 <= v['observed_pp'] <= v['upper_pp']+1e-10
                or v['relative_lower'] > v['relative_upper']+1e-10):
            raise ValueError('Secondary utility scorer failed synthetic replay')
    return len(data)


def summarize(by_member, roles):
    groups = defaultdict(list)
    for name, rows in by_member.items():
        if len(rows) != 3:
            raise ValueError('Expected three fixed seeds per original member')
        groups[roles[name]].append(rows)
    out = {}
    for group, members in sorted(groups.items()):
        metrics = {}
        for c in CONTROLS:
            source = [[r[c] for r in member] for member in members]
            kept = sum(r['kept'] for member in source for r in member)
            if not kept or any(any(not r['kept'] for r in m) and any(r['kept'] for r in m) for m in source):
                raise ValueError('A source has missing or seed-inconsistent comparison support')
            supported = [m for m in source if all(r['kept'] for r in m)]

            def avg(fn):
                return float(np.mean([np.mean([fn(r) for r in m]) for m in supported]))

            metrics[c] = dict(kept=kept, collapsed=sum(r['collapsed'] for m in source for r in m),
                members_with_support=len(supported), retained_integrated_fraction=avg(lambda r: r['retained_integrated_fraction']),
                retained_peak_fraction=avg(lambda r: r['retained_peak_fraction']),
                variants={k: {f: avg(lambda r: r['variants'][k][f]) for f in ('ade_envelope', 'fde_envelope', 'relative_envelope')}
                          for k in budget.VARIANTS})
        out[group] = dict(members=len(members), comparisons=metrics)
    return out


def run(which):
    local, prefixes, forecasts, matched, roles, binding = frozen.cohort(which)
    binding.update(driver=store.sha(Path(__file__)), comparison=store.sha(Path(budget.__file__)),
                   protocol=store.sha(OUT/'protocol.md'))
    folder = LOCAL/which
    folder.mkdir(parents=True, exist_ok=True)
    by_member, packets = defaultdict(list), {}
    checked_views, all_views, synthetic = 0, 0, 0
    for name in sorted(roles):
        stem = Path(name).stem
        past = frozen.checked(local/f'{stem}.prefix.npz', prefixes[name]['prefix_sha256'])
        bank = frozen.checked(local/f'{stem}.forecast.npz', forecasts[name]['forecast_sha256'])
        with np.load(past, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b:
            offsets, cv = p['offsets'], b['cv'].astype(float)
        for seed in (17, 29, 43):
            key = f'{stem}_{seed}'
            old = matched[key]
            path = frozen.checked(local/'matched'/f'{key}.npz', old.get('proposals_sha256', old.get('sha256')))
            ident = old['identity']
            if (ident['seed'] != seed or ident.get('name', ident.get('member')) != name
                    or ident['prefix_sha256'] != store.sha(past) or ident['forecast_sha256'] != store.sha(bank)):
                raise ValueError('Original proposal does not bind its exact causal inputs')
            local_arrays, metrics = {}, {}
            with np.load(path, allow_pickle=False) as z:
                q = np.arange(len(offsets)-1) if which == 'athens' else z['query_index']
                if (len(q) != old.get('queries', old.get('available')) or len(offsets)-1 != old.get('queries', old.get('original'))
                        or (len(q) and not np.array_equal(np.diff(offsets)[q], np.diff(z['offsets'])))):
                    raise ValueError('Frozen eligible or full roster changed')
                for c in CONTROLS:
                    pair = budget.build_pair(cv, offsets, q, z, c)
                    local_arrays.update({f'{c}_queries': pair['queries'], f'{c}_profile': pair['profiles']})
                    metrics[c] = geometry(cv, offsets, pair)
                    synthetic += rehearse(cv, offsets, pair)
            dst = folder/f'{key}.npz'
            if dst.exists():
                with np.load(dst, allow_pickle=False) as z:
                    if set(z.files) != set(local_arrays):
                        raise ValueError('Existing strict budget packet differs')
                    for a, x in local_arrays.items():
                        np.testing.assert_array_equal(x, z[a])
                digest = store.sha(dst)
            else:
                digest = store.save_npz(dst, local_arrays)
            packets[key] = dict(profile_sha256=digest, source_proposal_sha256=store.sha(path), matched=len(q), full=len(offsets)-1)
            by_member[name].append(metrics)
            checked_views += len(q)
            all_views += len(offsets)-1
        hb = dict(pid=os.getpid(), dataset=which, completed=len(packets), total=len(matched))
        pending = folder/'heartbeat.pending'
        pending.write_text(json.dumps(hb)+'\n')
        pending.replace(folder/'heartbeat.json')
        print(hb, flush=True)
    if (checked_views, all_views) != ((1929, 1929) if which == 'athens' else (7431, 7689)):
        raise ValueError('Expected the full frozen original roster')
    out = dict(result_source='fresh_matched_step_profiles_and_fabricated_metric_rehearsal', bindings=binding, packets=packets,
               source_groups=summarize(by_member, roles), matched_views=checked_views, full_views=all_views,
               fabricated_comparisons_checked=synthetic, future_coordinates_read=False, real_accuracy_scored=False,
               real_roles_assigned=False, original_2pct_unchanged=True, training=False, deployment_changed=False)
    store.write_json(OUT/f'{which}.json', out)


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dataset in DATASETS:
            run(dataset)
