"""Re-score the OpenDD slate contrast with an identical action menu in both arms."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_slate_imitation as exp
from src.world_model import m3w_slate_shared_menu as shared

study, base, np, torch = exp.study, exp.base, exp.np, exp.torch
OUT = exp.OUT.parent / 'opendd_slate_shared_menu_v1'
LOCAL = exp.LOCAL.parent / OUT.name
SITES = tuple(f'rdb{i}' for i in range(1, 8))
SEEDS = (17, 29, 43)
NAMES = ('joint', 'matched_independent')


def register():
    parent = exp.OUT / 'registration.json'
    frozen = exp.OUT / 'freeze.json'
    reg, f = study.read(parent), study.read(frozen)
    if f['registration'] != base.sha(parent) or not reg['original_2pct']:
        raise ValueError('Require registered original slate fits and forecasts')
    if set(reg['cohorts']) != set(SITES) or tuple(reg['seeds']) != SEEDS:
        raise ValueError('Unexpected source or seed allocation')
    base.immutable(OUT / 'registration.json', dict(parent=study.artifact(parent), freeze=study.artifact(frozen),
        comparison='same candidates and peak slots; ablate only the learned pair score',
        arms=NAMES, development_only=True, original_2pct=True, no_threshold_tuning=True,
        source_groups=SITES, seeds=SEEDS, bootstrap_seed=3183, bootstrap_draws=3000,
        code=[study.artifact(Path(p)) for p in (__file__, shared.__file__, exp.__file__, exp.slate.__file__,
                                                shared.costs.__file__, study.__file__)]))
    return base.sha(OUT / 'registration.json'), reg['cohorts'], f


def source_inputs(ref):
    with np.load(study.checked(ref), allow_pickle=False) as z:
        h = z['history']
        q = np.repeat(np.arange(len(z['offsets'])-1), np.diff(z['offsets']))
    cv, ols = study.method.analytic(h)
    bank = np.stack([ols, exp.previous.camera.linear_menu(h, 12)[:, 2]], axis=1)
    return h, q, cv, ols, bank


def replay(net, h, q, cv, ols, bank, original, prior_additive):
    a, b = cv.copy(), cv.copy()
    peak = np.linalg.norm(original-cv, axis=-1).max(axis=-1)
    stats = Counter()
    for j in np.unique(q[peak > 0]):
        ix = np.flatnonzero(q == j)
        p = peak[ix]
        radii = p[p > 0]
        a[ix], b[ix], candidates = shared.choose(net, h[ix], cv[ix], ols[ix], bank[ix], radii)
        stats[f'queries_{len(radii)}_edits'] += 1
        stats['candidate_assignments'] += candidates
        for paths in (a[ix], b[ix]):
            p = np.linalg.norm(paths-cv[ix], axis=-1).max(axis=-1)
            if np.count_nonzero(p) != len(radii):
                raise ValueError('Common-menu replay changed the intervention count')
            np.testing.assert_allclose(np.sort(p[p > 0]), np.sort(radii), atol=1e-9, rtol=1e-9)
        changed = not np.allclose(a[ix], b[ix], atol=1e-9, rtol=1e-9)
        if changed and len(radii) == 1:
            raise ValueError('Pair term affected a singleton')
        stats['joint_differs_additive'] += int(changed)
        stats['joint_differs_original_joint'] += int(not np.allclose(b[ix], original[ix], atol=1e-9, rtol=1e-9))
        stats['additive_differs_original_additive'] += int(not np.allclose(a[ix], prior_additive[ix], atol=1e-9, rtol=1e-9))
    return dict(cv=cv, ols4=ols, joint=b, matched_independent=a, q=q), dict(stats)


def freeze(key, cohorts, parent):
    forecasts, stats = {}, {}
    for site in SITES:
        h, q, cv, ols, bank = source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f'{site}_{seed}'
            fit = parent['fits'][name]
            identity = fit['identity']
            if (identity['omitted_site'] != site or set(identity['fit_sites']) != set(SITES)-{site}
                    or identity['seed'] != seed or identity['arm'] != 'analytic'):
                raise ValueError('Require held-source analytical checkpoint')
            checkpoint = torch.load(study.checked(fit['checkpoint']), weights_only=True, map_location='cpu')
            if checkpoint['identity'] != identity or checkpoint['step'] != exp.STEPS:
                raise ValueError('Checkpoint identity or completed training step differs')
            model = exp.slate.Slate().eval()
            model.load_state_dict(checkpoint['net'])
            with np.load(study.checked(parent['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], q)
                np.testing.assert_array_equal(z['cv'], cv)
                values, stats[name] = replay(model, h, q, cv, ols, bank, z['joint'], z['matched_independent'])
            forecasts[name] = exp.previous.save_bank(LOCAL / f'{name}.npz', values)
            heartbeat = LOCAL / 'heartbeat.pending'
            heartbeat.write_text(study.json.dumps(dict(pid=os.getpid(), completed=len(forecasts), total=21,
                                                       last=name))+'\n')
            heartbeat.replace(LOCAL / 'heartbeat.json')
            print(name, stats[name], flush=True)
    base.immutable(OUT / 'freeze.json', dict(registration=key, forecasts=forecasts, replay=stats,
        outcomes_accessed=False, training=False, new_inference=True))


def score(key, cohorts):
    fp = OUT / 'freeze.json'
    f = study.read(fp)
    if f['registration'] != key or f['outcomes_accessed'] or len(f['forecasts']) != 21:
        raise ValueError('Require all development-source predictions frozen before scoring')
    data = exp.loader(cohorts)
    groups = {}
    for site, d in data.items():
        rows = []
        cut = np.quantile(np.concatenate([data[s]['e'] for s in SITES if s != site]), [.25, .75])
        for seed in SEEDS:
            with np.load(study.checked(f['forecasts'][f'{site}_{seed}']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], d['q'])
                paths = {k: z[k] for k in ('cv', 'ols4', *NAMES)}
            for r in np.unique(d['rec']):
                at = d['rec'] == r
                row = exp.score_record({k: p[at] for k, p in paths.items()}, d['y'][at], d['v'][at],
                                       np.unique(d['q'][at], return_inverse=True)[1], cut)
                rows.append(row)
        delta = [r['direct']['matched_independent'] for r in rows if r['direct']['matched_independent'] is not None]
        geometry = [r['pair_relative_error_px']['matched_independent']-r['pair_relative_error_px']['joint']
                    for r in rows if r['direct']['matched_independent'] is not None
                    and all(r['pair_relative_error_px'][k] is not None for k in NAMES)]
        arms = {k: dict(edits=sum(r['arms'][k]['risk']['selected'] for r in rows),
                        risk_pass=sum(r['arms'][k]['risk']['supported'] for r in rows),
                        dual_pass=sum(r['arms'][k]['dual_pass'] for r in rows)) for k in NAMES}
        pass_pairs = Counter((r['arms']['joint']['dual_pass'], r['arms']['matched_independent']['dual_pass'])
                             for r in rows)
        groups[site] = dict(views=len(rows), recordings=len(rows)//len(SEEDS), estimable=len(delta),
                            known_pp=float(np.mean(np.array(delta)[:, 0])) if delta else None,
                            lower_pp=float(np.mean(np.array(delta)[:, 1])) if delta else None,
                            observed_pair_error_reduction_native=float(np.mean(geometry)) if geometry else None,
                            geometry_estimable=len(geometry), arms=arms,
                            paired_dual_pass=dict(joint_only=pass_pairs[True, False],
                                                  additive_only=pass_pairs[False, True], both=pass_pairs[True, True]))
    def interval(key):
        x = np.asarray([g[key] for g in groups.values() if g[key] is not None])
        draws = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(axis=1) if len(x) > 1 else None
        return dict(sites=len(x), mean=float(x.mean()) if len(x) else None,
                    ci95=np.quantile(draws, [.025, .975]).tolist() if draws is not None else None)
    result = dict(result_source='fresh_run', checkpoints='cached_verified', freeze=study.artifact(fp),
        groups=groups, inference_totals=dict(sum(map(Counter, f['replay'].values()), Counter())),
        contrasts={k: interval(k) for k in ('known_pp', 'lower_pp', 'observed_pair_error_reduction_native')},
        independent_confirmation=False, reserved_outcomes_opened=False, threshold_tuning=False,
        training=False, stage5c=False, smc=False)
    base.immutable(OUT / 'readout.json', result)
    print(result, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('freeze', 'score'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, parent)
        else:
            score(key, cohorts)


if __name__ == '__main__':
    main()
