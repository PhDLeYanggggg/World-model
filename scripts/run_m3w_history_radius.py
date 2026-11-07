"""Test past-conditioned clipping without changing the forecast-risk rule."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_joint_path_portfolio as frozen
from scripts import audit_m3w_recipient_budget_capacity as prior
from scripts import audit_m3w_easy_support_certificate as certificate
from scripts.refine_m3w_recipient_budget_capacity import prune
from src.world_model import m3w_history_radius as model

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC.parent/'native_history_radius_v1'
PRIVATE = prior.PRIVATE.parent/'native_history_radius_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_history_radius.py')
ARMS = ('original', 'scale_cap', 'history_cap', 'material_history_cap')
BASE, OLS, SEEDS = frozen.BASE, frozen.OLS, frozen.SEEDS


def register():
    cut, controller = prior.register()
    for p, k in ((frozen.PUBLIC, 'freeze'), (frozen.PUBLIC, 'replay'), (certificate.PUBLIC, '')):
        name = f'{k}_execution.json' if k else 'execution.json'
        run.checked(run.read(p/name)['artifact'])
    parents = (frozen.REG, frozen.PUBLIC/'analysis.json', certificate.REG, certificate.PUBLIC/'analysis.json',
               prior.REG, prior.PUBLIC/'refinement.json', Path(prune.__code__.co_filename), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut, seeds=SEEDS,
        source_roles=[13, 1], radius_fraction=model.FRACTION, quantile=model.QUANTILE,
        buckets=model.BUCKETS, material_displacement_px=model.MATERIAL_RADIUS, solver_time=6.,
        primary='material_history_cap_vs_original_fixed_switches', original_risk=True, external_access=False)))
    return cut, controller


def fit(d, cv, y, valid, cut):
    ref, _ = prior.model.errors(cv, y, valid)
    easy = valid.any(1) & (ref > 0.) & (ref <= cut['easy'])
    h, s, crowd = (d[k] for k in ('history', 'sites', 'crowd'))
    scale, bucket = model.describe(h)
    limits = np.empty((2, len(cv)))
    for held in sorted(set(s)):
        q, conditional = model.fit(ref, easy, scale, bucket, s, crowd, held)
        ix = s == held
        limits[0, ix] = q*scale[ix]
        limits[1, ix] = conditional[bucket[ix]]*scale[ix]
    return limits


def capacities(cv, ols, bank, y, valid, offsets, cap, radius, r, cut):
    small = model.clip(cv, bank, radius)
    is_new = ((small != bank).any(axis=(2, 3)) & (small != cv[:, None]).any(axis=(2, 3)))
    material = np.linalg.norm(small-cv[:, None], axis=-1).max(2) >= model.MATERIAL_RADIUS
    combined = np.concatenate((bank, small), axis=1)
    result = {}
    for name, added in (('literal', is_new), ('material', is_new & material)):
        mask = np.concatenate((np.ones(bank.shape[:2], bool), added), axis=1)
        local = np.where(mask[..., None, None], combined, cv[:, None])
        support = certificate.assess(cv, local, y, valid, np.repeat(cap > 0, np.diff(offsets)), cut['easy'])
        optimum = prior.model.solve(cv, ols, local, y, valid, offsets, cap, prune(cv, ols, local, valid),
                                    r, cut['easy'], cut['hard'], seconds=6.)
        if optimum['proved_infeasible'] and support['feasible'] or (
                optimum['feasible'] and not support['feasible']):
            raise ValueError('The exact risk-feasibility criterion disagrees with the witness')
        result[name] = dict(support=support, solve=optimum, added=int(added.sum()))
    return result


def prepare(d, refs, seed, controller, labels, cut):
    cv, _, bank = frozen.prev.prior.bank_for(d, refs, seed)
    with np.load(run.checked(controller['artifact']), allow_pickle=False) as f:
        np.testing.assert_array_equal(f['ids'], d['global_ids'])
        choice = f['choice'][frozen.ARMS.index('screen_first_joint')]
    y, valid = labels
    limits = fit(d, cv, y, valid, cut)
    original = run.model.selected_paths(choice, bank, cv)
    radius = model.FRACTION*limits
    small = np.stack([model.clip(cv, original[:, None], p)[:, 0] for p in radius])
    material = np.where((radius[1] >= model.MATERIAL_RADIUS)[:, None, None], small[1], original)
    out = np.stack((original, *small, material))
    for path in out:
        np.testing.assert_array_equal((path != cv).any(axis=(1, 2)), choice >= 0)
    return out, limits, choice


def score(d, refs, lab, records, cut, controller, freeze, beat):
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    keys = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    keys.update({f'{a}_vs_original': ('original', a, BASE) for a in ARMS[1:]})
    keys['history_vs_scale_cap'] = ('scale_cap', 'history_cap', BASE)
    rows, roster, cases, covered, changes = [], {}, [], np.zeros(3, int), np.zeros(3, int)
    for c, f in zip(controller['cells'], freeze['cells'], strict=True):
        seed = c['seed']
        if seed != f['seed']:
            raise ValueError('Inconsistent seed order')
        cv, bank = run.candidates(d, refs, seed)
        ols = bank[:, run.NAMES.index(OLS)]
        with np.load(run.checked(f['artifact']), allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'], d['global_ids'])
            path, limits = a['paths'], a['limits']
        ref, _ = prior.model.errors(cv, y, valid)
        easy = valid.any(1) & (ref > 0.) & (ref <= cut['easy']) & (path[0] != cv).any(axis=(1, 2))
        covered += [int(easy.sum()), int((easy & (ref >= limits[0])).sum()), int((easy & (ref >= limits[1])).sum())]
        changes += (path[1:] != path[0]).any(axis=(2, 3)).sum(1)
        with np.load(run.checked(c['cache']), allow_pickle=False) as z:
            coeff = dict(zip(z['sources'], z['excluded_source_coefficients']))
        for r, (ix, groups) in records.items():
            site = str(d['sites'][ix[0]])
            roster[r] = site
            out = run.parent.readout.score_recording({BASE: cv[ix], OLS: ols[ix],
                **{k: p[ix] for k, p in zip(ARMS, path)}}, keys, y[ix], valid[ix], d['queries'][ix], r,
                easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(out, seed=seed, source=site))
            mark = PRIVATE/f'capacity_{seed}_{r}.json'
            if not mark.exists():
                at = ix[np.concatenate(groups)]
                off = np.r_[0, np.cumsum(list(map(len, groups)))]
                paths, cap, _ = prior.query_design(d, at, off, cv[at], bank[at], coeff[site])
                result = capacities(cv[at], ols[at], paths, y[at], valid[at], off, cap,
                                    model.FRACTION*limits[1, at], r, cut)
                old = prior.PRIVATE/'refinement'/f'{seed}_{r}.json'
                prev = run.read(old)['result'] if old.exists() else run.read(
                    prior.PRIVATE/f'{seed}_{r}.json')['outcomes']['all_recipients']
                run.once(mark, run.stable(dict(registration=run.sha(REG), fit=f['artifact']['sha256'],
                    prior_feasible=prev['feasible'], prior_positive=prev['positive'], result=result)))
            packet = run.read(mark)
            if packet['registration'] != run.sha(REG) or packet['fit'] != f['artifact']['sha256']:
                raise ValueError('Incompatible capacity result')
            cases.append(packet)
        beat(seed=seed, state='score_complete')
    res = frozen.prev.prior.grouped.grouped_readout(rows, roster, SEEDS)
    prev = run.read(frozen.PUBLIC/'analysis.json')['grouped']['risk']['screen_first_joint']
    if res['risk']['original'] != prev or any(res['risk'][k]['switches'] != prev['switches'] for k in ARMS):
        raise ValueError('Original screened count or risk changed')
    cap = {}
    for k in ('literal', 'material'):
        c = [r['result'][k] for r in cases]
        cap[k] = dict(feasible=sum(v['support']['feasible'] for v in c),
            positive_witnesses=sum(v['solve']['positive'] or p['prior_positive'] for p, v in zip(cases, c)),
            prior_infeasible_recovered=sum(not p['prior_feasible'] and v['support']['feasible'] for p, v in zip(cases, c)),
            unresolved_objective=sum(not (v['solve']['optimum_proved'] or v['solve']['proved_infeasible']) for v in c),
            added_paths=sum(v['added'] for v in c))
    return dict(grouped=res, capacity=cap, changed_paths=changes.tolist(), selected_easy_coverage_counts=covered.tolist(),
        external_data_opened=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, controller = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit radius protocol before running')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        cache = PRIVATE/'freeze.json'
        if phase == 'freeze':
            labels = (lab['target_eval']-d['origin'][:, None], lab['valid'])
            rows = []
            for c in run.read(frozen.PRIVATE/'freeze.json')['cells']:
                seed = c['seed']
                dest = PRIVATE/f'seed_{seed}.npz'
                mark = dest.with_suffix('.json')
                if not mark.exists():
                    path, limits, _ = prepare(d, refs, seed, c, labels, cut)
                    tmp = dest.with_suffix('.pending.npz')
                    np.savez(tmp, ids=d['global_ids'], paths=path, limits=limits)
                    tmp.replace(dest)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(dest)))
                row = run.read(mark)
                if row['registration'] != run.sha(REG) or row['seed'] != seed:
                    raise ValueError('Stale radius checkpoint')
                run.checked(row['artifact'])
                rows.append(row)
                beat(seed=seed, state='forecast_frozen')
            run.once(cache, run.stable(dict(cells=rows)))
            out = cache
        else:
            run.checked(run.read(PUBLIC/'freeze_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            rec = cal.prior.h.previous.prior.blocks(d, lab['frames'])
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, rec, cut, controller, run.read(cache), beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), pid=os.getpid(), seconds=time.monotonic()-start))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Radius result replay mismatch')


if __name__ == '__main__':
    main()
