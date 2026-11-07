"""Source-cross-fit allocation on the frozen direct-kinematic forecast bank."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_kinematic_forecast as kin
from src.world_model import m3w_record_joint_actions as model

np, read, sha, artifact, checked, once, stable = kin.np, kin.read, kin.sha, kin.artifact, kin.checked, kin.once, kin.stable
parent, old = kin.parent, kin.bank.old
PUBLIC = old.HOME/'moving_record_joint_actions_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_record_joint_actions_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_record_joint_actions.py')
NAMES = ('eqmotion', 'pct', *(k for k in parent.BASELINES if k != parent.CV), 'self')
PAIRS = {f'{k}_vs_cv': (parent.CV, k, parent.CV) for k in model.ARMS[1:]}
PAIRS.update(joint_vs_independent=('independent', 'joint', parent.CV),
             joint_vs_scene=('scene', 'joint', parent.CV),
             joint_vs_pair_additive=('pair_additive', 'joint', parent.CV))


def register():
    result, replay, forecast, train = (kin.PUBLIC/p for p in ('analysis.json', 'replay_execution.json',
                                      'forecast_freeze.json', 'training_freeze.json'))
    a, f = read(result), read(forecast)
    if (read(replay)['analysis']['sha256'] != sha(result) or a['bank_support']['self']['feasible'] != 102
            or not f['all_before_A_labels'] or not read(train)['before_A_inference_and_labels']):
        raise ValueError('Frozen source-excluded candidate bank and exact replay required')
    parents = (kin.REG, result, replay, forecast, train, parent.REG, old.prior.PARENT/'population.json',
               Path(parent.readout.__file__), *old.PARENTS)
    once(REG, stable(dict(files=[artifact(p) for p in FILES], parents=[artifact(p) for p in parents],
        candidate_names=NAMES, seeds=kin.prior.SEEDS, arms=model.ARMS, contrasts=PAIRS, pair_weight=model.PAIR_WEIGHT,
        ridge=.002, bootstrap_draws=3000, source_crossfit=True, same_counts=True, budget=.02,
        development_only=True, original_rules_unchanged=True, external_outcomes_opened=False)))
    return f['forecasts'], read(parent.REG)['forecasts']


def population():
    d = old.prior.arrays(('global_ids', 'history', 'origin', 'queries', 'sites', 'recordings'))
    d['history'] = d['history'].astype(float)
    np.testing.assert_array_equal(d['history'][:, -1], d['origin'])
    if tuple(len(np.unique(d[k])) for k in ('global_ids', 'sites', 'recordings', 'queries')) != (125540, 14, 34, 1613):
        raise ValueError('Complete fixed roster required')
    d['groups'] = [g[np.argsort(d['global_ids'][g], kind='stable')] for g in old.prior.groups(d['queries'])]
    d['crowd'] = np.empty(len(d['sites']), dtype=int)
    for g in d['groups']:
        if len(np.unique(d['recordings'][g])) != 1 or len(np.unique(d['sites'][g])) != 1:
            raise ValueError('Queries may not cross source or recording')
        d['crowd'][g] = len(g)
    return d


def candidates(data, refs, seed):
    h = data['history']; origin = data['origin'][:, None]
    ids, sites = data['global_ids'], data['sites']
    f, old_f = refs
    old_paths = parent.read_banks(old_f, seed, ids, sites)
    source = next(v for v in f if v['seed'] == seed and v['arm'] == 'self')
    fit = next(v for v in read(kin.PUBLIC/'training_freeze.json')['fits'] if v['seed'] == seed and v['arm'] == 'self')
    job = next(v for v in read(kin.REG)['jobs'] if v['seed'] == seed)
    if (fit['checkpoint'] != source['checkpoint'] or fit['identity']['registration_sha256'] != sha(kin.REG)
            or fit['identity']['global_fit_rows_sha256'] != job['identity']['train_ids_sha256']
            or set(sites) & set(job['identity']['fit_groups'])):
        raise ValueError('Kinematic producer must exclude all A sources')
    with np.load(checked(source['forecast']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids); old_paths['self'] = z['prediction'].astype(float)
    floor = np.arange(1, 13)[None, :, None] * (h[:, -1] - h[:, -2])[:, None]
    old_paths.update({k: parent.baseline_numpy(h, i)-origin for i, k in enumerate(parent.BASELINES) if k != parent.CV})
    return floor, np.stack([old_paths[k] for k in NAMES], axis=1)


def cuts_by_source():
    cut = read(old.PARENTS[4])['cuts']
    return {s: cut[str(r['job']['producer_fold'])]
            for r in read(old.PARENTS[3])['forecasts'] for s in r['job']['donor_sites']}


def fit(data, refs, beat):
    labels = old.prior.arrays(('target_eval', 'valid'))
    ids, sites, rec, h, crowd = (data[k] for k in ('global_ids', 'sites', 'recordings', 'history', 'crowd'))
    cuts = cuts_by_source(); names = np.unique(sites); frozen = []
    for seed in kin.prior.SEEDS:
        mark = PRIVATE/f'{seed}.json'; cache = PRIVATE/f'{seed}.npz'
        if not mark.exists():
            floor, paths = candidates(data, refs, seed)
            moments = {}
            for s in names:
                ix = np.flatnonzero(sites == s)
                x, scale = model.descriptors(h[ix], paths[ix], floor[ix], crowd[ix])
                moments[s] = model.fitting_moments(x, paths[ix], floor[ix],
                    labels['target_eval'][ix]-data['origin'][ix, None], labels['valid'][ix],
                    scale, 1./crowd[ix], cuts[s]['easy'])
            coef = np.stack([model.coefficients([moments[r] for r in names if r != s]) for s in names])
            del moments
            # Decisions consume only the fit coefficients and past-derived tensors.
            choices = np.full((len(model.ARMS), len(ids)), -1, dtype=np.int16)
            utility = np.empty(paths.shape[:2]); a = np.empty_like(utility); e = np.empty_like(utility)
            scale = np.empty(len(ids)); legal = np.empty_like(utility, dtype=bool)
            for s, c in zip(names, coef):
                ix = np.flatnonzero(sites == s)
                x, scale[ix] = model.descriptors(h[ix], paths[ix], floor[ix], crowd[ix])
                utility[ix], a[ix], e[ix], legal[ix] = model.decision_costs(x, c, paths[ix], floor[ix], scale[ix])
            for g in data['groups']:
                choices[1:, g] = model.choose_query(h[g], floor[g], paths[g], paths[g, NAMES.index(parent.OLS)],
                    scale[g], utility[g], legal[g])
            for r in np.unique(rec):
                ix = np.flatnonzero(rec == r)
                choices[-1, ix] = model.budget_prefix(choices[-1, ix], utility[ix], a[ix], e[ix], scale[ix])
            # The held-out source is not part of any corresponding fit moment sum.
            selected = choices >= 0
            if not legal[np.broadcast_to(np.arange(len(ids)), choices.shape)[selected], choices[selected]].all():
                raise ValueError('Exact floor paths cannot count as interventions')
            tmp = cache.with_suffix('.pending.npz')
            np.savez_compressed(tmp, ids=ids, names=names, coefficients=coef, choices=choices); tmp.replace(cache)
            once(mark, dict(seed=seed, registration=sha(REG), cache=artifact(cache),
                fits=[dict(held=str(s), fitted_sources=[str(x) for x in names if x != s]) for s in names]))
        v = read(mark)
        if v['registration'] != sha(REG) or v['seed'] != seed:
            raise ValueError('Changed registration or seed in resumed cell')
        checked(v['cache']); frozen.append(v); beat(state='decisions_frozen', seed=seed, cells=len(v['fits']))
    once(PUBLIC/'freeze.json', dict(seeds=frozen, frozen_before_readout=True, held_source_moments_excluded=True))


def score(data, refs, beat):
    freeze = read(PUBLIC/'freeze.json')
    if not freeze['frozen_before_readout'] or len(freeze['seeds']) != 3:
        raise ValueError('Complete choice freeze required')
    labels = old.prior.arrays(('target_eval', 'valid'))
    y, valid = labels['target_eval']-data['origin'][:, None], labels['valid']
    sites, rec, q = (data[k] for k in ('sites', 'recordings', 'queries'))
    cuts = cuts_by_source(); records, pair_views = [], []
    same_count = changed = compared_queries = 0
    for f in freeze['seeds']:
        seed = f['seed']; floor, paths = candidates(data, refs, seed)
        with np.load(checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], data['global_ids']); choice = z['choices']
        if not ((choice[1:5] >= 0).sum(1) == (choice[1] >= 0).sum()).all():
            raise ValueError('Unmatched count')
        predictions = {arm: model.selected_paths(c, paths, floor) for arm, c in zip(model.ARMS, choice)}
        predictions[parent.CV] = predictions.pop('floor')
        agg = {s: np.zeros((len(model.ARMS), 3)) for s in np.unique(sites)}
        for g in data['groups']:
            n = (choice[1:5, g] >= 0).sum(1)
            if not np.all(n == n[0]):
                raise ValueError('Unmatched query count')
            same_count += int(n[0]); changed += int(np.any(choice[3, g] != choice[4, g])); compared_queries += 1
            for k, p in enumerate(predictions.values()):
                err, count = model.pair_error(data['history'][g], p[g], y[g], valid[g])
                agg[sites[g[0]]][k] += (0. if err is None else err, int(err is not None), count)
        for s, values in agg.items():
            pair_views.append(dict(source=str(s), seed=seed, by_arm={key: dict(
                mean_observed_pair_displacement_error=float(v[0]/v[1]) if v[1] else None,
                observed_queries=int(v[1]), observed_pair_steps=int(v[2])) for key, v in zip(predictions, values)}))
        for r in np.unique(rec):
            ix = np.flatnonzero(rec == r); s = sites[ix[0]]; c = cuts[s]
            out = parent.readout.score_recording({k: p[ix] for k, p in predictions.items()}, PAIRS,
                y[ix], valid[ix], q[ix], str(r), easy_cut=c['easy'], hard_cut=c['hard'])
            records.append(dict(seed=seed, source=str(s), **out))
        beat(state='scored_seed', seed=seed, views=len(records))
    if len(records) != 102 or len(pair_views) != 42:
        raise ValueError('Do not remove unsupported views or sources')
    names = np.unique(sites); draw = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    def ci(v):
        if any(x is None for x in v):
            return None
        v = np.asarray(v).reshape(len(names), 3).mean(1)
        return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
    grouped = [[r for r in records if r['source'] == s and r['seed'] == seed] for s in names for seed in kin.prior.SEEDS]
    contrasts = {}
    for k in PAIRS:
        v = [parent.readout.merge_comparisons([r['comparisons'][k] for r in block]) for block in grouped]
        contrasts[k] = {field: ci([m[field] for m in v]) for field in ('known_pp', 'completion_lower_pp')}
    pair = {(r['source'], r['seed']): r['by_arm'] for r in pair_views}
    def pair_change(left, right):
        k = 'mean_observed_pair_displacement_error'
        v = [(pair[s, seed][left][k], pair[s, seed][right][k]) for s in names for seed in kin.prior.SEEDS]
        return ci([a-b if a is not None and b is not None else None for a, b in v])
    risks = {}
    for k in records[0]['metrics']:
        vals = [r['metrics'][k]['risk'] for r in records]
        risks[k] = dict(passes=sum(v['supported'] for v in vals), switches=sum(v['selected'] for v in vals),
            unknown_switches=sum(v['unknown_selected'] for v in vals),
            undefined_easy=sum(v['completion_easy_risk_upper'] is None for v in vals),
            easy_over_2=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper'] > .02 for v in vals))
    once(PUBLIC/'analysis.json', stable(dict(contrasts=contrasts, risks=risks, records=records, pair_views=pair_views,
        pair_error_reduction_joint_vs_additive=pair_change('pair_additive', 'joint'),
        pair_error_reduction_joint_vs_independent=pair_change('independent', 'joint'),
        matched_switches_per_arm=same_count, joint_changes_vs_additive=changed, compared_queries=compared_queries,
        all_views_retained=True, original_rules_unchanged=True, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase; refs = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit registered allocator before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); start = time.monotonic()
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Preserve atomic decision checkpoints for resume')
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            kin.prior.source.atomic(PRIVATE/'heartbeat.json', event); print(event, flush=True)
        beat(state='start'); data = population()
        if phase == 'fit':
            fit(data, refs, beat)
        else:
            score(data, refs, beat)
        once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifact=artifact(PUBLIC/('freeze.json' if phase == 'fit' else 'analysis.json'))))


if __name__ == '__main__':
    main()
