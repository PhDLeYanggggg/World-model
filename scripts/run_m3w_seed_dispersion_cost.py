"""Fit and score the registered A-only, fixed-count disagreement ablation."""
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
import joblib
from scripts import run_m3w_pair_coupling as prior
from src.world_model import m3w_seed_dispersion_cost as model

np, read, sha, artifact, checked, once, stable = (prior.np, prior.read, prior.sha,
    prior.artifact, prior.checked, prior.once, prior.stable)
HOME = prior.HOME
PUBLIC = HOME/'moving_seed_dispersion_cost_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_seed_dispersion_cost_v1'
REG = PUBLIC/'registration.json'
SEEDS = (17, 29, 43)
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_seed_dispersion_cost.py')
PARENTS = (Path(prior.__file__), Path(prior.base.__file__), prior.PARENT/'population.json',
    prior.PARENT/'analysis.json', HOME/'moving_camera_frame_v1/registration_v2.json',
    HOME/'moving_source_forecast_supported_v1/analysis.json',
    HOME/'dronecrowd_three_seed_bank_v1/analysis.json')


def register():
    banks = sorted((b for b in read(PARENTS[-2])['prediction_receipts']
        if b['identity']['family'] == 'eqmotion' and b['identity']['fold'] == 2),
        key=lambda b: b['identity']['seed'])
    proof = {b['seed']: b for b in read(PARENTS[-1])['trials']
             if b['family'] == 'eqmotion' and b['fold'] == 2}
    if tuple(b['identity']['seed'] for b in banks) != SEEDS or any(
        proof[b['identity']['seed']]['prediction_sha256'] != b['sha256'] or
        proof[b['identity']['seed']]['checkpoint']['sha256'] != b['checkpoint_sha256'] for b in banks):
        raise ValueError('Need all three matched and verified actual training seeds')
    once(REG, stable(dict(files=[artifact(p) for p in FILES], parents=[artifact(p) for p in PARENTS],
        banks=banks, forecaster_seeds=SEEDS, fitted_sources=14, expected_rows=125540,
        arm_order=['single_seed', 'seed_dispersion'], bootstrap_draws=3000, bootstrap_seed=73107,
        original_rules_unchanged=True, independent_confirmation=False, stage5c_executed=False,
        smc_enabled=False, deployment_changed=False, submission_ready=False)))
    return banks


def population(banks):
    data = prior.arrays(('global_ids', 'history', 'origin', 'queries', 'sites', 'recordings'))
    ids, sites, q = (data[k] for k in ('global_ids', 'sites', 'queries'))
    if (len(ids), len(np.unique(sites)), len(np.unique(q)), len(np.unique(data['recordings']))) != (125540, 14, 1613, 34):
        raise ValueError('Keep the exact original A rows, sources, queries and recordings')
    forecasts = []
    for bank in banks:
        ident = bank['identity']; names = set(sites)
        if names & set(ident['fit_groups']) or not names <= set(ident['excluded_groups']):
            raise ValueError('Producer training must exclude every A source')
        with np.load(checked(bank), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            forecasts.append(z['relative_prediction'].astype(float))
    count = np.empty(len(ids), dtype=int)
    for ix in prior.groups(q):
        if len(np.unique(sites[ix])) != 1 or len(np.unique(data['recordings'][ix])) != 1:
            raise ValueError('A query cannot span sources or recordings')
        count[ix] = len(ix)
    return data, np.stack(forecasts), count


def model_paths(seed, n):
    prefix = PRIVATE/f'seed{seed}_source{n:02d}'
    return prefix.with_suffix('.joblib'), prefix.with_suffix('.json')


def fit_all(data, labels, forecasts, crowd, beat):
    sites, q, ids, history, origin = (data[k] for k in ('sites', 'queries', 'global_ids', 'history', 'origin'))
    names = np.unique(sites); weight = 1./crowd
    for s in names:
        at = sites == s; weight[at] /= weight[at].sum()
    receipts = []
    for j, seed in enumerate(SEEDS):
        x, spread, floor, scale = model.inputs(history.astype(float), forecasts, j, crowd)
        features = (x, np.column_stack((x, spread)))
        for n, held in enumerate(names):
            path, mark = model_paths(seed, n)
            fit = sites != held; hold = ~fit
            if not mark.exists():
                y = labels['target_eval'][fit] - origin[fit, None]
                costs = model.targets(floor[fit], forecasts[j, fit], y, labels['valid'][fit], scale[fit])
                heads = [model.fit(f[fit], costs, weight[fit]) for f in features]
                pred = np.stack([model.predict(h, f[hold], scale[hold]) for h, f in zip(heads, features)])
                choice = np.stack([model.matched_choices(q[hold], ids[hold], p) for p in pred])
                pending = path.with_suffix('.pending')
                joblib.dump(dict(heads=heads, prediction=pred, choices=choice, ids=ids[hold]), pending, compress=3)
                pending.replace(path)
                once(mark, dict(checkpoint=artifact(path), registration_sha256=sha(REG), seed=seed,
                    held_source=str(held), fitted_sources=names[names != held].tolist(),
                    fit_rows=int(fit.sum()), known_fit_rows=int(np.isfinite(costs[:, 0]).sum()),
                    held_rows=int(hold.sum())))
            receipt = read(mark)
            if receipt['registration_sha256'] != sha(REG) or receipt['held_source'] != held:
                raise ValueError('Changed fit identity on resume')
            checked(receipt['checkpoint']); receipts.append(receipt)
            beat(state='fit_checkpoint', seed=seed, held_source=str(held), complete=len(receipts), total=42)
    once(PUBLIC/'freeze.json', dict(fits=receipts, all_actions_frozen_before_scoring=True,
        completed_cost_fit_pairs=len(receipts), new_forecaster_training=False))


def evaluate(data, labels, forecasts, crowd, beat):
    q, sites, ids, history, origin, records = (data[k] for k in
        ('queries', 'sites', 'global_ids', 'history', 'origin', 'recordings'))
    previous = read(PARENTS[3])['forecasts']; cut_bank = read(PARENTS[4])['cuts']
    cuts = {s: cut_bank[str(ref['job']['producer_fold'])] for ref in previous for s in ref['job']['donor_sites']}
    y = labels['target_eval']-origin[:, None]; valid = labels['valid']
    weight = 1./crowd; sources, recordings = [], []
    for fit in read(PUBLIC/'freeze.json')['fits']:
        seed, s = fit['seed'], fit['held_source']; j = SEEDS.index(seed); ix = np.flatnonzero(sites == s)
        bank = joblib.load(checked(fit['checkpoint'])); np.testing.assert_array_equal(ids[ix], bank['ids'])
        x, spread, floor, scale = model.inputs(history[ix].astype(float), forecasts[:, ix], j, crowd[ix])
        prediction = np.stack([model.predict(h, f, scale) for h, f in zip(bank['heads'], (x, np.column_stack((x, spread))))])
        np.testing.assert_array_equal(prediction, bank['prediction'])
        choice = np.stack([model.matched_choices(q[ix], ids[ix], p) for p in prediction])
        np.testing.assert_array_equal(choice, bank['choices'])
        for g in prior.groups(q[ix]):
            if not (choice[:, g].sum(1) == min(2, len(g)-1)).all():
                raise ValueError('Both arms must preserve every query count')
        paths = np.where(choice[..., None, None], forecasts[j, ix], floor)
        def stats(local):
            take = ix[local]; w = weight[take]; cut = cuts[s]
            row = prior.base.statistics(paths[0, local], paths[1, local], floor[local],
                y[take], valid[take], w/w.sum(), cut['easy'], cut['hard'])[-1]
            truth = model.targets(floor[local], forecasts[j, take], y[take], valid[take], np.ones(len(take)))
            known = valid[take].any(1)
            row.update(seed=seed, source=s, switches=choice[:, local].sum(1).tolist(),
                changed_rows=int((choice[0, local] != choice[1, local]).sum()),
                known_cost_MSE_px2=np.average(np.square(prediction[:, local][:, known]-truth[known]),
                    axis=1, weights=w[known]).tolist() if known.any() else None,
                raw_forecaster=prior.base.statistics(floor[local], forecasts[j, take], floor[local],
                    y[take], valid[take], w/w.sum(), cut['easy'], cut['hard'])[-1]['treatment'])
            return row
        sources.append(stats(np.arange(len(ix))))
        for record in np.unique(records[ix]):
            recordings.append(dict(recording=str(record), **stats(np.flatnonzero(records[ix] == record))))
        beat(state='scored_frozen_checkpoint', seed=seed, held_source=s)
    if len(sources) != 42 or len(recordings) != 102:
        raise ValueError('Preserve every source and recording view for all three seeds')
    values = np.unique(sites); draws = np.random.default_rng(73107).integers(len(values), size=(3000, len(values)))
    def paired(fn):
        x = [[fn(r) for r in sources if r['source'] == s] for s in values]
        if any(v is None for cell in x for v in cell): return None
        a = np.asarray(x).mean(1)
        return dict(point=float(a.mean()), interval95=np.quantile(a[draws].mean(1), [.025, .975]).tolist())
    def easy(r):
        key = 'known_easy_positive_harm_percent'; a, b = (r[arm][key] for arm in ('control', 'treatment'))
        return None if a is None or b is None else b-a
    lower, harm = paired(lambda r: r['paired_completion_lower_pp']), paired(easy)
    once(PUBLIC/'analysis.json', stable(dict(rows=len(ids), sources=sources, recording_views=recordings,
        paired=dict(known_gain_pp=paired(lambda r: r['paired_known_pp']),
            completion_lower_pp=lower, easy_positive_harm_increase_pp=harm),
        component_screen_pass=bool(lower and harm and lower['interval95'][0] > 0 and harm['interval95'][1] <= 0),
        same_count_not_same_risk=True, original_full_policy_gate_run=False,
        independent_confirmation=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    args = parser.parse_args(); banks = register()
    if args.phase == 'register': return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit exact experiment before supervised fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = time.monotonic()
        def beat(**values):
            if shutil.disk_usage(ROOT).free < 2*1024**3: raise OSError('Keep 2 GiB free; resume intact fit cells')
            if time.monotonic()-start > 43200: raise TimeoutError('Resume intact fit cells')
            status = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start, **values)
            prior.base.source.atomic(PRIVATE/'heartbeat.json', status); print(status, flush=True)
        beat(state='start')
        data, forecasts, crowd = population(banks); labels = prior.arrays(('target_eval', 'valid'))
        if args.phase == 'fit': fit_all(data, labels, forecasts, crowd, beat)
        else: evaluate(data, labels, forecasts, crowd, beat)
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifact=artifact(PUBLIC/('freeze.json' if args.phase == 'fit' else 'analysis.json')),
            phase=args.phase, checkpoint_inference_and_actions_exact=args.phase != 'fit')
        once(PUBLIC/(args.phase+'_execution.json'), receipt); beat(state='complete')


if __name__ == '__main__':
    main()
