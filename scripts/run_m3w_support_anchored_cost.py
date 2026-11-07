"""Fit held-source visibility models and test fixed-bank anchored selection."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_seed_dispersion_cost as old
from src.world_model import m3w_support_anchored_cost as model

np, read, sha, artifact, checked, once, stable, joblib = (old.np, old.read, old.sha,
    old.artifact, old.checked, old.once, old.stable, old.joblib)
PUBLIC = old.HOME/'moving_support_anchored_cost_v1'
PRIVATE = old.ROOT/'data/stage_cvpr2027_experiments/moving_support_anchored_cost_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
    ROOT/'tests/test_m3w_support_anchored_cost.py')


def register():
    parent, freeze, replay = (old.PUBLIC/p for p in ('analysis.json', 'freeze.json', 'replay_execution.json'))
    proof, frozen, reg = (read(p) for p in (replay, freeze, old.REG))
    if (proof['artifact']['sha256'] != sha(parent) or not proof['checkpoint_inference_and_actions_exact']
            or not frozen['all_actions_frozen_before_scoring'] or len(frozen['fits']) != 42):
        raise ValueError('Complete exact parent freeze and replay required')
    for ref in (*reg['files'], *reg['parents']):
        checked(ref)
    once(REG, stable(dict(files=[artifact(p) for p in FILES],
        parents=[artifact(p) for p in (old.REG, parent, freeze, replay)], arms=model.ARMS,
        coefficients_tuned=False, original_rules_unchanged=True, independent_confirmation=False,
        new_forecaster_fits=0, new_support_models=42, stage5c_executed=False, smc_enabled=False)))
    return reg['banks'], frozen['fits']


def held_inputs(data, forecasts, crowd, fit):
    ix = np.flatnonzero(data['sites'] == fit['held_source']); j = old.SEEDS.index(fit['seed'])
    x, spread, floor, _ = old.model.inputs(data['history'][ix].astype(float), forecasts[:, ix], j, crowd[ix])
    radius = np.linalg.norm(forecasts[j, ix]-floor, axis=-1).max(1)
    parent = joblib.load(checked(fit['checkpoint']))
    np.testing.assert_array_equal(data['global_ids'][ix], parent['ids'])
    return ix, j, np.column_stack((x, spread)), floor, radius, parent


def actions(data, ix, radius, parent, p, rate):
    q = data['queries'][ix]
    choices = model.choices(q, parent['ids'], parent['prediction'], radius, p, rate)
    np.testing.assert_array_equal(choices[:2], parent['choices'])
    for g in old.prior.groups(q):
        if not (choices[:, g].sum(1) == min(2, len(g)-1)).all():
            raise ValueError('Count mismatch for a query')
    return choices


def fit_all(data, forecasts, crowd, refs, beat):
    sites, history = data['sites'], data['history']; weight = 1./crowd
    for s in np.unique(sites):
        at = sites == s; weight[at] /= weight[at].sum()
    valid = old.prior.arrays(('valid',))['valid']
    receipts = []
    for n, fit in enumerate(refs):
        path, mark = (PRIVATE/f'cell{n:02d}.{ext}' for ext in ('joblib', 'json'))
        if not mark.exists():
            ix, j, held_x, _, radius, parent = held_inputs(data, forecasts, crowd, fit)
            train = sites != fit['held_source']
            if set(sites[train]) != set(fit['fitted_sources']):
                raise ValueError('Changed the original source-held fitting boundary')
            x, spread, _, _ = old.model.inputs(history[train].astype(float), forecasts[:, train], j, crowd[train])
            missing = ~valid[train].any(1)
            clf, rate = model.fit_support(np.column_stack((x, spread)), missing, weight[train])
            p = clf.predict_proba(held_x)[:, 1]
            chosen = actions(data, ix, radius, parent, p, rate)
            pending = path.with_suffix('.pending')
            joblib.dump(dict(clf=clf, rate=rate, probability=p, choices=chosen, ids=parent['ids']), pending, compress=3)
            pending.replace(path)
            once(mark, dict(checkpoint=artifact(path), parent=fit['checkpoint'], registration_sha256=sha(REG),
                seed=fit['seed'], held_source=fit['held_source'], fit_rows=int(train.sum()),
                unknown_fit_rows=int(missing.sum()), fitted_unknown_rate=rate))
        receipt = read(mark)
        if receipt['registration_sha256'] != sha(REG) or receipt['parent'] != fit['checkpoint']:
            raise ValueError('Resume requires original support-model lineage')
        checked(receipt['checkpoint']); receipts.append(receipt)
        beat(state='support_fit', complete=n+1, total=42)
    once(PUBLIC/'freeze.json', dict(fits=receipts, all_actions_frozen_before_scoring=True))


def score(data, forecasts, crowd, refs, beat):
    labels = old.prior.arrays(('target_eval', 'valid'))
    cut_bank = read(old.PARENTS[4])['cuts']
    cuts = {s: cut_bank[str(r['job']['producer_fold'])]
        for r in read(old.PARENTS[3])['forecasts'] for s in r['job']['donor_sites']}
    parent_rows = {(r['seed'], r['source']): r for r in read(old.PUBLIC/'analysis.json')['sources']}
    rows, records = [], []
    frozen = read(PUBLIC/'freeze.json')
    if not frozen['all_actions_frozen_before_scoring'] or len(frozen['fits']) != len(refs):
        raise ValueError('All support fits and actions must freeze first')
    for fit, saved in zip(refs, frozen['fits']):
        if saved['parent'] != fit['checkpoint']: raise ValueError('Changed reference checkpoint')
        s, seed = fit['held_source'], fit['seed']; cut = cuts[s]
        ix, j, x, floor, radius, parent = held_inputs(data, forecasts, crowd, fit)
        bank = joblib.load(checked(saved['checkpoint']))
        np.testing.assert_array_equal(bank['ids'], parent['ids'])
        p = bank['clf'].predict_proba(x)[:, 1]
        np.testing.assert_array_equal(p, bank['probability'])
        chosen = actions(data, ix, radius, parent, p, bank['rate'])
        np.testing.assert_array_equal(chosen, bank['choices'])
        y = labels['target_eval'][ix]-data['origin'][ix, None]; valid = labels['valid'][ix]
        unknown = ~valid.any(1); weight = 1./crowd[ix]
        paths = np.where(chosen[..., None, None], forecasts[j, ix], floor)
        def block(at):
            w = weight[at]; w /= w.sum(); metrics = {}
            for k, name in enumerate(model.ARMS):
                m = old.prior.base.statistics(paths[0, at], paths[k, at], floor[at], y[at], valid[at],
                    w, cut['easy'], cut['hard'])[-1]
                diff = chosen[k, at] != chosen[0, at]
                m.update(switches=int(chosen[k, at].sum()), changed_rows=int(diff.sum()),
                    changed_unknown_rows=int((diff & unknown[at]).sum()))
                metrics[name] = m
            return dict(source=s, seed=seed, rows=len(w), unknown_rows=int(unknown[at].sum()),
                support_brier=float(w @ np.square(p[at]-unknown[at])),
                constant_brier=float(w @ np.square(bank['rate']-unknown[at])), arms=metrics)
        summary = block(np.ones(len(ix), dtype=bool)); previous = parent_rows[seed, s]
        current = summary['arms']['dispersion']
        if any(current[k] != previous[k] for k in ('paired_known_pp', 'paired_completion_lower_pp',
                'control', 'treatment', 'changed_rows')):
            raise ValueError('Original dispersion control did not replay exactly')
        rows.append(summary)
        recording = data['recordings'][ix]
        records.extend(dict(recording=str(r), **block(recording == r)) for r in np.unique(recording))
        beat(state='scored_frozen_decisions', seed=seed, held_source=s)
    if len(rows) != 42 or len(records) != 102: raise ValueError('Incomplete registered population')
    names = np.unique(data['sites']); draw = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    def paired(fn):
        v = [[fn(r) for r in rows if r['source'] == name] for name in names]
        if any(x is None for a in v for x in a): return None
        a = np.asarray(v).mean(1)
        return dict(point=float(a.mean()), interval95=np.quantile(a[draw].mean(1), [.025, .975]).tolist())
    def easy(m):
        key = 'known_easy_positive_harm_percent'; a, b = m['control'][key], m['treatment'][key]
        return None if a is None or b is None else b-a
    contrasts = {}
    for arm in model.ARMS:
        contrasts[arm] = dict(known_gain_pp=paired(lambda r: r['arms'][arm]['paired_known_pp']),
            completion_lower_pp=paired(lambda r: r['arms'][arm]['paired_completion_lower_pp']),
            easy_positive_harm_increase_pp=paired(lambda r: easy(r['arms'][arm])))
    p = contrasts['learned_anchor']; lo, harm = p['completion_lower_pp'], p['easy_positive_harm_increase_pp']
    once(PUBLIC/'analysis.json', stable(dict(rows_per_seed=len(crowd), sources=rows, recording_views=records,
        contrasts_vs_reference=contrasts, support_brier_change=paired(lambda r: r['support_brier']-r['constant_brier']),
        component_screen_pass=bool(lo and harm and lo['interval95'][0] > 0 and harm['interval95'][1] <= 0),
        parent_control_exact=True, original_full_policy_gate_run=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    args = parser.parse_args(); banks, refs = register()
    if args.phase == 'register': return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit the study before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); start = time.monotonic()
        def beat(**values):
            if shutil.disk_usage(ROOT).free < 2*1024**3: raise OSError('Keep 2 GiB free; resume saved cells')
            if time.monotonic()-start > 43200: raise TimeoutError('Resume saved cells')
            status = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start, **values)
            old.prior.base.source.atomic(PRIVATE/'heartbeat.json', status); print(status, flush=True)
        beat(state='start'); data, forecasts, crowd = old.population(banks)
        if args.phase == 'fit': fit_all(data, forecasts, crowd, refs, beat)
        else: score(data, forecasts, crowd, refs, beat)
        once(PUBLIC/(args.phase+'_execution.json'), dict(pid=os.getpid(), seconds=time.monotonic()-start,
            phase=args.phase, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifact=artifact(PUBLIC/('freeze.json' if args.phase == 'fit' else 'analysis.json')),
            probabilities_choices_and_parent_exact=args.phase != 'fit'))
        beat(state='complete')


if __name__ == '__main__':
    main()
