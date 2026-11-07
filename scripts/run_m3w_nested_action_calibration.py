"""Nested source calibration on the frozen eight-action development bank."""
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
from scripts import run_m3w_filter_first_actions as base
from src.world_model import m3w_nested_action_calibration as cal

run, np = base.run, base.np
PUBLIC = run.PUBLIC/'nested_calibration'
PRIVATE = run.PRIVATE/'nested_calibration'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(cal.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_nested_action_calibration.py')


def register():
    parent = [base.PUBLIC/f for f in ('analysis.json', 'replay_execution.json', 'freeze.json')]
    if run.read(parent[1])['artifact']['sha256'] != run.sha(parent[0]):
        raise ValueError('Require verified raw-filter parent')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in (*parent, base.REG, run.REG, run.PUBLIC/'freeze.json',
                 Path(run.__file__), Path(base.__file__), Path(run.model.__file__))],
        budget=.02, residual_quantile=.90, arms=cal.ARMS, source_leave_two_out=True,
        external_outcomes=False, original_rules_unchanged=True)))


def prepare(d, paths, floor, labels, cuts):
    blocks = {}
    for s in np.unique(d['sites']):
        ix = np.flatnonzero(d['sites'] == s); y = labels['target_eval'][ix]-d['origin'][ix, None]
        valid = labels['valid'][ix]; p, cv = paths[ix], floor[ix]
        x, scale = run.model.descriptors(d['history'][ix], p, cv, d['crowd'][ix])
        radius = np.linalg.norm(p - cv[:, None], axis=-1).max(2)
        g = run.old.prior.groups(d['queries'][ix]); groups = [j[np.argsort(d['global_ids'][ix[j]], kind='stable')] for j in g]
        blocks[str(s)] = dict(ix=ix, x=x, p=p, cv=cv, scale=scale, radius=radius,
            groups=groups, rec=d['recordings'][ix],
            moments=run.model.fitting_moments(x, p, cv, y, valid, scale, 1./d['crowd'][ix], cuts[s]['easy']),
            actual=cal.observed_components(p, cv, y, valid, radius, cuts[s]['easy']))
    return blocks


def predict(block, c):
    x, p, cv, scale = (block[k] for k in ('x', 'p', 'cv', 'scale'))
    return run.model.decision_costs(x, c, p, cv, scale)


def fit_seed(d, refs, seed, original, labels, beat):
    cv, paths = run.candidates(d, refs, seed); blocks = prepare(d, paths, cv, labels, run.cuts_by_source())
    receipts = []; choices = np.full((len(cal.ARMS), len(cv)), -1, dtype=np.int16)
    with np.load(run.checked(original['cache']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids']); sources, outer = z['names'], z['coefficients']
    # The inner coefficient for a pair of exclusions can serve either orientation.
    inner = {(h, c): run.model.coefficients([b['moments'] for s, b in blocks.items() if s not in (h, c)])
             for i, h in enumerate(blocks) for c in list(blocks)[i+1:]}
    for held, coef in zip(sources, outer):
        donors = {}
        for s, b in blocks.items():
            if s == held:
                continue
            inner_coef = inner[tuple(sorted((held, s)))]
            u, a, e, ok = predict(b, inner_coef)
            ch = base.select(b['groups'], u, a, e, b['scale'], ok)
            moments = cal.predicted_components(b['x'], inner_coef, b['scale'], b['radius'])
            donors[s] = cal.residuals(moments, b['actual'], b['radius'], ch, b['rec'])
        margin = cal.calibrate(donors, held)
        b = blocks[held]; u, a, e, ok = predict(b, coef)
        for k, arm in enumerate(cal.ARMS):
            ca, ce = cal.adjusted_excess(arm, a, e, b['radius'], margin)
            choices[k, b['ix']] = base.select(b['groups'], u, ca, ce, b['scale'], ok)
        receipts.append(dict(source=str(held), margin=margin.tolist() if margin is not None else None,
                             donor_records=donors, donors_exclude_held_from_predictions=True))
        beat(state='calibrated_source', seed=seed, source=str(held), complete=len(receipts))
    ref = next(f for f in run.read(base.PUBLIC/'freeze.json')['choices'] if f['seed'] == seed)
    with np.load(run.checked(ref['choices']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids']); np.testing.assert_array_equal(z['choice'], choices[0])
    dest = PRIVATE/f'{seed}.npz'; tmp = dest.with_suffix('.pending.npz')
    np.savez_compressed(tmp, choices=choices, ids=d['global_ids']); tmp.replace(dest)
    return dict(seed=seed, decisions=run.artifact(dest), calibrators=receipts, raw_actions_exact=True)


def fit(d, refs, beat):
    labels = run.old.prior.arrays(('target_eval', 'valid'))
    cells = []
    for v in run.read(run.PUBLIC/'freeze.json')['seeds']:
        p = PRIVATE/f'{v["seed"]}.json'
        if not p.exists():
            run.once(p, dict(registration=run.sha(REG), **fit_seed(d, refs, v['seed'], v, labels, beat)))
        r = run.read(p)
        if r['registration'] != run.sha(REG) or not r['raw_actions_exact']:
            raise ValueError('Resume from same registered fits only')
        run.checked(r['decisions']); cells.append(r)
    run.once(PUBLIC/'freeze.json', dict(cells=cells, all_choices_frozen_before_readout=True,
                                      independent_confirmation=False))


def score(d, refs, beat):
    f = run.read(PUBLIC/'freeze.json'); p = run.parent
    if not f['all_choices_frozen_before_readout'] or len(f['cells']) != 3:
        raise ValueError('Full decision freeze required')
    lab = run.old.prior.arrays(('target_eval', 'valid'))
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    sites, rec = d['sites'], d['recordings']; cuts = run.cuts_by_source(); records = []
    old = {(r['seed'], r['recording']): r['metrics']['filter'] for r in run.read(base.PUBLIC/'analysis.json')['records']}
    pairs = {f'{a}_vs_cv': (p.CV, a, p.CV) for a in cal.ARMS}
    pairs.update({f'{a}_vs_raw': ('raw', a, p.CV) for a in cal.ARMS[1:]})
    for cell in f['cells']:
        seed = cell['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['decisions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids']); choices = z['choices']
        for r in np.unique(rec):
            ix = np.flatnonzero(rec == r); s = str(sites[ix[0]]); k = str(r)
            value = p.readout.score_recording({p.CV: floor[ix], **{a: run.model.selected_paths(c[ix], paths[ix], floor[ix])
                for a, c in zip(cal.ARMS, choices)}}, pairs, y[ix], valid[ix], d['queries'][ix], k,
                easy_cut=cuts[s]['easy'], hard_cut=cuts[s]['hard'])
            if value['metrics']['raw'] != old[seed, k]:
                raise ValueError('Parent scalar control must replay exactly')
            records.append(dict(seed=seed, source=s, **value))
        beat(state='scored_seed', seed=seed, views=len(records))
    if len(records) != 102:
        raise ValueError('Keep all repeated recording views')
    names = np.unique(sites); draw = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    grouped = [[r for r in records if r['source'] == s and r['seed'] == seed] for s in names for seed in run.kin.prior.SEEDS]
    def ci(x):
        if any(v is None for v in x):
            return None
        v = np.asarray(x).reshape(len(names), 3).mean(1)
        return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
    contrasts = {}
    for k in pairs:
        v = [p.readout.merge_comparisons([r['comparisons'][k] for r in block]) for block in grouped]
        contrasts[k] = {key: ci([m[key] for m in v]) for key in ('known_pp', 'completion_lower_pp')}
    risk = {}
    for a in cal.ARMS:
        rows = [r['metrics'][a]['risk'] for r in records]
        risk[a] = dict(passes=sum(r['supported'] for r in rows), switches=sum(r['selected'] for r in rows),
            nonempty=sum(r['selected'] > 0 for r in rows), undefined_easy=sum(r['completion_easy_risk_upper'] is None for r in rows),
            known_easy_over_2=sum(r['known_easy_risk'] is not None and r['known_easy_risk'] > .02 for r in rows),
            completion_easy_over_2=sum(r['completion_easy_risk_upper'] is not None and r['completion_easy_risk_upper'] > .02 for r in rows))
    run.once(PUBLIC/'analysis.json', run.stable(dict(risk=risk, contrasts=contrasts, records=records,
        original_rules_unchanged=True, independent_confirmation=False, original_control_exact=True,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register','fit','score','replay'))
    phase = p.parse_args().phase; register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit calibration design before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume saved seed checkpoint')
            packet = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', packet); print(packet, flush=True)
        beat(state='start'); d = run.population(); refs = run.register()
        if phase == 'fit':
            fit(d, refs, beat)
        else:
            score(d, refs, beat)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifact=run.artifact(PUBLIC/('freeze.json' if phase == 'fit' else 'analysis.json'))))


if __name__ == '__main__':
    main()
