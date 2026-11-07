"""Test feasibility-before-ranking without changing the frozen cost models."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_record_joint_actions as run

np, m = run.np, run.model
PUBLIC = run.PUBLIC/'filter_first'
PRIVATE = run.PRIVATE/'filter_first'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_filter_first_actions.py')


def select(groups, u, a, e, scale, legal):
    safe = legal & (u > 0) & (a <= 0) & (e <= 0)
    score = np.where(safe, u / scale[:, None], -np.inf)
    best = score.argmax(1); value = score[np.arange(len(u)), best]
    out = np.full(len(u), -1, dtype=np.int16)
    for g in groups:
        count = min(2, len(g)-1)
        ix = g[np.argsort(-value[g], kind='stable')[:count]]
        ix = ix[np.isfinite(value[ix])]; out[ix] = best[ix]
    return out


def registration():
    files = (run.REG, run.PUBLIC/'freeze.json', run.PUBLIC/'budget_diagnosis.json',
             run.PUBLIC/'analysis.json', run.PUBLIC/'replay_execution.json', Path(run.__file__), Path(m.__file__))
    if run.read(files[4])['artifact']['sha256'] != run.sha(files[3]):
        raise ValueError('Require exact registered parent score')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES], parents=[run.artifact(p) for p in files],
        budget=.02, cap_per_query=2, cost_refit=False, all_actions_before_ranking=True, external_outcomes=False)))


def actions(d, refs):
    h, sites, ids, crowd = (d[k] for k in ('history', 'sites', 'global_ids', 'crowd'))
    files = []
    for v in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = v['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(v['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); names, coef = z['names'], z['coefficients']
        u = np.empty(paths.shape[:2]); a = np.empty_like(u); e = np.empty_like(u)
        scale = np.empty(len(u)); legal = np.empty_like(u, dtype=bool)
        for s, c in zip(names, coef):
            ix = np.flatnonzero(sites == s)
            x, scale[ix] = m.descriptors(h[ix], paths[ix], floor[ix], crowd[ix])
            u[ix], a[ix], e[ix], legal[ix] = m.decision_costs(x, c, paths[ix], floor[ix], scale[ix])
        chosen = select(d['groups'], u, a, e, scale, legal)
        selected = np.flatnonzero(chosen >= 0); j = (selected, chosen[selected])
        if not ((a[j] <= 0.) & (e[j] <= 0.) & (u[j] > 0) & legal[j]).all():
            raise ValueError('Predicted constraints not respected')
        p = PRIVATE/f'{seed}.npz'
        if not p.exists():
            tmp = p.with_suffix('.pending.npz'); np.savez_compressed(tmp, ids=ids, choice=chosen); tmp.replace(p)
        else:
            with np.load(p, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids); np.testing.assert_array_equal(z['choice'], chosen)
        files.append(dict(seed=seed, choices=run.artifact(p), switches=len(selected)))
        print(dict(seed=seed, frozen_switches=len(selected)), flush=True)
    run.once(PUBLIC/'freeze.json', dict(choices=files, future_labels_used=False))


def score(d, refs):
    f = run.read(PUBLIC/'freeze.json')
    if f['future_labels_used'] or len(f['choices']) != 3:
        raise ValueError('Full past-only decision freeze required')
    labels = run.old.prior.arrays(('target_eval', 'valid'))
    ids, y, valid = d['global_ids'], labels['target_eval']-d['origin'][:, None], labels['valid']
    sites, rec = d['sites'], d['recordings']; cuts = run.cuts_by_source()
    p = run.parent; results = []; pair = {'filter_vs_cv': (p.CV, 'filter', p.CV)}
    for r in f['choices']:
        seed = r['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(r['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); choice = z['choice']
        prediction = m.selected_paths(choice, paths, floor)
        for key in np.unique(rec):
            ix = np.flatnonzero(rec == key); s = str(sites[ix[0]]); cut = cuts[s]
            value = p.readout.score_recording({p.CV: floor[ix], 'filter': prediction[ix]}, pair,
                y[ix], valid[ix], d['queries'][ix], str(key), easy_cut=cut['easy'], hard_cut=cut['hard'])
            results.append(dict(seed=seed, source=s, **value))
    if len(results) != 102:
        raise ValueError('All recording views required')
    names = np.unique(sites); draws = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    merged = [p.readout.merge_comparisons([r['comparisons']['filter_vs_cv'] for r in results
        if r['source'] == s and r['seed'] == seed]) for s in names for seed in run.kin.prior.SEEDS]
    def ci(key):
        x = [v[key] for v in merged]
        if any(v is None for v in x):
            return None
        x = np.asarray(x).reshape(len(names), 3).mean(1)
        return dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    risks = [r['metrics']['filter']['risk'] for r in results]
    run.once(PUBLIC/'analysis.json', run.stable(dict(records=results,
        risk_passes=sum(v['supported'] for v in risks), switches=sum(v['selected'] for v in risks),
        unknown_switches=sum(v['unknown_selected'] for v in risks),
        nonempty_views=sum(v['selected'] > 0 for v in risks),
        undefined_easy=sum(v['completion_easy_risk_upper'] is None for v in risks),
        easy_over_2=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper'] > .02 for v in risks),
        known_pp=ci('known_pp'), completion_lower_pp=ci('completion_lower_pp'),
        cost_refit=False, original_rules_unchanged=True, independent_confirmation=False, deployment_changed=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register','freeze','score','replay'))
    phase = p.parse_args().phase; registration()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit solver change before evaluating')
    PRIVATE.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    refs = run.register(); d = run.population()
    if phase == 'freeze':
        actions(d, refs)
    else:
        score(d, refs)
    run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        artifact=run.artifact(PUBLIC/('freeze.json' if phase == 'freeze' else 'analysis.json'))))


if __name__ == '__main__':
    main()
