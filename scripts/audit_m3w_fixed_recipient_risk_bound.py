"""Exact component minima for the frozen recipients, without exporting hindsight choices."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_visibility_increment as prev
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments

run, np = prev.run, prev.np
PUBLIC = prev.PUBLIC.parent/'native_fixed_recipient_risk_bound_v1'
REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_fixed_recipient_risk_bound.py')
RISK = ('all_excess', 'easy_excess', 'easy_signed')


def query_minimum(anchor, neural_min, k):
    delta = neural_min-anchor
    if not 0 <= k <= len(anchor) or (np.isfinite(neural_min[:, 0]).sum() < k):
        raise ValueError('Cannot realize the frozen neural count')
    fixed = anchor.sum(0)+np.sort(delta, axis=0)[:k].sum(0)
    free = np.minimum(anchor, neural_min).sum(0)
    return fixed, free


def assess(cv, bank, choices, queries, y, valid, easy_cut):
    a, control = choices
    use = a >= 0
    active = np.flatnonzero(use)
    c, z, v = cv[use], y[use], valid[use]
    anchor = bank[active, a[use]]
    base = moments(c, anchor, z, v, easy_cut)
    baseline = np.column_stack([base[k] for k in RISK])
    candidate = np.full_like(baseline, np.inf)
    for k in prev.LEARNED:
        path = bank[use, k]
        legal = (path != c).any(axis=(1, 2)) & (path != anchor).any(axis=(1, 2))
        m = moments(c, path, z, v, easy_cut)
        h = np.column_stack([m[n] for n in RISK])
        candidate = np.minimum(candidate, np.where(legal[:, None], h, np.inf))
    local = np.cumsum(use)-1
    total = np.zeros((2, 3))
    for g in queries:
        at = local[g[use[g]]]
        k = int(np.isin(control[g], prev.LEARNED).sum())
        total += query_minimum(baseline[at], candidate[at], k)
    ref, _ = run.model.errors(cv, y, valid)
    easy = valid.any(1) & (ref > 0.) & (ref <= easy_cut)
    full_easy = float(ref[easy].sum())
    mass, easy_mass = float(base['base'].sum()), float(base['ref_easy'].sum())
    total[:, 2] -= BUDGET*full_easy
    tol = 1e-8*np.maximum([mass, easy_mass, full_easy], 1.)
    if np.any(total[1] > total[0]+tol):
        raise ValueError('Relaxing the count cannot raise a component minimum')
    lack = not (mass > 0. and easy_mass > 0. and full_easy > 0.)
    results = []
    for value in total:
        out = {f'proved_{name}': bool(x > t) for name, x, t in zip(RISK, value, tol)}
        out['missing_reference'] = lack
        out['provably_infeasible'] = any(out.values())
        results.append(out)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = prev.register()
    frozen = prev.receipt(prev.PUBLIC, 'freeze')
    parents = (prev.REG, prev.PUBLIC/'freeze_execution.json', prev.PUBLIC/'score_execution.json',
               Path(prev.__file__), Path(moments.__code__.co_filename), PUBLIC/'registration.json', *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(x) for x in parents],
        risk=.02, audit='componentwise_necessary_conditions', relative_tolerance=1e-8,
        roles='exposed_development_only', seeds=prev.SEEDS, external_access=False)))
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the risk-bound design before execution')
    if phase == 'replay':
        prev.receipt(PUBLIC, 'score')
    t = time.monotonic()
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = prev.prev.prior.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    labels, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    results = []
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            arms = z['choices'][:2]
        for rec, (ix, queries) in records.items():
            proof = assess(cv[ix], bank[ix], arms[:, ix], queries, labels[ix], valid[ix], cut['easy'])
            results.append((rec, str(d['sites'][ix[0]]), proof))
        print(dict(seed=seed, completed=len(results), phase=phase), flush=True)
    if len(results) != 102:
        raise ValueError('Require all recording-seed views')
    table = {}
    for i, name in enumerate(('fixed_neural_count', 'free_neural_count')):
        view = [r[2][i] for r in results]
        blocked = [r for r, v in zip(results, view) if v['provably_infeasible']]
        table[name] = dict(counts={k: sum(r[k] for r in view) for k in view[0]},
            distinct_recordings=len({r[0] for r in blocked}), source_groups=len({r[1] for r in blocked}),
            not_ruled_out=len(view)-len(blocked))
    out = PUBLIC/'analysis.json'
    run.once(out, run.stable(dict(views=len(results), bounds=table, no_hindsight_choices_exported=True,
        independent_confirmation=False, external_access=False, deployment=False, stage5c_executed=False, smc_enabled=False)))
    mark = PUBLIC/f'{phase}_execution.json'
    if not mark.exists():
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(out)))
    elif run.read(mark)['artifact'] != run.artifact(out):
        raise ValueError('Risk bound replay differs')


if __name__ == '__main__':
    main()
