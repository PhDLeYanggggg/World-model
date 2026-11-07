"""Measure the cost of resolving missing pair terms without guessing labels."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_pair_completion_witness as prior
from src.evaluation.m3w_pair_annotation_budget import aggregate, curves, minimum_relaxation

run, np, pair = prior.run, prior.np, prior.pair
PUBLIC = prior.PUBLIC.parent/'native_pair_annotation_budget_v1'
PRIVATE = prior.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(aggregate.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_pair_annotation_budget.py')
FRACTIONS = (.5, .75, .9, .95, .99)


def register():
    anchored, roster = prior.register()
    witness = prior.PUBLIC/'analysis.json'
    receipt = prior.PUBLIC/'replay_execution.json'
    if run.read(receipt)['artifact'] != run.artifact(witness):
        raise ValueError('Require the verified sign witnesses on the same cohort')
    parents = (*FILES, prior.REG, receipt, Path(prior.__file__), run.PUBLIC/'freeze.json', Path(pair.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], fractions=FRACTIONS,
        contrast='joint_vs_pair_additive', uses_forecast_disagreement=True, reads_label_coordinates=False,
        both_missing_endpoints_required=True, official_metric_unchanged=True, external_access=False)))
    return anchored, run.read(witness), roster


def missing_ids(agents, frames, valid, offset):
    i, j = np.nonzero(~valid)
    keys, inv = np.unique(np.column_stack((frames[i]+j+1, agents[i])), axis=0, return_inverse=True)
    lookup = np.full(valid.shape, -1, dtype=np.int64)
    lookup[i, j] = inv+offset
    return lookup, len(keys)


def score(old, witness, roster, beat):
    d = run.population()
    lab = run.old.prior.arrays(('valid', 'frames'))
    ids, agents = (np.load(run.checked(roster['files'][k]), mmap_mode='r', allow_pickle=False)
                   for k in ('global_ids', 'agents'))
    np.testing.assert_array_equal(ids, d['global_ids'])
    paths, switches = prior.get_paths(d)
    if any(old['matched_switches_per_arm'][a] != b for a, b in switches.items()):
        raise ValueError('The frozen matched-count cohort changed')
    valid, frame = lab['valid'], lab['frames']
    recs = prior.parent.prior.prior.h.previous.prior.blocks(d, frame)
    ordered = sorted(recs, key=int)
    sites = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in recs.items()}
    groups = sorted(set(sites.values()))
    counts = {g: sum(len(recs[rec][1]) for rec in ordered if sites[rec] == g) for g in groups}
    terms, masses, records = [], [], []
    offset = 0
    for rec in ordered:
        ix, queries = recs[rec]
        local_valid = valid[ix]
        h = d['history'][ix]
        link, n = missing_ids(agents[ix], frame[ix], local_valid, offset)
        offset += n
        by_seed = [p[:, ix] for p in paths.values()]
        before = len(terms)
        for q in queries:
            i, j = pair.local_edges(h[q])
            if not len(i):
                raise ValueError('The parent source average has a non-pair query')
            l, r = q[i], q[j]
            missing = ~(local_valid[l] & local_valid[r])
            denom = missing.size*len(by_seed)*counts[sites[rec]]*len(groups)
            for p in by_seed:
                a, b = p[:, l]-p[:, r]
                radius = np.linalg.norm(a-b, axis=2)
                use = missing & (radius > 0.)
                if use.any():
                    terms.append(np.column_stack((link[l][use], link[r][use])))
                    masses.append(radius[use]/denom)
        records.append(dict(recording=rec, missing_positions=n,
                            diagonal_mass=float(sum(v.sum() for v in masses[before:]))))
    a, w = np.concatenate(terms), np.concatenate(masses)
    if (offset != sum(r['unique_missing_targets'] for r in witness['by_recording'])
            or sum(counts.values()) != witness['original_queries']):
        raise ValueError('Expected the same full missing-position ledger')
    parent = old['comparisons']['joint_vs_pair_additive']
    # The reported parent provides the original lower and shared envelope.
    # The center is unchanged, so the old debit is center minus old lower.
    width = (parent['shared_upper']['point']+parent['shared_lower']['point'])/2.-parent['old_lower']['point']
    np.testing.assert_allclose(w.sum(), width, atol=1e-9, rtol=1e-10)
    keys, mass = aggregate(a, w)
    c = curves(keys, mass)
    n = int(keys.max())+1
    budget = {str(f): dict(lp_lower_positions=minimum_relaxation(keys, mass, f),
                          positions={k: int(np.searchsorted(v, f-1e-12)) for k, v in c.items()})
              for f in FRACTIONS}
    for q in budget.values():
        if q['lp_lower_positions'] > min(q['positions'].values())+1e-6:
            raise ValueError('Fractional lower bound exceeds a verified prefix budget')
    grid = np.unique(np.rint(np.linspace(0, n, 101)).astype(int))
    beat(state='budget_solved', influential_positions=n, terms=len(a), recordings=len(records))
    return dict(budgets=budget, curves=[dict(positions=int(i), **{k: float(v[i]) for k, v in c.items()})
                                       for i in grid],
        total_missing_positions=offset, influential_positions=n, raw_nonzero_pair_occurrences=len(a),
        unique_dependency_terms=len(keys), single_endpoint_terms=int((keys[:, 1] < 0).sum()),
        original_diagonal_radius_px=float(w.sum()), by_recording=records,
        sources=len(groups), recordings=len(recs), three_forecaster_seeds=True,
        matched_switches_per_arm=switches, all_positions_conditional_on_label_availability=True,
        sign_of_newly_observed_terms_unknown=True, labels_collected=False, official_metric_unchanged=True,
        external_access=False, independent_confirmation=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    old, witness, roster = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the protocol before scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = PUBLIC/'analysis.json'
        if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
            raise ValueError('Require a verified original budget')

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        run.once(out, run.stable(score(old, witness, roster, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Pair-label budget replay differs')


if __name__ == '__main__':
    main()
