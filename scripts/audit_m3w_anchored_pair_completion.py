"""Refine pair completion with common missing agents across observed neighbors."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_shared_pair_completion as prior
from src.evaluation.m3w_anchored_pair_completion import occurrence_terms

run, np, pair = prior.run, prior.np, prior.pair
PUBLIC = prior.PUBLIC.parent/'native_anchored_pair_completion_v1'
PRIVATE = prior.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(occurrence_terms.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_anchored_pair_completion.py')
ARMS, PAIRS = prior.ARMS, prior.PAIRS
KINDS = ('one_missing', 'two_missing')


def register():
    _, roster = prior.register()
    r = prior.PUBLIC/'analysis.json'
    receipt = prior.PUBLIC/'replay_execution.json'
    if run.read(receipt)['artifact'] != run.artifact(r):
        raise ValueError('Require exact shared-pair parent replay')
    refs = (*FILES, prior.REG, receipt, Path(prior.__file__), Path(prior.shared_bound.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in refs], contrasts=PAIRS,
        original_risk_unchanged=True, old_partition_retained_if_tighter=True,
        grouped_by_single_missing_agent_when_neighbor_observed=True, external_access=False,
        repeats_per_source_after_seed_average=3000)))
    return run.read(r), roster


def evaluate_recording(h, target, rel, valid, paths, agents, frames, qs, indices):
    count, old = prior.recording(h, rel, valid, paths, agents, frames, qs, indices)
    records = [{kind: [] for kind in KINDS} for _ in indices]
    coverage = np.zeros(2, dtype=np.int64)
    for q in qs:
        i, j = pair.local_edges(h[q])
        if not len(i):
            continue
        u, v = q[i], q[j]
        p = paths[:, u]-paths[:, v]+(h[u, -1]-h[v, -1])[None, :, None]
        groups, n = occurrence_terms(agents[u], agents[v], frames[q[0]]+np.arange(1, valid.shape[1]+1),
                                     p, target[u], target[v], valid[u], valid[v])
        coverage += n
        weight = 1./(len(i)*valid.shape[1])
        for kind, (key, vectors) in groups.items():
            for k, (a, b) in enumerate(indices):
                use = np.any(vectors[a] != vectors[b], axis=1)
                if use.any():
                    records[k][kind].append((key[use], vectors[a][use], vectors[b][use],
                                              np.full(int(use.sum()), weight)))
    grouped = []
    for k, event in enumerate(records):
        diagonal, bound = 0., 0.
        repeats = []
        for kind in KINDS:
            items = event[kind]
            if items:
                d, t, n, m = prior.shared_bound(*(np.concatenate(z) for z in zip(*items, strict=True)))
            else:
                d, t, n, m = 0., 0., 0, 0
            diagonal += d
            bound += t
            repeats += [n, m]
        np.testing.assert_allclose(diagonal, old[k, 1], atol=1e-8, rtol=1e-10)
        # Both plans are valid. Retain the old one when the nonuniform
        # greedy transport in the coarser partition was not as good.
        grouped.append([*old[k, :3], min(bound, old[k, 2]), *repeats])
    return count, np.asarray(grouped), coverage


def score(old, roster, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    agents, ids = (np.load(run.checked(roster['files'][k]), mmap_mode='r', allow_pickle=False)
                   for k in ('agents', 'global_ids'))
    np.testing.assert_array_equal(ids, d['global_ids'])
    target, valid, frames = (lab[k] for k in ('target_eval', 'valid', 'frames'))
    agreements = prior.audit_targets(d['recordings'], agents, frames, target, valid)
    if agreements != old['repeated_agent_step_consistency_checks']:
        raise ValueError('The repeated-label cohort changed')
    rel = target-d['origin'][:, None]
    records = prior.prior.h.previous.prior.blocks(d, frames)
    groups = sorted(set(str(x) for x in d['sites']))
    indices = [(ARMS.index(a), ARMS.index(b)) for a, b in PAIRS]
    seeds = tuple(run.kin.prior.SEEDS)
    cells, diagnostics = {}, np.zeros((len(PAIRS), 4), dtype=np.int64)
    coverage = np.zeros(2, dtype=np.int64)
    switch = dict.fromkeys(ARMS[2:], 0)
    queries = 0
    for cell in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            c = {k: z['choices'][run.model.ARMS.index(k)] for k in ARMS[2:]}
        for k, x in c.items():
            switch[k] += int((x >= 0).sum())
        moments = {g: np.zeros((len(PAIRS), 4)) for g in groups}
        counts = dict.fromkeys(groups, 0)
        for rec, (ix, qs) in records.items():
            cvq, b = cv[ix], bank[ix]
            paths = np.stack([cvq, b[:, run.NAMES.index(run.parent.OLS)],
                *(run.model.selected_paths(x[ix], b, cvq) for x in c.values())])
            n, row, cov = evaluate_recording(d['history'][ix], target[ix], rel[ix], valid[ix], paths,
                                             agents[ix], frames[ix], qs, indices)
            g = str(d['sites'][ix[0]])
            moments[g] += row[:, :4]
            diagnostics += row[:, 4:].astype(np.int64)
            counts[g] += n
            queries += n
            coverage += cov
        if not all(counts.values()):
            raise ValueError('Expected complete pair support in each source')
        for g in groups:
            cells[g, seed] = moments[g]/counts[g]
        beat(state='anchored_pair_scored', seed=seed, query_views=queries)
    if switch != old['matched_switches_per_arm'] or queries != old['pair_queries']:
        raise ValueError('Original roster or selected paths changed')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    cube = np.asarray([[cells[g, s] for s in seeds] for g in groups])
    result = {}
    for k, (a, b) in enumerate(PAIRS):
        name = f'{b}_vs_{a}'
        center, diag, shared, anchor = cube[:, :, k].transpose(2, 0, 1)
        row = {field: pair.bootstrap(value, draw) for field, value in dict(
            old_lower=center-diag, shared_lower=center-shared, shared_upper=center+shared,
            anchored_lower=center-anchor, anchored_upper=center+anchor,
            incremental_tightening=shared-anchor).items()}
        for label in ('old_lower', 'shared_lower', 'shared_upper'):
            got, expected = row[label], old['comparisons'][name][label]
            np.testing.assert_allclose([got['point'], *got['interval95']],
                [expected['point'], *expected['interval95']], rtol=1e-10, atol=1e-8)
        result[name] = dict(row, diagnostics=dict(zip(('one_missing_repeated', 'one_missing_reduced_groups',
                                                       'two_missing_repeated', 'two_missing_reduced_groups'),
                                                      map(int, diagnostics[k]))))
    return dict(comparisons=result, sources=len(groups), recordings=len(records), seeds=seeds, pair_queries=queries,
        matched_switches_per_arm=switch, repeated_agent_step_consistency_checks=agreements,
        missing_endpoint_occurrences=dict(zip(KINDS, map(int, coverage))), original_pair_bounds_reproduced=True,
        anchored_completion_is_sensitivity=True, fde_unchanged=True, original_risk_unchanged=True,
        external_access=False, independent_confirmation=False, deployment=False, stage5c_executed=False,
        smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    old, roster = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the diagnostic before real scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = PUBLIC/'analysis.json'
        if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
            raise ValueError('Require exact scoring receipt for replay')

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        run.once(out, run.stable(score(old, roster, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Anchored-pair replay differs')


if __name__ == '__main__':
    main()
