"""Re-evaluate frozen pair ADE bounds without treating repeated targets as free."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_joint_path_audit as prior
from src.evaluation.m3w_shared_pair_completion import audit_targets, shared_bound

run, np, pair = prior.run, prior.np, prior.model
PUBLIC = prior.PUBLIC.parent/'native_shared_pair_completion_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(shared_bound.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_shared_pair_completion.py')
ARMS = prior.ARMS[:6]
PAIRS = prior.PAIRS[:5]


def register():
    old = run.read(prior.PUBLIC/'replay_execution.json')
    result = prior.PUBLIC/'analysis.json'
    if old['analysis_sha256'] != run.sha(result):
        raise ValueError('Require the replayed original pair analysis')
    roster = run.old.prior.PARENT/'population.json'
    sources = (*FILES, result, prior.REG, prior.PUBLIC/'replay_execution.json', roster,
               run.PUBLIC/'freeze.json', Path(pair.__file__), Path(run.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in sources],
        contrasts=PAIRS, original_query_weights=True, original_gate_unchanged=True,
        native_annotation_offsets=list(range(1, 13)), bootstrap_draws=3000, bootstrap_seed=73107,
        shared_target_key=['recording', 'ordered_agent_pair', 'absolute_annotation_frame'],
        label_aware_scoring_only=True, external_access=False)))
    return run.read(result), run.read(roster)


def recording(h, y, valid, paths, agents, frames, queries, indices):
    center, debit = np.zeros(len(indices)), np.zeros(len(indices))
    missing = [[] for _ in indices]
    count = 0
    for q in queries:
        p = paths[:, q]
        i, j = pair.local_edges(h[q])
        if not len(i):
            continue
        count += 1
        _, exact, _, _ = pair.query(h[q], p, y[q], valid[q], indices)
        center += (exact[:, 1]+exact[:, 2])/2.
        debit += (exact[:, 2]-exact[:, 1])/2.
        qi, qj = q[i], q[j]
        sign = np.where(agents[qi] < agents[qj], 1., -1.)
        # Absolute pair vectors must use a canonical orientation. The offsets
        # from different query origins alone do not refer to one target.
        relative = (p[:, i]-p[:, j]+(h[qi, -1]-h[qj, -1])[None, :, None])*sign[None, :, None, None]
        unknown = ~(valid[qi] & valid[qj])
        if not np.all(frames[q] == frames[q[0]]):
            raise ValueError('Query mixes annotation origins')
        keys = np.empty((*unknown.shape, 3), dtype=np.int64)
        keys[..., 0] = np.minimum(agents[qi], agents[qj])[:, None]
        keys[..., 1] = np.maximum(agents[qi], agents[qj])[:, None]
        keys[..., 2] = frames[q[0]]+np.arange(1, valid.shape[1]+1)
        w = 1./unknown.size
        for k, (a, b) in enumerate(indices):
            use = unknown & np.any(relative[a] != relative[b], axis=-1)
            if use.any():
                missing[k].append((keys[use], relative[a][use], relative[b][use], np.full(int(use.sum()), w)))
    rows = []
    for k, items in enumerate(missing):
        if items:
            keys, a, b, weights = (np.concatenate(z) for z in zip(*items, strict=True))
            old, tight, repeated, improved = shared_bound(keys, a, b, weights)
            np.testing.assert_allclose(old, debit[k], atol=1e-8, rtol=1e-10)
        else:
            tight, repeated, improved = 0., 0, 0
            np.testing.assert_allclose(debit[k], 0., atol=1e-8)
        rows.append([center[k], debit[k], min(tight, debit[k]), repeated, improved])
    return count, np.asarray(rows)


def score(old, roster, beat):
    d, refs = run.population(), run.register()
    labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    agents, gids = (np.load(run.checked(roster['files'][k]), mmap_mode='r', allow_pickle=False)
                    for k in ('agents', 'global_ids'))
    np.testing.assert_array_equal(gids, d['global_ids'])
    target, valid, frames = (labels[k] for k in ('target_eval', 'valid', 'frames'))
    agreements = audit_targets(d['recordings'], agents, frames, target, valid)
    relative = target-d['origin'][:, None]
    records = prior.h.previous.prior.blocks(d, frames)
    groups = sorted(set(str(x) for x in d['sites']))
    ix_map = [(ARMS.index(a), ARMS.index(b)) for a, b in PAIRS]
    cells, support, switches = {}, np.zeros((len(PAIRS), 2), dtype=np.int64), dict.fromkeys(ARMS[2:], 0)
    queries = 0
    seeds = tuple(run.kin.prior.SEEDS)
    for cell in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], gids)
            choices = {k: z['choices'][run.model.ARMS.index(k)] for k in ARMS[2:]}
        for k, c in choices.items():
            switches[k] += int((c >= 0).sum())
        totals = {g: np.zeros((len(PAIRS), 3)) for g in groups}
        by_group = dict.fromkeys(groups, 0)
        for rec, (ix, qs) in records.items():
            floor, local = cv[ix], bank[ix]
            paths = np.stack([floor, local[:, run.NAMES.index(run.parent.OLS)],
                *(run.model.selected_paths(c[ix], local, floor) for c in choices.values())])
            n, stats = recording(d['history'][ix], relative[ix], valid[ix], paths, agents[ix], frames[ix], qs, ix_map)
            group = str(d['sites'][ix[0]])
            by_group[group] += n
            totals[group] += stats[:, :3]
            support += stats[:, 3:].astype(np.int64)
            queries += n
        for g in groups:
            if not by_group[g]:
                raise ValueError('Full source support required')
            cells[g, seed] = totals[g]/by_group[g]
        beat(state='source_group_bounds_scored', seed=seed, pair_queries=queries)
    if set(switches.values()) != {9675}:
        raise ValueError('Original matched decisions changed')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    table = np.asarray([[cells[g, s] for s in seeds] for g in groups])
    result = {}
    for k, (a, b) in enumerate(PAIRS):
        name = f'{b}_vs_{a}'
        center, debit, new = table[:, :, k].transpose(2, 0, 1)
        r = {name: pair.bootstrap(value, draw) for name, value in dict(
            old_lower=center-debit, old_upper=center+debit, shared_lower=center-new, shared_upper=center+new,
            bound_tightening=debit-new).items()}
        for bound in ('lower', 'upper'):
            expected = old['comparisons'][name][f'grid_ade_{bound}']
            got = r[f'old_{bound}']
            np.testing.assert_allclose([got['point'], *got['interval95']],
                [expected['point'], *expected['interval95']], atol=1e-8, rtol=1e-10)
        result[name] = dict(r, repeated_nonzero_unknown_occurrences=int(support[k, 0]),
                            groups_with_tighter_bound=int(support[k, 1]))
    return dict(comparisons=result, sources=len(groups), recordings=len(records), seeds=seeds,
        pair_queries=queries, repeated_agent_step_consistency_checks=agreements,
        matched_switches_per_arm=switches, bootstrap_draws=3000, original_pair_bounds_reproduced=True,
        shared_completion_is_sensitivity=True, fde_unchanged=True, original_risk_unchanged=True,
        external_access=False, independent_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    old, roster = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the diagnostic before scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = PUBLIC/'analysis.json'
        if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
            raise ValueError('Require the verified original analysis')

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        run.once(out, run.stable(score(old, roster, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Shared-target replay differs')


if __name__ == '__main__':
    main()
