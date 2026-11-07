"""Check whether concrete coherent completions can reverse a pair contrast."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_anchored_pair_completion as parent
from src.evaluation.m3w_pair_completion_witness import completion_ledger, extreme

run, np, pair = parent.run, parent.np, parent.pair
PUBLIC = parent.PUBLIC.parent/'native_pair_completion_witness_v1'
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(extreme.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_pair_completion_witness.py')
NAMES = ('pair_additive', 'joint')
CASES = ('interpolation', 'low_1x', 'high_1x', 'low_2x', 'high_2x')


def register():
    _, roster = parent.register()
    receipt, analysis = (parent.PUBLIC/k for k in ('replay_execution.json', 'analysis.json'))
    if run.read(receipt)['artifact'] != run.artifact(analysis):
        raise ValueError('Require the exact anchored-bound result')
    paths = (*FILES, parent.REG, receipt, Path(pair.__file__), run.PUBLIC/'freeze.json')
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in paths], cases=CASES,
        primary='joint_vs_pair_additive', source_bootstrap_draws=3000,
        same_completion_for_all_seeds=True, unchanged_official_score=True, external_access=False)))
    return run.read(analysis), roster


def get_paths(d):
    paths, switch = {}, dict.fromkeys(NAMES, 0)
    refs = run.register()
    for cell in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            c = [z['choices'][run.model.ARMS.index(k)] for k in NAMES]
        for name, choice in zip(NAMES, c, strict=True):
            switch[name] += int((choice >= 0).sum())
        paths[seed] = np.stack([run.model.selected_paths(choice, bank, cv) for choice in c])
    return paths, switch


def recording(agents, frames, h, target, valid, paths, queries):
    ix, known, x0, radius, gaps = completion_ledger(agents, frames, h, target, valid)
    edges, left, right, weights, audit = [], [], [], [], []
    observed = np.zeros(len(paths))
    queries_used = 0
    for q in queries:
        i, j = pair.local_edges(h[q])
        if not len(i):
            continue
        queries_used += 1
        l, r = q[i], q[j]
        mask = valid[l] & valid[r]
        w = 1./mask.size
        origin = h[l, -1]-h[r, -1]
        for k, p in enumerate(paths):
            a, b = p[:, l]-p[:, r]+origin[None, :, None]
            # Identical forecasts contribute exactly zero in every completion.
            change = np.any(a != b, axis=-1)
            v = mask & change
            obs = target[l][v]-target[r][v]
            observed[k] += w*(np.linalg.norm(a[v]-obs, axis=1)-np.linalg.norm(b[v]-obs, axis=1)).sum()
            use = ~mask & change
            if use.any():
                e = np.column_stack((ix[l][use], ix[r][use]))
                edges.append(e)
                left.append(a[use])
                right.append(b[use])
                weights.append(np.full(int(use.sum()), w/len(paths)))
                audit.append((k, e, a[use], b[use], w))
    if not queries_used:
        raise ValueError('Expected an original recording with pair queries')

    def measured(x):
        effect = observed.copy()
        for k, e, a, b, w in audit:
            t = x[e[:, 0]]-x[e[:, 1]]
            effect[k] += w*(np.linalg.norm(a-t, axis=1)-np.linalg.norm(b-t, axis=1)).sum()
        return effect

    out = [measured(x0)]
    status = []
    if edges:
        e, a, b, w = (np.concatenate(v) for v in (edges, left, right, weights))
        for scale in (1., 2.):
            for sign in (-1., 1.):
                gain, x, s = extreme(x0, scale*radius, gaps, e, a, b, w, sign)
                np.testing.assert_array_equal(x[known], x0[known])
                if np.any(np.max(np.abs(x-x0), axis=1) > scale*radius+1e-9):
                    raise ValueError('Completion left the registered bounded perturbation')
                value = measured(x)
                np.testing.assert_allclose(value.mean()-observed.mean(), gain, atol=1e-8, rtol=1e-10)
                out.append(value)
                status.append(s)
    else:
        out.extend([out[0]]*4)
    # Independently use the original pair evaluator on the interpolation
    # world and on all observed terms; the optimization path shares no masks.
    verified, full = np.zeros(len(paths)), np.zeros(len(paths))
    for q in queries:
        for k, p in enumerate(paths):
            rel = target[q]-h[q, -1, None]
            _, v, _, _ = pair.query(h[q], p[:, q], rel, valid[q], [(0, 1)])
            if np.isfinite(v[0, 1]):
                verified[k] += (v[0, 1]+v[0, 2])/2.
            fill = x0[ix[q]]-h[q, -1, None]
            _, v, _, _ = pair.query(h[q], p[:, q], fill, np.ones_like(valid[q]), [(0, 1)])
            if np.isfinite(v[0, 1]):
                full[k] += v[0, 1]
    np.testing.assert_allclose(observed, verified, atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(out[0], full, atol=1e-8, rtol=1e-10)
    return queries_used, np.asarray(out), observed, dict(
        active_occurrences=sum(map(len, edges)), optimizer_runs=status,
        unique_missing_targets=int(len(np.unique(ix[~valid]))))


def score(old, roster, beat):
    d = run.population()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    a, ids = (np.load(run.checked(roster['files'][k]), mmap_mode='r', allow_pickle=False)
              for k in ('agents', 'global_ids'))
    np.testing.assert_array_equal(ids, d['global_ids'])
    y, valid, frames = (lab[k] for k in ('target_eval', 'valid', 'frames'))
    frozen, switch = get_paths(d)
    seeds = tuple(run.kin.prior.SEEDS)
    records = parent.prior.prior.h.previous.prior.blocks(d, frames)
    groups = sorted(set(str(x) for x in d['sites']))
    summary = {g: np.zeros((len(CASES)+1, len(seeds))) for g in groups}
    counts = dict.fromkeys(groups, 0)
    info = []
    for rec, (ix, qs) in records.items():
        n, world, observed, details = recording(a[ix], frames[ix], d['history'][ix], y[ix], valid[ix],
                                                 [frozen[s][:, ix] for s in seeds], qs)
        g = str(d['sites'][ix[0]])
        counts[g] += n
        summary[g] += np.vstack((world, observed))
        info.append(dict(recording=rec, queries=n, **details))
        if len(info) % 8 == 0 or len(info) == len(records):
            beat(state='recordings_scored', recordings=len(info), queries=sum(counts.values()))
    if (sum(counts.values())*len(seeds) != old['pair_queries']
            or any(old['matched_switches_per_arm'][k] != n for k, n in switch.items())):
        raise ValueError('The original matched cohort changed')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    cube = np.stack([summary[g]/counts[g] for g in groups])
    rows = {k: pair.bootstrap(cube[:, j], draw) for j, k in enumerate((*CASES, 'observed_term'))}
    base = old['comparisons']['joint_vs_pair_additive']
    a, b = (base[f'anchored_{q}']['point'] for q in ('lower', 'upper'))
    np.testing.assert_allclose(rows['observed_term']['point'], (a+b)/2., atol=1e-8)
    if any(not a-1e-8 <= rows[k]['point'] <= b+1e-8 for k in CASES):
        raise ValueError('A coherent completion violated the existing envelope')
    return dict(cases=rows, empirical_opposite_signs={str(m): rows[f'low_{m}x']['point'] < 0. <
                rows[f'high_{m}x']['point'] for m in (1, 2)}, by_recording=info, seeds=seeds,
        source_groups=len(groups), recordings=len(records), original_queries=sum(counts.values()),
        bootstrap_scope='conditional_sensitivity_not_inference_about_missing_labels',
        all_seeds_use_the_same_world=True, released_labels_unchanged=True, label_imputation=False,
        global_optimum_claimed=False, official_metric_unchanged=True, original_risk_unchanged=True,
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
            raise ValueError('Commit the protocol before solving')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = PUBLIC/'analysis.json'
        if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
            raise ValueError('Require the verified original result')

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        run.once(out, run.stable(score(old, roster, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Coherent completion replay differs')


if __name__ == '__main__':
    main()
