"""Compare the causal and mixed forecast menus at matched actual intervention counts."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import freeze_m3w_dronecrowd_transport_candidate as prior
from scripts import audit_m3w_neural_bank_capacity as capacity

run, np, pair = prior.run, prior.np, prior.pair
BASE, OLS = prior.BASE, prior.OLS
PUBLIC = prior.PUBLIC.parent/'native_causal_menu_control_v1'
PRIVATE = capacity.PRIVATE.parent.parent/'native_causal_menu_control_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_causal_menu_control.py')
ARMS = ('causal_joint', 'mixed_joint', 'causal_independent', 'mixed_independent',
        'causal_screen', 'mixed_screen', 'causal_common_screen', 'mixed_common_screen')
MENU = np.array([run.NAMES.index(n) for n in capacity.CAUSAL])
LEARNED = np.array([run.NAMES.index(n) for n in capacity.LEARNED])
MATCHED = ((0, 1), (2, 3), (6, 7))
SEEDS = tuple(run.kin.prior.SEEDS)


def register():
    cut = prior.register()
    for p, phase in ((prior.PUBLIC, 'freeze'), (prior.PUBLIC, 'replay'), (capacity.PUBLIC, 'replay')):
        run.checked(run.read(p/f'{phase}_execution.json')['artifact'])
    parents = (prior.REG, prior.PUBLIC/'freeze_execution.json', prior.PUBLIC/'replay_execution.json',
               capacity.REG, capacity.PUBLIC/'replay_execution.json', Path(prior.model.__file__),
               Path(run.model.__file__), Path(pair.__file__), Path(run.parent.readout.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in parents], cut=cut,
        seeds=SEEDS, arms=ARMS, causal_candidates=capacity.CAUSAL,
        learned_candidates=capacity.LEARNED, primary='mixed_joint_vs_causal_joint',
        risk=.02, bootstrap_draws=3000, external_access=False)))
    return cut


def remap(choice, menu):
    """Map causal choices back to the common bank without remapping CV."""
    return np.where(choice >= 0, menu[choice.clip(0)], -1)


def matched(a, b):
    """Abstain symmetrically when a query lacks common count support."""
    ok = int((a >= 0).sum()) == int((b >= 0).sum())
    return (a, b) if ok else (np.full_like(a, -1), np.full_like(b, -1))


def choose(history, bank, cv, crowd, coefficient):
    ols = bank[:, run.NAMES.index(OLS)]
    c = prior.model.decide(history, bank[:, MENU], cv, ols, crowd, coefficient[MENU])
    m = prior.model.decide(history, bank, cv, ols, crowd, coefficient)
    cj, mj = remap(c['joint'], MENU), m['joint']
    ci, mi = remap(c['independent'], MENU), m['independent']
    cg, mg = remap(c['gated_joint'], MENU), m['gated_joint']
    j = matched(cj, mj)
    ind = matched(ci, mi)
    gated = matched(cg, mg)
    if any(np.any(x >= 0) != np.any(y >= 0) for x, y in (j, ind, gated)):
        raise ValueError('Matched controls lost common query support')
    return np.stack((*j, *ind, cg, mg, *gated)), int((cj >= 0).sum() != (mj >= 0).sum())


def freeze(d, refs, beat):
    packet = run.read(run.checked(run.read(prior.PUBLIC/'freeze_execution.json')['artifact']))
    seeds = [c['seed'] for c in packet['cells']]
    if seeds != list(SEEDS):
        raise ValueError('Original frozen seed order changed')
    cells = []
    for cell in packet['cells']:
        seed = cell['seed']
        path = PRIVATE/f'choices_{seed}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            cv, bank = run.candidates(d, refs, seed)
            with np.load(run.checked(cell['cache']), allow_pickle=False) as z:
                coefficients = dict(zip(z['sources'], z['excluded_source_coefficients']))
            choices = np.full((len(ARMS), len(cv)), -1, np.int16)
            mismatches = 0
            for at in d['groups']:
                source = d['sites'][at[0]]
                choices[:, at], miss = choose(d['history'][at], bank[at], cv[at], d['crowd'][at],
                                              coefficients[source])
                mismatches += miss
            if np.isin(choices[[0, 2, 4, 6]], LEARNED).any():
                raise ValueError('A causal-only arm chose a learned trajectory')
            tmp = path.with_suffix('.pending.npz')
            np.savez(tmp, ids=d['global_ids'], choices=choices)
            tmp.replace(path)
            run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(path),
                mismatched_queries=mismatches, per_arm_switches=(choices >= 0).sum(1).tolist(),
                per_arm_learned=np.isin(choices, LEARNED).sum(1).tolist()))
        result = run.read(mark)
        if result['registration'] != run.sha(REG) or result['seed'] != seed:
            raise ValueError('Invalid freeze checkpoint')
        run.checked(result['artifact'])
        cells.append(result)
        beat(seed=seed, state='past_only_choices_frozen')
    out = PRIVATE/'freeze.json'
    run.once(out, run.stable(dict(registration=run.sha(REG), cells=cells, score_labels_read=False)))
    return out


def quality(rows):
    out = {}
    for a in (BASE, OLS, *ARMS):
        x = [r['metrics'][a] for r in rows]
        data = {'available_ADE_pixel': [(v['known_rows'], v['known_ADE_pixel']) for v in x],
                'available_endpoint_FDE_pixel': [(v['endpoint_rows'], v['available_endpoint_FDE_diagnostic_pixel']) for v in x]}
        for subset in ('easy', 'hard'):
            data[f'{subset}_available_ADE_pixel'] = [
                (v['subsets'][subset]['known_rows'], v['subsets'][subset]['model_ADE_pixel']) for v in x]
        out[a] = {}
        for metric, cells in data.items():
            n = sum(k for k, _ in cells)
            out[a][metric] = dict(rows=n, mean=sum(k*v for k, v in cells if k > 0)/n if n else None)
    return out


def score(d, refs, lab, frozen, cut, beat):
    labels, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    records = capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    rows, pairs = [], {}
    count = np.sum([c['per_arm_switches'] for c in frozen['cells']], 0)
    common = bool(count[6])
    contrasts = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    contrasts.update(mixed_joint_vs_causal_joint=('causal_joint', 'mixed_joint', BASE),
        mixed_independent_vs_causal_independent=('causal_independent', 'mixed_independent', BASE),
        mixed_screen_vs_causal_screen=('causal_screen', 'mixed_screen', BASE))
    if common:
        contrasts['mixed_common_screen_vs_causal_common_screen'] = (ARMS[6], ARMS[7], BASE)
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        ols = bank[:, run.NAMES.index(OLS)]
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choices = z['choices']
        by_group = {k: [] for k in groups}
        for rec, (ix, q) in records.items():
            src = roster[rec]
            paths = {BASE: cv[ix], OLS: ols[ix], **{a: run.model.selected_paths(c[ix], bank[ix], cv[ix])
                     for a, c in zip(ARMS, choices)}}
            for at in q:
                actual = [(paths[a][at] != paths[BASE][at]).any(axis=(1, 2)).sum() for a in ARMS]
                if any(actual[a] != actual[b] for a, b in MATCHED):
                    raise ValueError('Actual switch counts differ within a matched query')
                pred = np.stack([paths[a][at] for a in ARMS[:2]])
                _, result, _, _ = pair.query(d['history'][ix[at]], pred,
                                             labels[ix[at]], valid[ix[at]], [(0, 1)])
                by_group[src].append(result[0])
            row = run.parent.readout.score_recording(paths, contrasts, labels[ix], valid[ix], d['queries'][ix],
                                                      rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=src))
        for src, values in by_group.items():
            pairs[src, seed] = pair.mean_finite(values)
        beat(seed=seed, state='frozen_predictions_scored')
    agg = prior.grouped_readout(rows, roster, SEEDS)
    prev = run.read(prior.PUBLIC/'analysis.json')['grouped']['risk']
    mismatch = sum(c['mismatched_queries'] for c in frozen['cells'])
    for name, other in ((BASE, BASE), (OLS, OLS), ('mixed_screen', 'gated_joint')):
        if agg['risk'][name] != prev[other]:
            raise ValueError(f'Original policy changed: {name}')
    if not mismatch:
        for name, other in (('mixed_joint', 'joint'), ('mixed_independent', 'independent')):
            if agg['risk'][name] != prev[other]:
                raise ValueError(f'Original matched policy changed: {name}')
    cells = np.asarray([[pairs[g, seed] for seed in SEEDS] for g in groups])
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    return dict(grouped=agg, pooled_descriptive_accuracy=quality(rows),
        learned_switches=dict(zip(ARMS, np.sum([f['per_arm_learned'] for f in frozen['cells']], 0).tolist())),
        primary_pair_geometry={k: pair.bootstrap(cells[:, :, i], draw) for i, k in enumerate(pair.PAIR_FIELDS)},
        mismatched_queries=mismatch, common_screen_estimable=common,
        independent_confirmation=False, external_access=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = parser.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit the registered design before inference')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            item = dict(pid=os.getpid(), seconds=time.monotonic()-start, phase=phase, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', item)
            print(item, flush=True)

        d, refs = run.population(), run.register()
        if phase == 'freeze':
            out = freeze(d, refs, beat)
        else:
            frozen = run.read(run.checked(run.read(PUBLIC/'freeze_execution.json')['artifact']))
            if frozen['registration'] != run.sha(REG) or frozen['score_labels_read']:
                raise ValueError('Require the registered pre-score freeze')
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, frozen, cut, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Frozen decision readout changed')


if __name__ == '__main__':
    main()
