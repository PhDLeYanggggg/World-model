"""Test multiple causal path assignments at the same recipient and switch budgets."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_witness_retrieval as prev
from src.evaluation import m3w_joint_path_audit as pair
from src.world_model import m3w_joint_path_portfolio as model

run, np, cal = prev.run, prev.np, prev.cal
PUBLIC = prev.PUBLIC.parent/'native_joint_path_portfolio_v1'
PRIVATE = prev.PRIVATE.parent/'native_joint_path_portfolio_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_joint_path_portfolio.py')
BASE, OLS, SEEDS = prev.BASE, prev.OLS, prev.SEEDS
ARMS = ('single', 'single_gated', 'expanded_raw', 'matched_additive', 'matched_joint', 'screen_first_joint')
EXTRA = (('single', 'expanded_raw'), ('single_gated', 'matched_joint'),
         ('matched_additive', 'matched_joint'), ('single_gated', 'screen_first_joint'))


def register():
    cut, costs = prev.register()
    for k in ('fit', 'replay', 'retention'):
        run.checked(run.read(prev.PUBLIC/f'{k}_execution.json')['artifact'])
    parents = (prev.REG, prev.PUBLIC/'analysis.json', prev.PUBLIC/'retention.json',
               prev.PUBLIC/'retention_execution.json', Path(pair.__file__), Path(run.model.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut,
        seeds=SEEDS, arms=ARMS, contrast=EXTRA, primary='matched_joint_vs_single_gated',
        original_risk_gate=True, no_new_training=True, same_recipient_roster=True,
        labels_for_readout_only=True, external_data_access=False)))
    return cut, costs, run.read(prev.PRIVATE/'freeze.json')['cells']


def freeze(d, refs, fixed, costs, beat):
    seed = fixed['seed']
    dest = PRIVATE/f'seed_{seed}.json'
    if not dest.exists():
        cv, ols, bank = prev.prior.bank_for(d, refs, seed)
        with np.load(run.checked(fixed['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            old, mask, cap = f['chosen'][[1, 3]], f['retrieved'][1], f['cap']
        sites, h, crowd = (d[k] for k in ('sites', 'history', 'crowd'))
        with np.load(run.checked(costs['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            coef = dict(zip(sorted(set(sites)), f['cost_coefficients']))
        choice = np.full((len(ARMS), len(cv)), -1, dtype=np.int16)
        choice[:2] = old
        stats = []
        for q in d['groups']:
            c = coef[sites[q[0]]]
            f, scale = run.model.descriptors(h[q], bank[q], cv[q], crowd[q])
            u, a, e, legal = run.model.decision_costs(f, c, bank[q], cv[q], scale)
            score = np.einsum('nad,adc->nac', f, c)
            support = np.minimum(score[..., 4].clip(0.), score[..., 3].clip(0.))*scale[:, None]
            val = model.choose(h[q], cv[q], ols[q], bank[q], scale, u, a, e, support, legal & mask[q, None],
                int(cap[q[0]]), old[0, q], (old[1, q] >= 0).any())
            choice[2:, q] = np.stack(val[:4])
            counts = (choice[:, q] >= 0).sum(1)
            if (counts[2] != counts[0] or not np.all(counts[3:5] == counts[1])
                    or counts[5] > counts[0] or counts[5] < counts[1]):
                raise ValueError('Lost the declared matched budgets')
            stats.append([*val[4:], *(int((choice[b, q] != choice[a, q]).any())
                           for a, b in ((0, 2), (1, 4), (3, 4)))])
        out, tmp = dest.with_suffix('.npz'), dest.with_suffix('.pending.npz')
        np.savez(tmp, ids=d['global_ids'], choice=choice, diagnostics=np.asarray(stats))
        tmp.replace(out)
        run.once(dest, run.stable(dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(out))))
    row = run.read(dest)
    if row['registration'] != run.sha(REG) or row['seed'] != seed:
        raise ValueError('Stale path portfolio freeze')
    run.checked(row['artifact'])
    beat(seed=seed, state='past_only_options_frozen')
    return row


def score(d, refs, lab, records, cut, cells, beat):
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    contrasts = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    contrasts.update({f'{b}_vs_{a}': (a, b, BASE) for a, b in EXTRA})
    indices = [(ARMS.index(a), ARMS.index(b)) for a, b in EXTRA]
    rows, roster, interactions, diag = [], {}, {}, np.zeros(5, int)
    for c in cells:
        seed = c['seed']
        cv, ols, bank = prev.prior.bank_for(d, refs, seed)
        with np.load(run.checked(c['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            chosen = f['choice']
            diag += f['diagnostics'].sum(0)
        for r, (ix, groups) in records.items():
            s = str(d['sites'][ix[0]])
            roster[r] = s
            values = np.stack([run.model.selected_paths(v[ix], bank[ix], cv[ix]) for v in chosen])
            row = run.parent.readout.score_recording({BASE: cv[ix], OLS: ols[ix], **dict(zip(ARMS, values))},
                contrasts, y[ix], valid[ix], d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=s))
            block = interactions.setdefault((s, seed), [])
            for g in groups:
                _, p, _, _ = pair.query(d['history'][ix[g]], values[:, g], y[ix[g]], valid[ix[g]], indices)
                block.append(p)
        beat(seed=seed, state='original_scoring_complete')
    result = prev.prior.grouped.grouped_readout(rows, roster, SEEDS)
    orig = run.read(prev.PUBLIC/'analysis.json')['grouped']['risk']
    for a, b in (('single', 'witness'), ('single_gated', 'gated_witness')):
        if result['risk'][a] != orig[b]:
            raise ValueError('Single-path reference changed')
    src = sorted(set(roster.values()))
    values = np.array([[pair.mean_finite(interactions[s, z]) for z in SEEDS] for s in src])
    draw = np.random.default_rng(73107).integers(len(src), size=(3000, len(src)))
    geometry = {f'{b}_vs_{a}': {k: pair.bootstrap(values[:, :, i, j], draw)
                 for j, k in enumerate(pair.PAIR_FIELDS)} for i, (a, b) in enumerate(EXTRA)}
    return dict(grouped=result, relative_geometry=geometry,
        portfolio_diagnostics=dict(zip(('enumerated_options', 'admissible_options', 'raw_changed_queries',
                                       'matched_joint_changed_queries', 'coupling_changed_queries'), diag.tolist())),
        independent_confirmation=False, external_access=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, cost_cells, prev_cells = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit path portfolio protocol before executing')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        d, refs = run.population(), run.register()
        receipt = PRIVATE/'freeze.json'
        if phase == 'freeze':
            cells = [freeze(d, refs, a, b, beat) for a, b in zip(prev_cells, cost_cells, strict=True)
                     if a['seed'] == b['seed']]
            if tuple(c['seed'] for c in cells) != SEEDS:
                raise ValueError('Incomplete aligned seed freeze')
            run.once(receipt, run.stable(dict(cells=cells)))
            out = receipt
        else:
            run.checked(run.read(PUBLIC/'freeze_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            records = cal.prior.h.previous.prior.blocks(d, labels['frames'])
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, labels, records, cut, run.read(receipt)['cells'], beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(artifact=run.artifact(out), pid=os.getpid(), seconds=time.monotonic()-start))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Replayed portfolio differs')


if __name__ == '__main__':
    main()
