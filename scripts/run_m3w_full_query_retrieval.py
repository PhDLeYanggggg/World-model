"""Fit fixed-size recipient shortlists on the full past-only query."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_recipient_budget_capacity as prior
from src.evaluation import m3w_grouped_recording_readout as grouped
from src.world_model import m3w_full_query_retrieval as model

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC.parent/'native_full_query_retrieval_v1'
PRIVATE = prior.PRIVATE.parent/'native_full_query_retrieval_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_full_query_retrieval.py')
NAMES = ('gain', 'opportunity', 'gated_gain', 'gated_opportunity', 'gated_original')
BASE, OLS = cal.BASE, cal.OLS
SEEDS = prior.g.prior.SEEDS


def register():
    cut, ctrl = prior.register()
    parents = (prior.REG, prior.PUBLIC/'refinement.json', prior.PUBLIC/'refinement_execution.json',
               Path(grouped.__file__), Path(prior.model.__file__),
               Path(model.history_features.__code__.co_filename), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut, seeds=SEEDS,
        arms=NAMES, shortlist=model.SHORTLIST, ridge=model.RIDGE, max_solver_seconds=4.,
        primary='matched_shortlist_positive_ols4_capacity', original_gate_unchanged=True,
        fit_rule='leave_one_exposed_source_out_equal_source_and_query_weight', external_data_access=False)))
    return cut, ctrl


def bank_for(d, refs, seed):
    cv, bank = run.candidates(d, refs, seed)
    ols, kin = (bank[:, run.NAMES.index(k)] for k in (OLS, 'self'))
    common = np.empty_like(ols)
    for q in d['groups']:
        common[q] = ols[q] + ((cv[q]-ols[q])+(kin[q]-ols[q])).mean(0)/3.
    return cv, ols, np.concatenate((bank, common[:, None]), axis=1)


def gate(choice, u, a, e, s):
    return cal.model.transport.gate(choice, u, a, e, s)


def fit(d, refs, y, valid, cut, cell, beat):
    seed = cell['seed']
    dest = PRIVATE/f'seed_{seed}.npz'
    mark = dest.with_suffix('.json')
    if not mark.exists():
        cv, ols, bank = bank_for(d, refs, seed)
        h, sites, crowd, queries = (d[k] for k in ('history', 'sites', 'crowd', 'groups'))
        x = model.features(h, bank, cv, ols, queries, crowd)
        old = bank[:, :-1]
        legal = np.linalg.norm(old-cv[:, None], axis=-1).max(2) > 0.
        cap = [min(2, len(q)-1, int(legal[q].sum(0).max())) for q in queries]
        scale = 1.+6.5*np.median(np.linalg.norm(np.diff(h, axis=1), axis=-1), axis=1)
        labels = model.opportunity(cv, bank, y, valid, scale, queries, cut['easy'], cap)
        src = sorted(set(sites))
        m, cost = {}, {}
        for s in src:
            i = np.flatnonzero(sites == s)
            f, z = run.model.descriptors(h[i], bank[i], cv[i], crowd[i])
            w = 1./crowd[i]
            m[s] = model.moments(x[i], labels[i], w)
            cost[s] = run.model.fitting_moments(f, bank[i], cv[i], y[i], valid[i], z, w, cut['easy'])
        with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
            old_coeff = dict(zip(f['sources'], f['excluded_source_coefficients']))
        coefficients = {s: (model.coefficients([m[v] for v in src if s != v]),
                             run.model.coefficients([cost[v] for v in src if s != v])) for s in src}
        chosen = np.full((len(NAMES), len(h)), -1, dtype=np.int16)
        retrieved = np.zeros((2, len(h)), dtype=bool)
        cap_rows = np.zeros(len(h), dtype=np.int8)
        for q, k in zip(queries, cap):
            s = sites[q[0]]
            b, coeff = coefficients[s]
            f, z = run.model.descriptors(h[q], bank[q], cv[q], crowd[q])
            u, a, e, ok = run.model.decision_costs(f, coeff, bank[q], cv[q], z)
            p = np.einsum('nad,adc->nac', f, coeff)
            support = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*z[:, None]
            inc = cal.model.decide(h[q], old[q], cv[q], ols[q], old_coeff[s])
            if int((inc[3] >= 0).sum()) != k:
                raise ValueError('Original matched counts changed')
            chosen[-1, q] = inc[4]
            cap_rows[q] = k
            for j, pool in enumerate(model.lists(x[q]@b, ok)):
                retrieved[j, q[pool]] = True
                mask = ok & retrieved[j, q, None]
                choice = run.model.choose_query(h[q], cv[q], bank[q], ols[q], z, u, mask)[3]
                if int((choice >= 0).sum()) != k:
                    raise ValueError('Matched query count changed after retrieval')
                chosen[j, q] = choice
                chosen[j+2, q] = gate(choice, u, a, e, support)
        if not np.array_equal(retrieved.sum(1), np.repeat(retrieved.sum(1)[0], 2)):
            raise ValueError('Candidate retrieval budget changed')
        tmp = dest.with_suffix('.pending.npz')
        np.savez(tmp, ids=d['global_ids'], chosen=chosen, retrieved=retrieved, cap=cap_rows,
                 source_coefficients=np.stack([coefficients[s][0] for s in src]),
                 cost_coefficients=np.stack([coefficients[s][1] for s in src]))
        tmp.replace(dest)
        run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(dest)))
    row = run.read(mark)
    if row['seed'] != seed or row['registration'] != run.sha(REG):
        raise ValueError('Incompatible retrieval checkpoint')
    run.checked(row['artifact'])
    beat(seed=seed, phase_detail='source_excluded_shortlists_frozen')
    return row


def score(d, refs, y, valid, records, cut, freeze, beat):
    pairs = {f'{n}_vs_cv': (BASE, n, BASE) for n in NAMES}
    pairs.update({f'{n}_vs_ols4': (OLS, n, BASE) for n in NAMES})
    pairs.update(opportunity_vs_gain=('gain', 'opportunity', BASE),
                 gated_opportunity_vs_gain=('gated_gain', 'gated_opportunity', BASE))
    rows, roster, capacity = [], {}, []
    slots, changed = np.zeros(2, int), 0
    for cell in freeze:
        seed = cell['seed']
        cv, ols, bank = bank_for(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice, retrieved, cap = z['chosen'], z['retrieved'], z['cap']
        slots += retrieved.sum(1)
        changed += sum(int((retrieved[0, q] != retrieved[1, q]).any()) for q in d['groups'])
        for r, (ix, groups) in records.items():
            site = str(d['sites'][ix[0]])
            roster[r] = site
            pred = {k: run.model.selected_paths(v[ix], bank[ix], cv[ix]) for k, v in zip(NAMES, choice)}
            res = run.parent.readout.score_recording({BASE: cv[ix], OLS: ols[ix], **pred}, pairs,
                y[ix], valid[ix], d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(res, seed=seed, source=site))
            a = PRIVATE/f'capacity_{seed}_{r}.json'
            if not a.exists():
                loc = ix[np.concatenate(groups)]
                counts = np.array([len(g) for g in groups])
                offsets = np.r_[0, counts.cumsum()]
                lim = cap[loc[offsets[:-1]]]
                full = bank[loc]
                outcomes = [prior.model.solve(cv[loc], ols[loc], full, y[loc], valid[loc], offsets, lim,
                    np.broadcast_to(m[loc, None], full.shape[:2]), r, cut['easy'], cut['hard'])
                    for m in retrieved]
                run.once(a, run.stable(dict(registration=run.sha(REG), fit=cell['artifact']['sha256'],
                                            result=outcomes)))
            saved = run.read(a)
            if saved['registration'] != run.sha(REG) or saved['fit'] != cell['artifact']['sha256']:
                raise ValueError('Incompatible shortlist capacity checkpoint')
            capacity.append(dict(site=site, seed=seed, outcomes=saved['result']))
        beat(seed=seed, phase_detail='scored_capacity_and_operating_rule')
    result = grouped.grouped_readout(rows, roster, SEEDS)
    if result['risk']['gain']['switches'] != result['risk']['opportunity']['switches']:
        raise ValueError('Matched prediction counts changed')
    if slots[0] != slots[1]:
        raise ValueError('Recipient budgets differ')
    source = sorted(set(roster.values()))
    contrast = np.array([np.mean([[int(r['outcomes'][1][k])-int(r['outcomes'][0][k])
                                   for k in ('feasible', 'positive')]
                                  for r in capacity if r['site'] == s], axis=0)
                         for s in source])*100.
    draw = np.random.default_rng(73107).integers(len(source), size=(3000, len(source)))
    bound = np.quantile(contrast[draw].mean(1), [.025, .975], axis=0).T
    caps = {}
    for i, k in enumerate(NAMES[:2]):
        cells = [r['outcomes'][i] for r in capacity]
        caps[k] = dict(feasible=sum(c['feasible'] for c in cells), positive=sum(c['positive'] for c in cells),
            unresolved=sum(not (c['optimum_proved'] or c['proved_infeasible']) for c in cells))
    return dict(grouped=result, capacity=caps, capacity_contrast={n: dict(point=float(contrast[:, i].mean()),
        interval95=bound[i].tolist()) for i, n in enumerate(('feasible_pp', 'positive_pp'))},
        selected_slots=slots.tolist(), changed_seed_queries=changed, independent_confirmation=False,
        external_access=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = parser.parse_args().phase
    cut, ctrl = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit retrieval design before executing it')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        freeze = PRIVATE/'freeze.json'
        if phase == 'fit':
            run.once(freeze, run.stable(dict(cells=[fit(d, refs, y, valid, cut, c, beat) for c in ctrl['cells']])))
            out = freeze
        else:
            run.checked(run.read(PUBLIC/'fit_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            out = PUBLIC/'analysis.json'
            records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
            run.once(out, run.stable(score(d, refs, y, valid, records, cut, run.read(freeze)['cells'], beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Retrieval receipt differs')


if __name__ == '__main__':
    main()
