"""Train past-only recipient ranking against verified recording-level witnesses."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_full_query_retrieval as prior
from src.world_model import m3w_recording_witness_retrieval as model

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC.parent/'native_recording_witness_retrieval_v1'
PRIVATE = prior.PRIVATE.parent/'native_recording_witness_retrieval_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_recording_witness_retrieval.py')
NAMES = ('gain', 'witness', 'gated_gain', 'gated_witness', 'gated_original')
BASE, OLS = prior.BASE, prior.OLS
SEEDS = prior.SEEDS


def register():
    cut, _ = prior.register()
    parents = (prior.REG, prior.PUBLIC/'analysis.json', prior.PUBLIC/'fit_execution.json',
               prior.PUBLIC/'replay_execution.json', *FILES)
    for k in ('fit', 'replay'):
        run.checked(run.read(prior.PUBLIC/f'{k}_execution.json')['artifact'])
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut, seeds=SEEDS,
        primary='fixed_shortlist_positive_ols4_capacity', known_only_teacher=True, teacher_seconds=6.,
        source_excluded=True, capacity_seconds=4., scope='exposed_development_only',
        original_query_caps_and_recording_rule=True, external_access=False)))
    return cut, run.read(prior.PRIVATE/'freeze.json')['cells']


def fit_labels(d, records, cv, ols, bank, cap, y, valid, cut, seed, beat):
    labels = np.zeros(len(cv))
    teachers = []
    for r, (ix, groups) in records.items():
        p = PRIVATE/f'teacher_{seed}_{r}.json'
        loc = ix[np.concatenate(groups)]
        if not p.exists():
            off = np.r_[0, np.cumsum(list(map(len, groups)))]
            t, info = model.teacher(cv[loc], ols[loc], bank[loc], y[loc], valid[loc], off,
                                    cap[loc[off[:-1]]], r, cut)
            run.once(p, run.stable(dict(registration=run.sha(REG), count=len(loc),
                                       selected=np.flatnonzero(t).tolist(), info=info)))
        z = run.read(p)
        if z['registration'] != run.sha(REG) or z['count'] != len(loc):
            raise ValueError('Stale teacher labels')
        labels[loc[np.asarray(z['selected'], int)]] = 1.
        teachers.append(z['info'])
    beat(seed=seed, phase_detail='teacher_labels_frozen', positive=sum(t['positive'] for t in teachers))
    return labels, teachers


def decisions(d, cv, ols, bank, x, cap, old, labels):
    # This fit uses each label only via the source where it was generated.
    h, sites, crowd = (d[k] for k in ('history', 'sites', 'crowd'))
    sources = sorted(set(sites))
    moments = {}
    for s in sources:
        at = sites == s
        moments[s] = prior.model.moments(x[at], labels[at, None], 1./crowd[at])
    coef = {s: prior.model.coefficients([moments[v] for v in sources if v != s])[:, 0] for s in sources}
    c = dict(zip(sources, old['cost_coefficients']))
    gain = dict(zip(sources, old['source_coefficients'][:, :, 0]))
    choice = np.full((len(NAMES), len(cv)), -1, dtype=np.int16)
    choice[[0, 2, 4]] = old['chosen'][[0, 2, 4]]
    masks = np.stack((old['retrieved'][0], np.zeros(len(cv), bool)))
    for q in d['groups']:
        s = sites[q[0]]
        f, z = run.model.descriptors(h[q], bank[q], cv[q], crowd[q])
        u, a, e, ok = run.model.decision_costs(f, c[s], bank[q], cv[q], z)
        p = np.einsum('nad,adc->nac', f, c[s])
        ref = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*z[:, None]
        first, second = prior.model.lists(np.column_stack((x[q]@gain[s], x[q]@coef[s])), ok)
        if not np.array_equal(np.sort(first), np.flatnonzero(masks[0, q])):
            raise ValueError('Frozen gain control differs')
        masks[1, q[second]] = True
        res = run.model.choose_query(h[q], cv[q], bank[q], ols[q], z, u, ok & masks[1, q, None])[3]
        if int((res >= 0).sum()) != cap[q[0]]:
            raise ValueError('Proposed count differs from original')
        choice[1, q] = res
        choice[3, q] = prior.gate(res, u, a, e, ref)
    return choice, masks, np.stack([coef[s] for s in sources])


def fit(d, refs, y, valid, records, cut, cell, beat):
    seed = cell['seed']
    p = PRIVATE/f'seed_{seed}.json'
    if not p.exists():
        cv, ols, bank = prior.bank_for(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            old = {k: f[k] for k in f.files}
        cap = old['cap']
        t, info = fit_labels(d, records, cv, ols, bank, cap, y, valid, cut, seed, beat)
        x = prior.model.features(d['history'], bank, cv, ols, d['groups'], d['crowd'])
        choice, masks, coef = decisions(d, cv, ols, bank, x, cap, old, t)
        dest, tmp = p.with_suffix('.npz'), p.with_suffix('.pending.npz')
        np.savez(tmp, ids=d['global_ids'], chosen=choice, retrieved=masks, cap=cap, coefficients=coef)
        tmp.replace(dest)
        run.once(p, run.stable(dict(registration=run.sha(REG), seed=seed, artifact=run.artifact(dest),
            teachers=info, positives=int(t.sum()))))
    out = run.read(p)
    if out['registration'] != run.sha(REG) or out['seed'] != seed:
        raise ValueError('Stale fit checkpoint')
    run.checked(out['artifact'])
    beat(seed=seed, phase_detail='source_excluded_decisions_frozen')
    return out


def score(d, refs, y, valid, records, cut, freeze, beat):
    roster, rows, capacity, slots = {}, [], [], np.zeros(2, int)
    pairs = {f'{n}_vs_cv': (BASE, n, BASE) for n in NAMES}
    pairs.update({f'{n}_vs_ols4': (OLS, n, BASE) for n in NAMES})
    pairs.update(witness_vs_gain=('gain', 'witness', BASE),
                 gated_witness_vs_gain=('gated_gain', 'gated_witness', BASE))
    for cell in freeze:
        seed = cell['seed']
        cv, ols, bank = prior.bank_for(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            choice, masks, cap = f['chosen'], f['retrieved'], f['cap']
        slots += masks.sum(1)
        for r, (ix, groups) in records.items():
            site = str(d['sites'][ix[0]])
            roster[r] = site
            values = {k: run.model.selected_paths(a[ix], bank[ix], cv[ix]) for k, a in zip(NAMES, choice)}
            res = run.parent.readout.score_recording({BASE: cv[ix], OLS: ols[ix], **values}, pairs,
                y[ix], valid[ix], d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(res, seed=seed, source=site))
            old = run.read(prior.PRIVATE/f'capacity_{seed}_{r}.json')
            if old['registration'] != run.sha(prior.REG):
                raise ValueError('Stale gain capacity comparator')
            prev = old['result'][0]
            p = PRIVATE/f'capacity_{seed}_{r}.json'
            if not p.exists():
                loc = ix[np.concatenate(groups)]
                off = np.r_[0, np.cumsum(list(map(len, groups)))]
                allowed = np.broadcast_to(masks[1, loc, None], bank[loc].shape[:2])
                new = prior.prior.model.solve(cv[loc], ols[loc], bank[loc], y[loc], valid[loc],
                    off, cap[loc[off[:-1]]], allowed, r, cut['easy'], cut['hard'])
                run.once(p, run.stable(dict(registration=run.sha(REG), result=new, fit=cell['artifact']['sha256'])))
            saved = run.read(p)
            if saved['registration'] != run.sha(REG) or saved['fit'] != cell['artifact']['sha256']:
                raise ValueError('Stale witness capacity')
            capacity.append((site, prev, saved['result']))
        beat(seed=seed, phase_detail='readout_complete')
    out = prior.grouped.grouped_readout(rows, roster, SEEDS)
    old = run.read(prior.PUBLIC/'analysis.json')
    for k in ('gain', 'gated_gain', 'gated_original'):
        if out['risk'][k] != old['grouped']['risk'][k]:
            raise ValueError('Frozen control score did not replay')
    if slots[0] != slots[1] or out['risk']['gain']['switches'] != out['risk']['witness']['switches']:
        raise ValueError('Proposal count is not matched')
    if len(capacity) != 102:
        raise ValueError('Require 102 recording-seed views')
    keys = ('feasible', 'positive')
    src = sorted(set(roster.values()))
    delta = np.array([np.mean([[int(b[k])-int(a[k]) for k in keys] for s0, a, b in capacity
                              if s0 == s], axis=0) for s in src])*100.
    draw = np.random.default_rng(73107).integers(len(src), size=(3000, len(src)))
    ci = np.quantile(delta[draw].mean(1), [.025, .975], axis=0).T
    cap_table = {}
    for j, k in enumerate(NAMES[:2], start=1):
        cells = [r[j] for r in capacity]
        cap_table[k] = dict(**{v: sum(c[v] for c in cells) for v in keys},
            unresolved=sum(not (c['optimum_proved'] or c['proved_infeasible']) for c in cells))
    from collections import Counter
    info = [t for c in freeze for t in c['teachers']]
    return dict(grouped=out, slots=slots.tolist(), capacity=cap_table,
        contrasts={k: dict(point_pp=float(delta[:, i].mean()), interval95_pp=ci[i].tolist(),
            gained=sum(not a[k] and b[k] for _, a, b in capacity),
            lost=sum(a[k] and not b[k] for _, a, b in capacity)) for i, k in enumerate(keys)},
        teacher=dict(positive_views=sum(t['positive'] for t in info),
            selected_rows=sum(c['positives'] for c in freeze), statuses=dict(Counter(t['status'] for t in info))),
        external_data_opened=False, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = parser.parse_args().phase
    cut, cells = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit witness protocol before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
        freeze = PRIVATE/'freeze.json'
        if phase == 'fit':
            run.once(freeze, run.stable(dict(cells=[fit(d, refs, y, valid, records, cut, c, beat) for c in cells])))
            out = freeze
        else:
            run.checked(run.read(PUBLIC/'fit_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, y, valid, records, cut, run.read(freeze)['cells'], beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs from original witness receipt')


if __name__ == '__main__':
    main()
