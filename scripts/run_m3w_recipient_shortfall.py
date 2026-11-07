"""Test recipient choice at the same query budgets and frozen path nominations."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_causal_menu_control as old
from scripts import audit_m3w_fixed_recipient_risk_bound as bound
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments

run, np, pair = old.run, old.np, old.pair
BASE, OLS, SEEDS = old.BASE, old.OLS, old.SEEDS
PUBLIC = old.PUBLIC.parent/'native_recipient_shortfall_v1'
PRIVATE = old.PRIVATE.parent/'native_recipient_shortfall_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_recipient_shortfall.py')
MATCHED = ('gain', 'support_first', 'shortfall')
ARMS = (*MATCHED, *(a+'_screen' for a in MATCHED))
FIELDS = ('all_excess', 'easy_excess', 'easy_signed')


def receipt(folder, key):
    return run.read(run.checked(run.read(folder/f'{key}_execution.json')['artifact']))


def register():
    cut = old.register()
    for folder in (old.PUBLIC, bound.PUBLIC):
        receipt(folder, 'replay')
    parents = (old.REG, old.PUBLIC/'replay_execution.json', bound.REG, bound.PUBLIC/'replay_execution.json',
               Path(run.model.__file__), Path(old.prior.model.__file__), Path(moments.__code__.co_filename), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(x) for x in parents],
        cut=cut, arms=ARMS, seeds=SEEDS, risk=.02, primary='shortfall_vs_support_first',
        equal_query_counts=True, identical_path_nomination=True, external_access=False)))
    return cut


def recipient_rules(gain, all_cost, easy_cost, ref, eligible, k):
    at = np.flatnonzero(eligible)
    if not 0 <= k <= min(2, len(at)):
        raise ValueError('Unsupported original switch count')
    if not k:
        return (at[:0],)*3
    ordered = at[np.argsort(-gain[at], kind='stable')]
    first = ordered[:k]
    leader = at[np.lexsort((-gain[at], -ref[at]))[0]]
    second = np.r_[leader, ordered[ordered != leader][:k-1]] if ref[leader] > 0. else first
    if k == 1:
        proposals = at[:, None]
    else:
        i, j = np.triu_indices(len(at), k=1)
        proposals = np.column_stack((at[i], at[j]))
    unsupported = (ref[proposals].sum(1) <= 0)
    shortfall = np.maximum(0., np.maximum(all_cost[proposals].sum(1), easy_cost[proposals].sum(1)))
    best = np.lexsort((-gain[proposals].sum(1), shortfall, unsupported))[0]
    return first, second, proposals[best]


def query_choices(h, bank, cv, crowd, coef):
    x, scale = run.model.descriptors(h, bank, cv, crowd)
    u, all_cost, easy_cost, legal = run.model.decision_costs(x, coef, bank, cv, scale)
    score = np.where(legal, u/scale[:, None], -np.inf)
    nominee = score.argmax(1)
    at = np.arange(len(cv)), nominee
    k = min(2, len(cv)-1, int(legal.sum(0).max()))
    p = np.einsum('nad,adc->nac', x, coef)
    ref = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*scale[:, None]
    proposed = np.full((len(MATCHED), len(cv)), -1, dtype=np.int16)
    for dest, chosen in zip(proposed, recipient_rules(score[at], all_cost[at], easy_cost[at],
                                                     ref[at], legal[at], k)):
        dest[chosen] = nominee[chosen]
    guard = [old.prior.model.gate(c, u, all_cost, easy_cost, ref) for c in proposed]
    return np.vstack((proposed, guard)), np.where(legal[at], nominee, -1)


def freeze(d, refs, beat):
    parent = receipt(old.prior.PUBLIC, 'freeze')
    old_choice = receipt(old.PUBLIC, 'freeze')
    cells = []
    for cell, check in zip(parent['cells'], old_choice['cells'], strict=True):
        seed = cell['seed']
        if seed != check['seed'] or check['mismatched_queries']:
            raise ValueError('Prior exact-count control unavailable')
        path = PRIVATE/f'choices_{seed}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            cv, bank = run.candidates(d, refs, seed)
            with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
                coefficients = dict(zip(f['sources'], f['excluded_source_coefficients']))
            choice = np.full((len(ARMS), len(cv)), -1, dtype=np.int16)
            nominee = np.full(len(cv), -1, dtype=np.int16)
            for at in d['groups']:
                choice[:, at], nominee[at] = query_choices(d['history'][at], bank[at], cv[at], d['crowd'][at],
                                                            coefficients[d['sites'][at[0]]])
            with np.load(run.checked(check['artifact']), allow_pickle=False) as f:
                np.testing.assert_array_equal(f['ids'], d['global_ids'])
                np.testing.assert_array_equal(choice[0], f['choices'][old.ARMS.index('mixed_independent')])
            if np.any((choice >= 0) & (choice != nominee[None])):
                raise ValueError('A recipient rule changed its nominated path')
            tmp = path.with_suffix('.pending.npz')
            np.savez(tmp, ids=d['global_ids'], choice=choice, nominee=nominee)
            tmp.replace(path)
            run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(path)))
        item = run.read(mark)
        if item['registration'] != run.sha(REG) or item['seed'] != seed:
            raise ValueError('Incompatible recipient checkpoint')
        run.checked(item['artifact'])
        cells.append(item)
        beat(seed=seed, state='past_only_recipients_frozen')
    dest = PRIVATE/'freeze.json'
    run.once(dest, run.stable(dict(cells=cells, registration=run.sha(REG), score_labels_read=False)))
    return dest


def capacity(cv, bank, nominee, selected, queries, y, valid, easy_cut):
    """Separate necessary-condition minima for the exact same query budgets."""
    n = len(cv)
    ref, _ = run.model.errors(cv, y, valid)
    easy = valid.any(1) & (ref > 0.) & (ref <= easy_cut)
    full_min = np.full((n, 3), np.inf)
    fixed = np.full((n, 3), np.inf)
    for j in range(bank.shape[1]):
        m = moments(cv, bank[:, j], y, valid, easy_cut)
        v = np.column_stack([m[k] for k in FIELDS])
        full_min = np.minimum(full_min, np.where((m['radius'] > 0.)[:, None], v, np.inf))
        fixed[nominee == j] = v[nominee == j]
    out = []
    for terms in (fixed, full_min):
        total, support = np.zeros(3), False
        for g in queries:
            k = int(selected[g].sum())
            if k:
                total += np.sort(terms[g], axis=0)[:k].sum(0)
                support |= bool((easy[g] & np.isfinite(terms[g, 0])).any())
        if not np.isfinite(total).all():
            raise ValueError('Exact query budget is unattainable despite a verified witness')
        total[2] -= BUDGET*ref[easy].sum()
        tol = 1e-8*max(1., float(ref[valid.any(1)].sum()))
        item = {f'positive_minimum_{k}': bool(v > tol) for k, v in zip(FIELDS, total)}
        item['missing_possible_easy_reference'] = not support
        item['provably_infeasible'] = any(item.values())
        out.append(item)
    if out[1]['provably_infeasible'] and not out[0]['provably_infeasible']:
        raise ValueError('Full path expansion cannot lose risk capacity')
    return out


def score(d, refs, lab, frozen, cut, beat):
    records = old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    sources = sorted(set(roster.values()))
    labels, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    pairs = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    pairs.update({f'{a}_vs_{b}': (b, a, BASE) for a, b in (
        ('support_first', 'gain'), ('shortfall', 'gain'), ('shortfall', 'support_first'),
        ('shortfall_screen', 'gain_screen'), ('shortfall_screen', 'support_first_screen'))})
    rows, bounds, rel = [], [], {}
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice, nominee = z['choice'], z['nominee']
        pair_seed = {s: [] for s in sources}
        for r, (ix, groups) in records.items():
            local = {BASE: cv[ix], OLS: bank[ix, run.NAMES.index(OLS)],
                     **{a: run.model.selected_paths(c[ix], bank[ix], cv[ix]) for a, c in zip(ARMS, choice)}}
            changed = np.stack([(local[a] != cv[ix]).any(axis=(1, 2)) for a in MATCHED])
            for g in groups:
                n = changed[:, g].sum(1)
                if not np.all(n == n[0]):
                    raise ValueError('Real query counts changed')
                p = np.stack([local[a][g] for a in ('support_first', 'shortfall')])
                _, a, _, _ = pair.query(d['history'][ix[g]], p, labels[ix[g]], valid[ix[g]], [(0, 1)])
                pair_seed[roster[r]].append(a[0])
            row = run.parent.readout.score_recording(local, pairs, labels[ix], valid[ix], d['queries'][ix],
                                                      r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            proofs = capacity(cv[ix], bank[ix], nominee[ix], changed[0], groups, labels[ix], valid[ix], cut['easy'])
            if proofs[0]['provably_infeasible'] and any(row['metrics'][a]['risk']['supported'] for a in MATCHED):
                raise ValueError('Infeasibility bound contradicts a realized pass')
            bounds.append(proofs)
            rows.append(dict(row, seed=seed, source=roster[r]))
        for s, a in pair_seed.items():
            rel[s, seed] = pair.mean_finite(a)
        beat(seed=seed, state='choices_scored_and_capacity_checked')
    agg = old.prior.grouped_readout(rows, roster, SEEDS)
    if agg['risk']['gain'] != run.read(old.PUBLIC/'analysis.json')['grouped']['risk']['mixed_independent']:
        raise ValueError('Matched gain control did not replay')
    draw = np.random.default_rng(73107).integers(len(sources), size=(3000, len(sources)))
    z = np.asarray([[rel[s, seed] for seed in SEEDS] for s in sources])
    return dict(grouped=agg, component_bound={key: {k: sum(r[i][k] for r in bounds) for k in bounds[0][i]}
            for i, key in enumerate(('fixed_nomination', 'full_forecast_bank'))},
        relative_pair={k: pair.bootstrap(z[:, :, i], draw) for i, k in enumerate(pair.PAIR_FIELDS)},
        external_access=False, independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = parser.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the design before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            state = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', state)
            print(state, flush=True)

        d, refs = run.population(), run.register()
        if phase == 'freeze':
            out = freeze(d, refs, beat)
        else:
            frozen = receipt(PUBLIC, 'freeze')
            if frozen['registration'] != run.sha(REG) or frozen['score_labels_read']:
                raise ValueError('Require pre-score choice freeze')
            if phase == 'replay':
                receipt(PUBLIC, 'score')
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, frozen, cut, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Exact replay changed')


if __name__ == '__main__':
    main()
