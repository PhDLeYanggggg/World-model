"""Fixed-cap development oracle ablation of the three learned forecast families."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_easy_support_certificate as support
from scripts import audit_m3w_recipient_budget_capacity as prior
from scripts import refine_m3w_recipient_budget_capacity as refine

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC.parent/'native_neural_bank_ablation_v1'
PRIVATE = prior.PRIVATE.parent/'native_neural_bank_ablation_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_neural_bank_capacity.py')
ARMS = ('causal_only', 'native_with_learned', 'prior_expanded')
LEARNED = ('eqmotion', 'pct', 'self')
CAUSAL = tuple(k for k in run.parent.BASELINES if k != run.parent.CV)
SEEDS = prior.g.prior.SEEDS
LIMITS = (6., 25.)


def register():
    cut, controller = prior.register()
    for name in ('execution', 'refinement_execution'):
        run.checked(run.read(prior.PUBLIC/f'{name}.json')['artifact'])
    if set(run.NAMES) != set(CAUSAL+LEARNED) or len(CAUSAL) != 5:
        raise ValueError('Unexpected forecast family roster')
    parents = (prior.REG, refine.REG, prior.PUBLIC/'execution.json', prior.PUBLIC/'refinement_execution.json',
               prior.PUBLIC/'analysis.json', prior.PUBLIC/'refinement.json',
               Path(prior.__file__), Path(refine.__file__), Path(prior.model.__file__),
               Path(support.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], roles='exposed_moving_A_only',
        arms=ARMS, causal=CAUSAL, learned=LEARNED, seeds=SEEDS, cuts=cut, solver_seconds=LIMITS,
        bootstrap_draws=3000, risk=.02, external_access=False)))
    return cut, controller


def completed(result):
    return result['optimum_proved'] or result['proved_infeasible']


def positive_state(result):
    if result['feasible'] and result['positive']:
        return True
    upper = result.get('upper_mass')
    if completed(result) or (upper is not None and upper <= 0.):
        return False
    return None


def old_cell(seed, recording):
    original = run.read(prior.PRIVATE/f'{seed}_{recording}.json')
    if original['registration'] != run.sha(prior.REG):
        raise ValueError('Initial comparator registration differs')
    previous = original['outcomes']['all_recipients']
    if not completed(previous):
        item = run.read(prior.PRIVATE/'refinement'/f'{seed}_{recording}.json')
        if item['registration'] != run.sha(refine.REG):
            raise ValueError('Refined comparator registration differs')
        result = item['result']
        if not completed(result):
            raise ValueError('Prior expanded comparator is unresolved')
        original['outcomes']['all_recipients'] = combine(
            result, previous, original['outcomes']['same_recipients'])
    return original


def combine(result, *previous):
    """Retain the best checked witness and tightest valid upper bound."""
    bounds = [v['upper_mass'] for v in (result, *previous) if v.get('upper_mass') is not None]
    witnesses = [v for v in (result, *previous) if v['feasible']]
    out = dict(result, upper_mass=min(bounds, default=None))
    if witnesses:
        witness = max(witnesses, key=lambda x: x['witness_mass'])
        if result['proved_infeasible'] or (out['upper_mass'] is not None
                                          and witness['witness_mass'] > out['upper_mass']+1e-5):
            raise ValueError('Solver certificate contradicts a checked witness')
        out.update({k: witness[k] for k in ('feasible', 'positive', 'witness_mass')})
    return out


def ensure_nested(smaller, larger):
    if (smaller['feasible'] and larger['proved_infeasible']) or (
            smaller['feasible'] and larger.get('upper_mass') is not None
            and smaller['witness_mass'] > larger['upper_mass']+1e-5):
        raise ValueError('Nested slate lost a known feasible option')


def assess(cv, ols, bank, y, valid, offsets, caps, recording, cut, beat):
    ok = support.assess(cv, bank, y, valid, np.repeat(caps > 0, np.diff(offsets)), cut['easy'])
    allow = refine.prune(cv, ols, bank, valid)
    args = (cv, ols, bank, y, valid, offsets, caps, allow, recording, cut['easy'], cut['hard'])
    result = prior.model.solve(*args, seconds=LIMITS[0])
    refined = not completed(result)
    if refined:
        beat(state='refining_capacity')
        result = combine(prior.model.solve(*args, seconds=LIMITS[1]), result)
    if (result['feasible'] and not ok['feasible']) or (result['proved_infeasible'] and ok['feasible']):
        raise ValueError('Exact support condition and optimization disagree')
    return dict(result, support_feasible=ok['feasible'], refined=refined)


def summarize(rows):
    def arm(name):
        a = [r['arms'][name] for r in rows]
        states = [positive_state(v) for v in a]
        return dict(risk_feasible=sum(v['support_feasible'] for v in a),
                    positive=sum(s is True for s in states), nonpositive=sum(s is False for s in states),
                    positivity_unresolved=sum(s is None for s in states),
                    objective_unresolved=sum(not completed(v) for v in a),
                    refined=sum(v.get('refined', False) for v in a))

    def pair(before, after):
        proven, groups = [], {}
        all_known = True
        improvements = []
        for row in rows:
            a, b = (row['arms'][k] for k in (before, after))
            sa, sb = positive_state(a), positive_state(b)
            if sa is True and sb is False:
                raise ValueError('Nested bank lost positive capacity')
            if sa is False and sb is True:
                proven.append(row)
            all_known &= sa is not None and sb is not None
            groups.setdefault(row['source'], {}).setdefault(row['seed'], []).append(
                int(sb is True)-int(sa is True))
            if all(v['optimum_proved'] and v['feasible'] for v in (a, b)):
                improvements.append(100*(b['witness_mass']-a['witness_mass'])/a['reference_mass'])
        interval = None
        if all_known:
            values = np.asarray([np.mean([np.mean(v) for v in cells.values()])
                                 for _, cells in sorted(groups.items())])
            draw = np.random.default_rng(73107).integers(len(values), size=(3000, len(values)))
            interval = dict(point=float(100*values.mean()),
                            ci95=(100*np.quantile(values[draw].mean(1), [.025, .975])).tolist())
        return dict(new_positive_views=len(proven),
                    new_positive_recordings=len({r['recording'] for r in proven}),
                    new_positive_sources=len({r['source'] for r in proven}),
                    source_balanced_positive_rate_change_pp=interval,
                    comparable_optima=len(improvements),
                    median_optimal_lower_gain_delta_pp=float(np.median(improvements)) if improvements else None)

    return dict(arms={a: arm(a) for a in ARMS},
                comparisons={f'{b}_vs_{a}': pair(a, b) for a, b in zip(ARMS, ARMS[1:])})


def score(cut, controller, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, original = [], []
    causal_cols = [run.NAMES.index(k) for k in CAUSAL]
    registration = run.sha(REG)
    for cell in controller['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as a:
            coef = dict(zip(a['sources'], a['excluded_source_coefficients']))
        for rec, (ix, queries) in records.items():
            old = old_cell(seed, rec)
            original.append(old)
            src = str(d['sites'][ix[0]])
            path = PRIVATE/f'{seed}_{rec}.json'
            if not path.exists():
                loc = ix[np.concatenate(queries)]
                offsets = np.r_[0, np.cumsum(list(map(len, queries)))]
                paths, caps, _ = prior.query_design(d, loc, offsets, cv[loc], bank[loc], coef[src])
                if (src != old['source'] or len(loc) != old['rows'] or len(caps) != old['query_count']
                        or (caps > 0).sum() != old['nonzero_cap_queries']):
                    raise ValueError('Prior comparator roster differs')
                ols = paths[:, run.NAMES.index(cal.OLS)]
                full = old['outcomes']['all_recipients']
                arms = {}
                for k, slate in zip(ARMS[:2], (paths[:, causal_cols], paths[:, :-1])):
                    beat(seed=seed, arm=k, completed=len(cells), state='solving_capacity')
                    arms[k] = assess(cv[loc], ols, slate, y[loc], valid[loc], offsets, caps, rec, cut, beat)
                    if arms[k]['reference_mass'] != full['reference_mass']:
                        raise ValueError('The common CV denominator changed')
                arms[ARMS[2]] = dict(full, support_feasible=full['feasible'])
                for a, b in zip(ARMS, ARMS[1:]):
                    ensure_nested(arms[a], arms[b])
                run.once(path, run.stable(dict(registration=registration, seed=seed,
                    recording=str(rec), source=src, arms=arms)))
            result = run.read(path)
            if (result['registration'], result['seed'], result['recording'], result['source']) != (
                    registration, seed, str(rec), src):
                raise ValueError('Incompatible recording checkpoint')
            cells.append(result)
            beat(seed=seed, completed=len(cells), state='recording_capacity_checked')
    if len(cells) != 102 or prior.summarize(original) != run.read(prior.PUBLIC/'refinement.json')['summary']:
        raise ValueError('Prior comparator did not replay on the same roster')
    return summarize(cells)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, controller = register()
    if phase == 'register':
        return
    for p in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{p.relative_to(ROOT)}'], cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit experiment design before scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'replay':
            run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])

        def beat(**v):
            doc = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', doc)
            if v.get('completed', -1) % 10 == 0 or v.get('completed') == 102:
                print(doc, flush=True)

        result = score(cut, controller, beat)
        out = PUBLIC/'analysis.json'
        run.once(out, run.stable(dict(summary=result, registration=run.sha(REG),
            result_source='fresh_fixed_cap_capacity_exposed_development',
            independent_confirmation=False, external_access=False, deployment=False,
            stage5c_executed=False, smc_enabled=False)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(artifact=run.artifact(out), pid=os.getpid(), seconds=time.monotonic()-start))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Capacity aggregate changed')


if __name__ == '__main__':
    main()
