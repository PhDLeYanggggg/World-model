"""Evaluate the sparse replacement envelope on the exposed development roster."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_quarter_allocation as parent
from src.evaluation import m3w_quarter_swap_capacity as model
from src.evaluation.m3w_recording_forecast_comparison import score_recording

h, run, np = parent.h, parent.run, parent.np
PUBLIC = parent.PUBLIC.parent/'moving_quarter_swap_capacity_v1'
PRIVATE = run.PRIVATE/'quarter_swap_capacity'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_quarter_swap_capacity.py')


def register():
    a, replay = (parent.PUBLIC/name for name in ('analysis.json', 'replay_execution.json'))
    if run.read(replay)['artifact']['sha256'] != run.sha(a) or run.read(a)['external_labels_opened']:
        raise ValueError('Require exact, exposed development parent')
    paths = (a, replay, parent.REG, Path(parent.__file__), Path(model.moments.__code__.co_filename))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], caps=model.CAPS, time_limit=model.SECONDS,
        amplitude=.25, original_risk_budget=.02, fixed_query_counts=True,
        primary='new_no_loss_repairs_at_cap_4', external_outcomes_opened=False)))


def inspect(m, rec, ix, groups, y, valid, cuts, phase):
    d, cv = m.d, run.parent.CV
    site = str(d['sites'][ix[0]])
    c, b, a = m.floor[ix], m.bank[ix], m.direct[ix]
    easy, hard = cuts[site]['easy'], cuts[site]['hard']
    p = model.prepare(c, b, a, groups, y[ix], valid[ix], easy)
    path = PRIVATE/f'{m.seed}_{rec}.json'
    identity = dict(seed=m.seed, recording=rec, source=site, registration=run.sha(REG))
    if phase == 'score' and not path.exists():
        solutions = {}
        for k in model.CAPS:
            take, status, bound = model.solve(p, k)
            solutions[str(k)] = dict(pick=None if take is None else take.tolist(),
                                      status=status, bound=bound)
        run.once(path, run.stable(dict(identity=identity, solutions=solutions)))
    saved = run.read(path)
    if saved['identity'] != identity:
        raise ValueError('Incompatible capacity witness receipt')
    original = score_recording({cv: c, 'quarter': p['before']}, {}, y[ix], valid[ix],
                                d['queries'][ix], rec, easy_cut=easy, hard_cut=hard)
    risk = original['metrics']['quarter']['risk']
    if bool(model.passes(p['base'], p['limits'])) != risk['supported']:
        raise ValueError('Original risk constraints do not match scorer')
    result = {}
    previous_gain = None
    for k in model.CAPS:
        s = saved['solutions'][str(k)]
        r = dict(status=s['status'], upper_mass=s['bound'])
        if s['pick'] is not None:
            take = np.asarray(s['pick'], dtype=int)
            if len(take) > k or len(np.unique(p['query'][take])) != len(take):
                raise ValueError('Invalid replacement witness')
            pred = model.construct(p, take)
            active = (pred != c).any((1, 2))
            if any(active[q].sum() != (a[q] >= 0).sum() for q in groups):
                raise ValueError('Per-query switch count differs')
            out = score_recording({cv: c, 'quarter': p['before'], 'witness': pred},
                {'gain': ('quarter', 'witness', cv)}, y[ix], valid[ix], d['queries'][ix], rec,
                easy_cut=easy, hard_cut=hard)
            gain = out['comparisons']['gain']
            lower = gain['known_delta']-gain['unknown_radius']
            if not np.isclose(lower, p['gains'][take].sum(), rtol=1e-8, atol=1e-8):
                raise ValueError('Witness utility disagrees with original score')
            if not out['metrics']['witness']['risk']['supported'] or lower < -1e-8:
                r['status'] = 'unresolved_scorer_postcheck'
            else:
                r.update(swaps=len(take), gain_lower_pp=100.*lower/gain['reference'])
                if previous_gain is not None and lower+1e-8 < previous_gain and r['status'] == 'optimal':
                    raise ValueError('An optimal larger budget cannot underperform a valid smaller witness')
                previous_gain = max(lower, previous_gain if previous_gain is not None else 0.)
        result[str(k)] = r
    return dict(source=site, seed=m.seed, recording=rec, original_pass=risk['supported'],
        original_easy_undefined=risk['completion_easy_risk_upper'] is None,
        zero_switches=risk['selected'] == 0, switches=risk['selected'],
        potential_swaps=len(p['gains']), caps=result)


def summary(rows):
    orig = run.read(parent.PUBLIC/'analysis.json')['grouped']['risk']['quarter']
    if (len(rows), sum(r['original_pass'] for r in rows), sum(r['original_easy_undefined'] for r in rows),
            sum(r['switches'] for r in rows)) != (102, orig['supported_views'],
            orig['undefined_easy_views'], orig['switches']):
        raise ValueError('Must replay the unchanged quarter control')
    caps = {}
    for k in model.CAPS:
        key = str(k)
        wins = [r for r in rows if 'gain_lower_pp' in r['caps'][key]]
        new = [r for r in wins if not r['original_pass']]
        gain = [r['caps'][key]['gain_lower_pp'] for r in new]
        caps[key] = dict(status=dict(Counter(r['caps'][key]['status'] for r in rows)),
            verified_witness_views=len(wins), newly_passing_no_loss=len(new),
            newly_passing_missing_easy=sum(r['original_easy_undefined'] for r in new),
            newly_passing_strictly_positive=sum(x > 1e-8 for x in gain),
            source_groups_with_new_witness=len({r['source'] for r in new}),
            new_witness_median_gain_lower_pp=float(np.median(gain)) if gain else None,
            new_witness_swaps=dict(Counter(str(r['caps'][key]['swaps']) for r in new)))
    return dict(caps=caps, views=len(rows), source_groups=len({r['source'] for r in rows}),
        original_passes=orig['supported_views'], original_undefined_easy=orig['undefined_easy_views'],
        zero_switch_views=sum(r['zero_switches'] for r in rows), switches=orig['switches'],
        total_candidate_swaps=sum(r['potential_swaps'] for r in rows), label_aware_diagnostic_only=True,
        original_rules_unchanged=True, external_outcomes_opened=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for p in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))],
                                   cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit capacity protocol before scoring')
    if phase == 'replay' and not (PUBLIC/'score_execution.json').exists():
        raise ValueError('Run score before witness replay')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = h.previous.prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        cells, controls, ranks = h.cost.banks()
        cuts, rows = run.cuts_by_source(), []
        for cell in cells:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            for rec, (ix, groups) in records.items():
                rows.append(inspect(m, rec, ix, groups, y, valid, cuts, phase))
                if len(rows) % 6 == 0:
                    print(dict(phase=phase, views=len(rows), seconds=time.monotonic()-start), flush=True)
            del m
        out = PUBLIC/'analysis.json'
        run.once(out, run.stable(summary(rows)))
        receipt = dict(analysis_sha256=run.sha(out), views=len(rows), seconds=time.monotonic()-start,
                       phase=phase, registration_sha256=run.sha(REG))
        dest = PUBLIC/f'{phase}_execution.json'
        if dest.is_file():
            if run.read(dest)['analysis_sha256'] != receipt['analysis_sha256']:
                raise ValueError('Previous execution binds a different result')
        else:
            run.once(dest, receipt)
        print(receipt, flush=True)


if __name__ == '__main__':
    main()
