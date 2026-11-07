"""Bound whole-record support acquisition without exporting oracle decisions."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_pruning_router as previous
from src.evaluation import m3w_support_acquisition as model

run, np, cost, base = previous.run, previous.np, previous.cost, previous.base
prior = previous.prior
PUBLIC, PRIVATE = prior.PUBLIC/'support_acquisition', prior.PRIVATE/'support_acquisition'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_support_acquisition.py')


def register():
    a = previous.PUBLIC/'analysis.json'
    if run.read(previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require verified preceding routing result')
    for reg in (cost.REG, previous.REG):
        for item in run.read(reg)['parents']:
            run.checked(item)
    parents = (a, previous.REG, base.REG, base.PUBLIC/'analysis.json', cost.REG,
               cost.PUBLIC/'freeze.json', Path(cost.__file__), Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], modes=model.MODES, budget=.02,
        exact_additions=1, max_deletions_per_query=1, independent_outcomes_opened=False)))


def view(m, rec, ix, groups, y, valid, e, h):
    d, p = m.d, run.parent
    s = str(d['sites'][ix[0]])
    c, b = m.floor[ix], m.bank[ix]
    direct = m.direct[ix]
    paths = {p.CV: c, 'direct': run.model.selected_paths(direct, b, c)}
    control = p.readout.score_recording(paths, {}, y[ix], valid[ix], d['queries'][ix], rec,
                                        easy_cut=e, hard_cut=h)['metrics']['direct']['risk']
    if control['completion_easy_risk_upper'] is not None:
        return None
    _, examples = cost.proposals(m.x[ix], m.scale[ix], b, c, groups, m.co[s], m.rank(s))
    shortlist = np.zeros(b.shape[:2], bool)
    for _, i, j, _ in examples:
        shortlist[i, j] = True
    problem = model.pool(c, b, y[ix], valid[ix], e, direct, groups, shortlist)
    result = {}
    for name in model.MODES:
        answer, status, columns = model.solve(problem, name)
        removal, gain, insertion_index, both = None, None, None, None
        if answer is not None:
            drop, add = answer
            chosen = direct.copy()
            chosen[problem['selected'][drop]] = -1
            chosen[problem['row'][add]] = problem['action'][add]
            prediction = run.model.selected_paths(chosen, b, c)
            out = p.readout.score_recording({**paths, 'edit': prediction},
                {'pair': ('direct', 'edit', p.CV)}, y[ix], valid[ix], d['queries'][ix], rec,
                easy_cut=e, hard_cut=h)
            pair = out['comparisons']['pair']
            g = pair['known_delta']-pair['unknown_radius']
            if not out['metrics']['edit']['risk']['supported'] or g < -1e-8:
                status = 'unresolved_postcheck'
            else:
                removal, gain = int(drop.sum()), float(g)
                insertion_index = int(problem['query'][add])
                both = int((problem['selected_query'][drop] == insertion_index).any())
                if any((chosen[q] >= 0).sum() > min(2, len(q)-1) for q in groups):
                    raise ValueError('Exact witness violates current-query cap')
                if name in ('early_joint', 'ordered_joint', 'shortlist_joint') and (
                        problem['selected_query'][drop] < insertion_index).any():
                    raise ValueError('Witness uses later support for earlier deletion')
        result[name] = dict(status=status, solver_anchor_columns=columns, removals=removal,
                            paired_lower_mass=gain, insertion_query_ordinal=insertion_index,
                            insertion_and_removal_same_query=both)
    return dict(seed=m.seed, source=s, zero_direct=not bool(control['selected']),
        full_legal_easy_anchors=int(len(problem['query'])),
        shortlist_easy_anchors=int(problem['shortlist'].sum()), result=result)


def execute(d, refs, records, y, valid, beat, replay):
    parent, controls, ranks = cost.banks()
    cuts = run.cuts_by_source()
    rows = []
    for cell in parent:
        m = base.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for j, (rec, (ix, g)) in enumerate(records.items()):
            site = str(d['sites'][ix[0]])
            mark = PRIVATE/f'{m.seed}_{j:03d}.json'
            identity = dict(registration=run.sha(REG), seed=m.seed, recording=rec, source=site)
            if not mark.exists():
                if replay:
                    raise ValueError('Replay requires complete solver receipts')
                row = view(m, rec, ix, g, y, valid, cuts[site]['easy'], cuts[site]['hard'])
                run.once(mark, run.stable(dict(identity=identity, row=row)))
            packet = run.read(mark)
            if packet['identity'] != identity:
                raise ValueError('Mismatched recording receipt')
            if packet['row'] is not None:
                rows.append(packet['row'])
            beat(state='view_checked', seed=m.seed, count=j+1)
        del m
    if (len(rows), sum(r['zero_direct'] for r in rows)) != (35, 18):
        raise ValueError('Original missing-support split differs')
    buckets = dict(all_missing=rows, zero_direct=[r for r in rows if r['zero_direct']],
                   active_no_easy=[r for r in rows if not r['zero_direct']])
    out = {}
    for name, records in buckets.items():
        stat = {}
        for mode in model.MODES:
            v = [r['result'][mode] for r in records]
            yes = [r for r in v if r['removals'] is not None]
            stat[mode] = dict(status=dict(Counter(r['status'] for r in v)), checked_feasible=len(yes),
                positive_gain=sum(r['paired_lower_mass'] > 0. for r in yes),
                removals_in_witnesses=sum(r['removals'] for r in yes),
                same_query_replacement_views=sum(r['insertion_and_removal_same_query'] for r in yes))
        out[name] = dict(views=len(records), modes=stat)
    def transition(a, b):
        return sum(r['result'][a]['removals'] is None and r['result'][b]['removals'] is not None for r in rows)
    run.once(PUBLIC/'analysis.json', run.stable(dict(groups=out, input_views=102, source_groups=14,
        checked_more_than_insertion=transition('insert_only', 'ordered_joint'),
        checked_more_than_early=transition('early_joint', 'ordered_joint'),
        checked_only_after_order_relaxation=transition('ordered_joint', 'free_joint'),
        checked_lost_at_shortlist=transition('shortlist_joint', 'ordered_joint'),
        oracle_only=True, oracle_choices_exported=False, inference_policy_evaluated=False,
        original_rules_unchanged=True, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
                                   cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit registration before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last completed recording')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('count', 0) % 5 == 0 or v.get('count') == 34:
                print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        execute(d, refs, records, lab['target_eval']-d['origin'][:, None], lab['valid'],
                beat, phase == 'replay')
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
