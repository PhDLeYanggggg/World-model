"""Compare oracle limits imposed by prescreening and matched counts."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_selected_positive_harm as study
from src.evaluation import m3w_admissible_pool_ceiling as oracle

run, np = study.run, study.np
PUBLIC = study.PUBLIC/'oracle'; PRIVATE = study.PRIVATE/'oracle'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(oracle.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_admissible_pool_ceiling.py')
ARMS = ('screened_matched', 'full_matched', 'screened_capped_free')


def register():
    result, replay, freeze = [study.PUBLIC/n for n in ('analysis.json','replay_execution.json','freeze.json')]
    if run.read(replay)['artifact']['sha256'] != run.sha(result):
        raise ValueError('Require exact matched policy readout')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES], parents=[run.artifact(p) for p in
        (result, replay, freeze, study.REG, Path(study.__file__), run.REG, Path(run.__file__))],
        arms=ARMS, timeout_seconds=5., expected_views=102, original_rules_unchanged=True,
        label_aware_only=True, choices_exported=False, external_outcomes=False)))


def execute(replay):
    d = run.population(); refs = run.register(); cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid')); s, rec = d['sites'], d['recordings']
    records = []
    for f in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = f['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids']); names, coef = z['names'], z['coefficients']
        ref = next(r for r in run.read(study.nested.base.PUBLIC/'freeze.json')['choices'] if r['seed'] == seed)
        with np.load(run.checked(ref['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids']); raw = z['choice']
        allowed = np.empty(paths.shape[:2], bool)
        for name, c in zip(names, coef):
            ix = np.flatnonzero(s == name)
            x, scale = run.model.descriptors(d['history'][ix], paths[ix], floor[ix], d['crowd'][ix])
            allowed[ix] = study.model.admissible(*run.model.decision_costs(x, c, paths[ix], floor[ix], scale))
        for record in np.unique(rec):
            ix = np.flatnonzero(rec == record); src = str(s[ix[0]]); c = cuts[src]; r = str(record)
            checkpoint = PRIVATE/f'{seed}_{r}.json'
            ident = dict(seed=seed, source=src, recording=r, registration=run.sha(REG))
            if replay or not checkpoint.exists():
                args = (raw[ix], labels['target_eval'][ix]-d['origin'][ix, None], labels['valid'][ix],
                        d['queries'][ix], r, c['easy'], c['hard'])
                arms = {key: oracle.solve(floor[ix], paths[ix], ok, *args, fixed=fixed) for key, ok, fixed in
                    zip(ARMS, (allowed[ix], np.ones_like(allowed[ix]), allowed[ix]), (True,True,False))}
                run.once(checkpoint, run.stable(dict(identity=ident, arms=arms, raw_switches=int((raw[ix]>=0).sum()))))
            out = run.read(checkpoint)
            if out['identity'] != ident:
                raise ValueError('Cannot resume a different oracle view')
            records.append(out)
        print(dict(pid=os.getpid(), seed=seed, complete=len(records)), flush=True)
    if len(records) != 102:
        raise ValueError('No view omitted')
    summary = {}
    for a in ARMS:
        v = [r['arms'][a] for r in records]
        summary[a] = dict(status=Counter(x['status'] for x in v),
            infeasible_reasons=Counter(x['reason'] for x in v if x['status']=='infeasible'),
            positive_feasible=sum(x.get('completion_gain_mass',0.)>0 for x in v))
    run.once(PUBLIC/'analysis.json', run.stable(dict(records=records, summary=summary,
        original_rules_unchanged=True, no_label_aware_choices_exported=True, independent_confirmation=False)))
    print(summary, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register','score','replay'))
    phase = p.parse_args().phase; register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit diagnostic definition before scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(phase == 'replay')
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
