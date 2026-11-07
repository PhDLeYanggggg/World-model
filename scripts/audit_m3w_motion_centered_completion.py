"""Re-test coherent pair ranking with the better-supported tail center."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_completion_occlusion as occlusion
from src.evaluation import m3w_motion_centered_completion as motion

witness, run, np = occlusion.witness, occlusion.run, occlusion.np
PUBLIC = witness.PUBLIC.parent/'native_motion_centered_completion_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(motion.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_motion_centered_completion.py')


def register():
    roster = occlusion.register()
    bindings = [*FILES, occlusion.REG, witness.REG]
    for parent in (occlusion.PUBLIC, witness.PUBLIC):
        analysis, receipt = (parent/f for f in ('analysis.json', 'replay_execution.json'))
        if run.read(receipt)['artifact'] != run.artifact(analysis):
            raise ValueError('Require replayed predecessor analysis')
        bindings.extend((analysis, receipt))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], cases=witness.CASES,
        contrast='joint_vs_pair_additive', tail_center='anchored_ols4_when_supported',
        primary='opposite_sign_witnesses_at_1x', official_rule_unchanged=True, external_access=False)))
    return roster


def score(roster):
    old = run.read(witness.PUBLIC/'analysis.json')
    outer = run.read(witness.parent.PUBLIC/'analysis.json')['comparisons']['joint_vs_pair_additive']
    d = run.population()
    labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    y, v, frames = (labels[k] for k in ('target_eval', 'valid', 'frames'))
    agents, ids = (np.load(run.checked(roster['files'][k]), mmap_mode='r', allow_pickle=False)
                  for k in ('agents', 'global_ids'))
    np.testing.assert_array_equal(ids, d['global_ids'])
    paths, switches = witness.get_paths(d)
    if any(count != 9675 for count in switches.values()):
        raise ValueError('Intervention budget changed')
    records = witness.parent.prior.prior.h.previous.prior.blocks(d, frames)
    groups = sorted(set(str(g) for g in d['sites']))
    seeds = tuple(old['seeds'])
    sums = {g: np.zeros((len(witness.CASES)+1, len(seeds))) for g in groups}
    sizes = dict.fromkeys(groups, 0)
    counts = {}
    for rec, (ix, qs) in records.items():
        n, result, counts[rec] = motion.recording(agents[ix], frames[ix], d['history'][ix], y[ix], v[ix],
                                                  [paths[s][:,ix] for s in seeds], qs)
        g = str(d['sites'][ix[0]])
        sizes[g] += n
        sums[g] += result
    if sum(sizes.values()) != old['original_queries'] or len(records) != old['recordings'] or len(groups) != old['source_groups']:
        raise ValueError('Original query or source roster changed')
    draws = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    cube = np.stack([sums[g]/sizes[g] for g in groups])
    names = ('recentered_reference', *witness.CASES[1:], 'observed_term')
    rows = {k: witness.pair.bootstrap(cube[:, j], draws) for j,k in enumerate(names)}
    np.testing.assert_allclose(rows['observed_term']['point'], old['cases']['observed_term']['point'], atol=1e-8)
    lo, hi = (outer[f'anchored_{k}']['point'] for k in ('lower','upper'))
    if any(not lo-1e-8 <= rows[k]['point'] <= hi+1e-8 for k in names[:-1]):
        raise ValueError('A feasible completion left the existing outer bounds')
    return dict(cases=rows, opposite_signs={str(k): rows[f'low_{k}x']['point'] < 0 < rows[f'high_{k}x']['point']
        for k in (1,2)}, counts={k: sum(c[k] for c in counts.values()) for k in next(iter(counts.values()))},
        recordings=len(records), source_groups=len(groups), seeds=seeds, queries=sum(sizes.values()),
        matched_switches=switches, bootstrap_draws=3000, feasible_witnesses_not_global_extrema=True,
        independent_confirmation=False, missing_labels_imputed=False, official_rule_changed=False,
        external_access=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    roster = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the protocol before scoring')
    started = time.monotonic()
    out = PUBLIC/'analysis.json'
    if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
        raise ValueError('Require verified original score')
    run.once(out, run.stable(score(roster)))
    mark = PUBLIC/f'{phase}_execution.json'
    if not mark.exists():
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-started, artifact=run.artifact(out)))
    elif run.read(mark)['artifact'] != run.artifact(out):
        raise ValueError('Witness replay differs')


if __name__ == '__main__':
    main()
