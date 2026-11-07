"""Replay a synthetic gap check of the registered pair witness radius."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_pair_completion_witness as witness
from src.evaluation import m3w_completion_occlusion as check

run, np = witness.run, witness.np
PUBLIC = witness.PUBLIC.parent/'native_completion_occlusion_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(check.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_completion_occlusion.py')


def register():
    _, roster = witness.register()
    analysis, receipt = (witness.PUBLIC/f for f in ('analysis.json', 'replay_execution.json'))
    if run.read(receipt)['artifact'] != run.artifact(analysis):
        raise ValueError('Require the replay-verified pair witness')
    paths = (*FILES, witness.REG, analysis, receipt, Path(witness.completion_ledger.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in paths], cases=check.KEYS,
        primary='terminal_12_ols4_minus_constant_gap_coverage_1x', bootstrap=3000,
        external_access=False, official_rule_unchanged=True, no_missing_labels_filled=True)))
    return roster


def score(roster):
    d = run.population()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    a, ids = (np.load(run.checked(roster['files'][k]), mmap_mode='r', allow_pickle=False)
              for k in ('agents', 'global_ids'))
    np.testing.assert_array_equal(ids, d['global_ids'])
    y, valid, frames = (lab[k] for k in ('target_eval', 'valid', 'frames'))
    records = witness.parent.prior.prior.h.previous.prior.blocks(d, frames)
    groups = sorted(set(str(x) for x in d['sites']))
    counts = {g: np.zeros((len(check.KEYS), 7)) for g in groups}
    actual = {k: np.zeros(13, dtype=int) for k in ('leading', 'internal', 'terminal')}
    for rec, (ix, _) in records.items():
        agent, group = a[ix], str(d['sites'][ix[0]])
        tx, known, base, _, gap_id = witness.completion_ledger(agent, frames[ix], d['history'][ix], y[ix], valid[ix])
        tracks = np.unique(agent)
        n = len(known)//len(tracks)
        masked = np.unique(tx[~valid[ix]])
        gap, multiplicity = np.unique(gap_id[masked], return_counts=True)
        track_idx, insert = divmod(gap, n+1)
        for j, name in enumerate(tracks):
            local = slice(j*n, (j+1)*n)
            seen = known[local]
            rows = check.pseudo_gaps(base[local], seen, key=f'{rec}:{name}')
            for r, key in enumerate(check.KEYS):
                if key in rows:
                    counts[group][r] += rows[key]
            idx = track_idx == j
            pos, length = insert[idx], np.minimum(multiplicity[idx], 12)
            for kind, mask in (('leading', pos == 0), ('internal', (pos > 0) & (pos < seen.sum())),
                               ('terminal', pos == seen.sum())):
                np.add.at(actual[kind], length[mask], 1)
    if len(groups) != 14 or len(records) != 34:
        raise ValueError('Expected the original exposed development roster')
    matrix = np.stack([counts[g] for g in groups])
    draws = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    rows = {'_'.join(map(str, k)): check.summary(matrix[:, i], draws) for i,k in enumerate(check.KEYS)}
    left, right = [matrix[:, check.KEYS.index(('terminal', 12, k))] for k in ('constant', 'ols4')]
    np.testing.assert_array_equal(left[:, :2], right[:, :2])
    diff = (right[:, 4]-left[:, 4])/left[:, 0]
    primary = dict(point=float(diff.mean()), interval95=np.quantile(diff[draws].mean(1), [.025, .975]).tolist())
    return dict(recordings=len(records), source_groups=len(groups), bootstrap_draws=3000,
        actual_requested_gap_counts={k: v.tolist() for k,v in actual.items()},
        gap_count_index_12_includes_larger=True, pseudo_gap_diagnostics=rows, primary=primary,
        truth_mask_consistent_across_overlapping_queries=True, independent_confirmation=False,
        synthetic_deletion_only=True, official_risk_rule_changed=False, missing_positions_filled=False,
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
            raise ValueError('Commit the protocol before the stress test')
    started = time.monotonic()
    out = PUBLIC/'analysis.json'
    if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
        raise ValueError('Require intact original score')
    run.once(out, run.stable(score(roster)))
    mark = PUBLIC/f'{phase}_execution.json'
    if not mark.exists():
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-started, artifact=run.artifact(out)))
    elif run.read(mark)['artifact'] != run.artifact(out):
        raise ValueError('Stress test replay changed')


if __name__ == '__main__':
    main()
