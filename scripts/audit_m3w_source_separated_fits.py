"""Audit saved neural fits while training or after completion, never read Vlabels."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_separated_critics as run
from src.evaluation.m3w_source_separated_fit_audit import audit_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    source = run.source
    reg = source.read(run.REG)
    jobs = {j['key']: j for j in reg['jobs']}
    pairs, cells = {}, []
    for path in sorted((run.PUBLIC / 'fits').glob('*.json')):
        r = source.read(path)
        state = run.amplitude.base.base.read_checkpoint(source.checked(r['checkpoint']))
        part = reg['partitions'][r['job']][r['part']]
        expected_seed = jobs[r['job']]['seed']+1009*r['replica']
        result = audit_state(state, r, part, expected_seed)
        pair = pairs.setdefault((r['job'], r['part']), {})
        pair[r['replica']] = (r['preprocess_array_hashes'], source.array_hash(state['initial_model']['encoder.weight'].numpy()))
        cells.append(dict(fit=source.artifact(path), checks=result))
        del state
    complete_pairs = 0
    for pair in pairs.values():
        if set(pair) == {0, 1}:
            if pair[0][0] != pair[1][0] or pair[0][1] == pair[1][1]:
                raise ValueError('Same-data replicas must share fitting normalizer but have different initial weights')
            complete_pairs += 1
    result = source.stable(dict(completed_fits_audited=len(cells), complete_seed_pairs=complete_pairs, cells=cells,
        every_completed_fit_passed=True, future_targets_or_validation_data_read=False,
        status='complete_fit_isolation_audit' if args.complete else 'partial_fit_only_progress_not_validation'))
    if args.complete:
        freeze = source.read(run.PUBLIC / 'training_freeze.json')
        if len(cells) != 108 or complete_pairs != 54 or freeze['complete_new_heads'] != 108:
            raise ValueError('Do not call a partial audit complete')
        source.once(run.PUBLIC / 'fit_isolation_audit.json', result)
    print(source.json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))


if __name__ == '__main__': main()
