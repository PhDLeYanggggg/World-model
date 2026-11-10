"""Bind the candidate zero-shot easy/hard cutoffs using OpenDD only."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_pneuma_confirmation as reader
from scripts import run_m3w_vehicle_pair_learning as learn


def derive(f):
    expected = set(combinations(learn.base.SITES, 2))
    records = defaultdict(dict)
    for c in f['cells'].values():
        split = tuple(sorted(c['packets']))
        if split not in expected or set(c['fit_sites']) != set(learn.base.SITES)-set(split) or (
                c['seed'] not in learn.base.SEEDS or c['seed'] in records[split]):
            raise ValueError('Unexpected donor training roles')
        cuts = c['cut']
        if len(cuts) != 2 or not (0 < cuts[0] <= cuts[1] < reader.np.inf):
            raise ValueError('Invalid donor cutoffs')
        records[split][c['seed']] = tuple(cuts)
    if set(records) != expected or any(set(v) != set(learn.base.SEEDS) or len(set(v.values())) != 1
                                      for v in records.values()):
        raise ValueError('Donor folds or seed-invariant cutoffs did not reproduce')
    values = [next(iter(v.values())) for _, v in sorted(records.items())]
    return reader.np.median(values, axis=0).tolist(), len(values)


def main():
    frozen, reg = (learn.OUT/f'{k}.json' for k in ('freeze', 'registration'))
    data = reader.pilot.read(frozen)
    if data['registration'] != reader.storage.sha(reg) or not data['held_choices_frozen']:
        raise ValueError('Unverified donor freeze')
    (easy, hard), n = derive(data)
    reader.REHEARSAL.mkdir(exist_ok=True)
    result = dict(result_source='cached_verified_donor_only', donor_freeze=reader.storage.sha(frozen),
        donor_registration=reader.storage.sha(reg), method='componentwise_median_of_21_seed_collapsed_donor_fold_cuts',
        donor_exclusion_pairs=n, easy_cut=easy, hard_cut=hard, official_risk_budget=.02,
        receiver_labels_opened=False, predictive_roles_assigned=False, operating_rule_approved=False)
    reader.storage.write_json(reader.REHEARSAL/'donor_cuts.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
