"""Score the primary Athens test and both frozen secondary controls in one join."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import freeze_m3w_external_step_budget as frozen
from scripts import run_m3w_pneuma_confirmation_with_attribution as attributed

primary, read, storage, np = attributed.primary, attributed.read, attributed.storage, attributed.np
budget = frozen.budget
PUBLIC = primary.REHEARSAL.parent/'pneuma_single_readout_v1'
PROTOCOL = PUBLIC/'protocol.md'
FIELDS = ('observed_pp', 'lower_pp', 'upper_pp', 'relative_observed', 'relative_lower', 'relative_upper')


def restore(cv, offsets, proposal, profiles):
    """Rebuild all comparisons on the original roster and verify the freeze."""
    expected = {f'{c}_{suffix}' for c in frozen.CONTROLS for suffix in ('queries', 'profile')}
    if set(profiles.files) != expected or not np.array_equal(proposal['offsets'], offsets):
        raise ValueError('Stepwise roster or profile schema changed')
    q = np.arange(len(offsets)-1)
    out = {}
    for control in frozen.CONTROLS:
        pair = budget.build_pair(cv, offsets, q, proposal, control)
        if (pair['collapsed'] or not np.array_equal(q, pair['queries'])
                or not np.array_equal(q, profiles[f'{control}_queries'])
                or not np.array_equal(pair['profiles'], profiles[f'{control}_profile'])):
            raise ValueError('Stepwise comparison does not reproduce its frozen profile and full roster')
        out[control] = pair
    return out


def same_scores(current, reference, fields):
    for field in fields:
        a, b = current[field], reference[field]
        if (a is None) != (b is None):
            raise ValueError('Stepwise and primary scores have different support')
        if a is not None:
            np.testing.assert_allclose(a, b, atol=1e-11, rtol=0.)


class Dataset(attributed.Dataset):
    def __init__(self, spec):
        super().__init__(spec)
        p = frozen.OUT/'athens.json'
        freeze = read(p)
        if (freeze['future_coordinates_read'] or freeze['real_accuracy_scored'] or freeze['real_roles_assigned']
                or not freeze['original_2pct_unchanged'] or freeze['deployment_changed']
                or (freeze['matched_views'], freeze['full_views']) != (1929, 1929)
                or set(freeze['packets']) != set(self.menu)):
            raise ValueError('Require the original forecast-only Athens stepwise freeze')
        check = dict(comparison=storage.sha(Path(budget.__file__)), driver=storage.sha(Path(frozen.__file__)),
                     protocol=storage.sha(frozen.OUT/'protocol.md'), forecast=self.binding['forecast_receipt.json'],
                     matched=self.binding['matched_receipt.json'], prefix=self.binding['prefix_receipt.json'],
                     source=self.binding['samples.json'])
        if freeze['bindings'] != check:
            raise ValueError('Frozen stepwise inputs or scorer have changed')
        self.binding.update(stepwise_freeze=storage.sha(p), stepwise_protocol=storage.sha(PROTOCOL),
                            stepwise_runner=storage.sha(Path(__file__)))
        self.pairs, self.cv = {}, {}
        self.step_capture = None
        # Validate the complete prospective bank before any target read.
        for name in sorted(self.dates):
            prefix, forecast = self._forecasts(name)
            self.cv[name] = forecast['cv'].astype(float)
            stem = Path(name).stem
            for seed in primary.readout.SEEDS:
                tag = f'{stem}_{seed}'
                row, meta = freeze['packets'][tag], self.menu[tag]
                if (row['matched'], row['full'], row['source_proposal_sha256']) != (
                        meta['queries'], meta['queries'], meta['proposals_sha256']):
                    raise ValueError('Stepwise packet does not bind the original proposal')
                original = primary.pilot.LOCAL/'matched'/f'{tag}.npz'
                profile = frozen.LOCAL/'athens'/f'{tag}.npz'
                primary.verify_hash(original, row['source_proposal_sha256'])
                primary.verify_hash(profile, row['profile_sha256'])
                with np.load(original, allow_pickle=False) as proposal, np.load(profile, allow_pickle=False) as z:
                    self.pairs[tag] = restore(self.cv[name], prefix['offsets'], proposal, z)

    def _labels(self, name, prefix):
        data = super()._labels(name, prefix)
        if self.step_capture is not None:
            if self.step_capture:
                raise ValueError('Secondary scoring cannot join futures again')
            self.step_capture.append((prefix['offsets'], data))
        return data

    def score(self, name, easy, hard):
        if name not in self.confirmation:
            return super().score(name, easy, hard)
        self.step_capture = []
        try:
            score = super().score(name, easy, hard)
            (offsets, (y, valid)), = self.step_capture
        finally:
            self.step_capture = None
        cv = self.cv[name]
        for seed, result in score.items():
            pairs = self.pairs[f'{Path(name).stem}_{seed}']
            comparisons = {}
            for control, pair in pairs.items():
                item = budget.score_pair(cv, offsets, pair, y, valid)
                if item is None:
                    raise ValueError('An expected matched control was empty')
                same_scores(item['original'], result['matched_contrasts'][control],
                            (*FIELDS, 'relative_observed_queries'))
                comparisons[control] = item
            result['stepwise_budget_sensitivity'] = comparisons
        return score

def report(result, public, work):
    """Replay the secondary scalars without opening targets."""
    records = {}
    for name in result['members']:
        row = read(work/'confirmation'/f'{Path(name).stem}.readout.json')
        if row['binding'] != result['binding'] or row['member'] != name:
            raise ValueError('Stepwise scores do not bind the primary run')
        records[name] = row['score']
    controls = {}
    for control in frozen.CONTROLS:
        variants = {}
        for variant in budget.VARIANTS:
            def metric(field):
                return primary.readout.date_interval(records, result['dates'],
                    lambda r: r['stepwise_budget_sensitivity'][control][variant][field])
            variants[variant] = {field: metric(field) for field in FIELDS}
        for field, r in variants['original'].items():
            base = result['matched']['joint_minus'][control][field]
            if r['status'] != base['status'] or set(r['by_date']) != set(base['by_date']):
                raise ValueError('Stepwise and primary date rosters differ')
            same_scores(r['by_date'], base['by_date'], r['by_date'])
            same_scores(r, base, ('point', 'ci95'))
        controls[control] = variants
    out = dict(primary_confirmation=storage.sha(public/'confirmation.json'),
               attribution_receipt=storage.sha(public/'recipient_path_attribution.json'),
               mode=result['mode'], source_date_count=len(set(result['dates'].values())),
               primary_unchanged=True, thresholds_changed=False, release_recomputed=False,
               secondary_utility_only=True, by_comparator=controls)
    storage.write_json(public/'stepwise_budget_sensitivity.json', out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'calibrate', 'confirm'))
    p.add_argument('--protocol', type=Path, required=True)
    args = p.parse_args()
    spec = read(args.protocol)
    public, work = primary.OUT, primary.WORK
    public.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with (work/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = Dataset(spec)
        data.check_cuts(spec)
        value = primary.execute(args.phase, spec, data, public, work)
        if args.phase == 'confirm':
            attributed.report(value, public, work)
            value = report(value, public, work)
    print({k: v for k, v in value.items() if k != 'by_comparator'}, flush=True)


if __name__ == '__main__':
    main()
