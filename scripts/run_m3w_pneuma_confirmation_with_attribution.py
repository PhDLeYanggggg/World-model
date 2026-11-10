"""Add prespecified same-recipient attribution to the single Athens readout."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import freeze_m3w_external_recipient_factorial as frozen
from scripts import run_m3w_pneuma_confirmation as primary
from src.evaluation import m3w_recipient_factorial_readout as factorial
from src.evaluation import m3w_two_factor_bounds as bounds

np, storage, read = primary.np, primary.storage, primary.pilot.read
PROTOCOL = primary.REHEARSAL.parent/'pneuma_recipient_attribution_v1'/'protocol.md'
MATCHUPS = {'independent': ('II', 'IJ', 'JI', 'JJ'), 'additive': ('AA', 'AJ', 'JA', 'JJ')}
FIELDS = ('observed_pp', 'lower_pp', 'upper_pp', 'missing_radius_pp', 'separate_pair_radius_pp')


def attribute(cv, offsets, pack, y, valid):
    q, p = factorial.restore(cv, offsets, pack)
    if len(q) != len(offsets)-1:
        raise ValueError('Athens factorial must use the full prespecified matched query roster')
    w = np.repeat(1./np.diff(offsets), np.diff(offsets))
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    return {k: bounds.contrast(*(p[a] for a in cells), y, valid, w, mass) for k, cells in MATCHUPS.items()}


def check_diagonal(value, original):
    for name, r in value.items():
        diagonal = original['matched_contrasts'][name]
        if r is None:
            if diagonal['observed_pp'] is not None:
                raise ValueError('Attribution and primary label support differ')
        else:
            np.testing.assert_allclose([r['total'][k] for k in FIELDS[:3]],
                                       [diagonal[k] for k in FIELDS[:3]], atol=1e-11, rtol=0.)


class Dataset(primary.Dataset):
    def __init__(self, spec):
        super().__init__()
        self.capture = None
        roles = primary.protocol.assignments(spec, self.dates)
        self.confirmation = {n for n, role in roles.items() if role == 'confirmation'}
        path = frozen.OUT/'athens_freeze.json'
        bank = read(path)
        if (bank['future_coordinates_read'] or bank['roles_assigned'] or bank['accuracy_scored']
                or not bank['original_diagonals_reproduced'] or bank['matched_views'] != 1929
                or set(bank['packets']) != set(self.menu)):
            raise ValueError('Require the exact unscored crossed proposal bank')
        self.crossed = bank['packets']
        for tag, r in self.crossed.items():
            if r['original_sha256'] != self.menu[tag]['proposals_sha256'] or r['full'] != self.menu[tag]['queries']:
                raise ValueError('Crossed proposals do not bind the original forecasts')
            primary.verify_hash(frozen.LOCAL/'athens'/f'{tag}.npz', r['packet_sha256'])
        self.binding.update(crossed_forecast_freeze=storage.sha(path), crossed_protocol=storage.sha(PROTOCOL),
                            crossed_runner=storage.sha(Path(__file__)), crossed_readout=storage.sha(Path(factorial.__file__)),
                            crossed_bounds=storage.sha(Path(bounds.__file__)))

    def load_targets(self, name, prefix):
        return super()._labels(name, prefix)

    def _labels(self, name, prefix):
        target = self.load_targets(name, prefix)
        if self.capture is not None:
            if self.capture:
                raise ValueError('Confirmation may join the member future only once')
            self.capture.append((prefix['offsets'], target))
        return target

    def score(self, name, easy, hard):
        if name not in self.confirmation:
            return super().score(name, easy, hard)
        _, f = self._forecasts(name)
        self.capture = []
        try:
            main = super().score(name, easy, hard)
            (offsets, (y, valid)), = self.capture
        finally:
            self.capture = None
        for s, value in main.items():
            tag = f'{Path(name).stem}_{s}'
            p = frozen.LOCAL/'athens'/f'{tag}.npz'
            primary.verify_hash(p, self.crossed[tag]['packet_sha256'])
            with np.load(p, allow_pickle=False) as pack:
                summary = attribute(f['cv'], offsets, pack, y, valid)
            check_diagonal(summary, value)
            value['recipient_path_attribution'] = summary
        return main


def report(result, public, work):
    """Rebuild a secondary report entirely from the original member receipts."""
    records = {}
    for name in result['members']:
        p = work/'confirmation'/f'{Path(name).stem}.readout.json'
        r = read(p)
        if r['binding'] != result['binding'] or r['member'] != name:
            raise ValueError('Scalars do not bind the primary run')
        records[name] = r['score']
    out = {}
    for k in MATCHUPS:
        out[k] = {}
        for a in ('total', 'path', 'recipient'):
            def extract(r, field):
                part = r['recipient_path_attribution'][k]
                return part[a][field] if part is not None else None
            out[k][a] = {m: primary.readout.date_interval(records, result['dates'], lambda r: extract(r, m))
                         for m in FIELDS}
        for m in FIELDS[:3]:
            a, b = out[k]['total'][m], result['matched']['joint_minus'][k][m]
            if a['status'] != b['status'] or set(a['by_date']) != set(b['by_date']):
                raise ValueError('Secondary effects and primary roster differ')
            if a['point'] is not None:
                np.testing.assert_allclose(a['point'], b['point'], atol=1e-11, rtol=0.)
    result = dict(primary_confirmation=storage.sha(public/'confirmation.json'),
                  unchanged_primary_protocol=True, no_threshold_changes=True,
                  design='same_recipient_path_two_order_average', mode=result['mode'],
                  source_date_count=len(set(result['dates'].values())), by_comparator=out)
    storage.write_json(public/'recipient_path_attribution.json', result)
    return result


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
            value = report(value, public, work)
    print({k: v for k, v in value.items() if k not in ('by_comparator', 'matched', 'release')}, flush=True)


if __name__ == '__main__':
    main()
