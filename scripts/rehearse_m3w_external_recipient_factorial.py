"""Exercise the complete external factorial with fabricated future targets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import freeze_m3w_external_recipient_factorial as study
from src.evaluation import m3w_recipient_factorial_readout as readout

np, read, sha = study.np, study.pilot.pilot.read, study.store.sha


def run():
    cohorts = {}
    for which in ('athens', 'swiss'):
        local, _, forecasts, _, _, _ = study.cohort(which)
        path = study.OUT/f'{which}_freeze.json'
        frozen = read(path)
        if frozen['future_coordinates_read'] or frozen['roles_assigned'] or not frozen['original_diagonals_reproduced']:
            raise ValueError('Require the unscored factorial')
        roster, matched, full = 0, 0, 0
        for tag, row in frozen['packets'].items():
            stem, _ = tag.rsplit('_', 1)
            fp = study.checked(local/f'{stem}.forecast.npz', forecasts[stem+'.csv']['forecast_sha256'])
            pf = study.checked(study.LOCAL/which/f'{tag}.npz', row['packet_sha256'])
            with np.load(fp, allow_pickle=False) as f, np.load(pf, allow_pickle=False) as p:
                cv = f['cv']
                y = cv.copy()
                y[:, :, 0] += (.2+.05*(np.arange(len(cv)) % 23))[:, None]
                valid = np.ones(y.shape[:2], dtype=bool)
                valid[::11, 6:] = False
                valid[::53] = False
                y[~valid] = np.nan
                result = readout.score(cv, f['ols4'], y, valid, p['offsets'], p, study.CONTRASTS,
                                       easy_cut=.6, hard_cut=1.)
            if (result['matched'] != row['matched'] or result['matched']+result['unmatched'] != row['full']
                    or any(result['arms'][a]['original_risk']['rows'] != len(cv) or
                           result['arms'][a]['original_risk']['selected'] != 2*row['matched'] for a in readout.CELLS)):
                raise ValueError('Readout changed original matched support or risk roster')
            matched += result['matched']
            full += row['full']
            roster += 1
            print(which, roster, len(frozen['packets']), flush=True)
        if matched != frozen['matched_views'] or full != frozen['full_views']:
            raise ValueError('Rehearsal failed to cover the complete freeze')
        cohorts[which] = dict(freeze=sha(path), packets=roster, matched=matched, full=full)
    study.store.write_json(study.OUT/'rehearsal.json', dict(cohorts=cohorts, driver=sha(Path(__file__)),
        evaluator=sha(Path(readout.__file__)), real_futures_read=False, original_2pct=True,
        absolute_or_relative_metrics_from_fake_targets_not_exported=True))


if __name__ == '__main__':
    run()
