"""Exercise registration and scoring with frozen proposals but no real label access."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_songdo_confirmation as runner

np, pack, protocol = runner.np, runner.pack, runner.protocol
OUT = runner.PUBLIC.parent / 'songdo_protocol_paths_rehearsal_v2'


class FabricatedTargets(runner.Songdo):
    def targets(self, name, cache, prefix, item):
        past = prefix['history']
        last_velocity = past[:, -1]-past[:, -2]
        y = last_velocity[:, None, :]*np.arange(1, 13)[None, :, None]
        actors = np.arange(len(past))
        y[:, :, 0] += np.where(actors % 2 == 0, .1, -.1)[:, None]
        valid = np.ones(y.shape[:2], dtype=bool)
        valid[actors % 9 == 0] = False
        valid[actors % 5 == 0, -4:] = False
        return np.where(valid[..., None], y, np.nan), valid


def run():
    data = FabricatedTargets()
    sites = sorted(set(data.sites.values()))
    spec = dict(mode='zero_shot', approval_reference='temporary synthetic rehearsal only',
                site_roles={site: 'confirmation' for site in sites}, cutoff_rule='not_used',
                release=dict(method='matched_only'))
    with TemporaryDirectory() as path, redirect_stdout(io.StringIO()):
        root = Path(path)
        work = root/'zero_shot'
        work.mkdir()
        runner.execute(work, 'register', spec, data)
        r = runner.execute(work, 'confirm', spec, data)
        assert runner.execute(work, 'confirm', spec, data) == r
        counts = {k: r[k] for k in ('flights', 'source_components', 'flight_seed_views', 'matched_seed_views')}
        if r['risk_and_release'] != 'not_run_no_calibration' or 'released' in r:
            raise ValueError('Matched-only mode reported release')
        work = root/'calibrated'
        work.mkdir()
        spec = dict(mode='calibrated', approval_reference='temporary synthetic rehearsal only',
                    site_roles={site: 'calibration' if site in 'ABCE' else 'confirmation' for site in sites},
                    cutoff_rule=protocol.CALIBRATION_CUTOFF_RULE, release=dict(method='risk_passing_score_prefix'))
        runner.execute(work, 'register', spec, data)
        runner.execute(work, 'calibrate', spec, data)
        c = runner.first.read(work/'calibration_cutoffs.json')
        d = runner.execute(work, 'confirm', spec, data)
        calibrated = {k: d[k] for k in ('flights', 'source_components', 'flight_seed_views', 'matched_seed_views')}
        calibrated.update(calibration_flights=len(c['calibration_flights']),
                          cutoff_receipts=len(list((work/'calibration_errors').glob('*.json'))))
        if counts != dict(flights=200, source_components=17, flight_seed_views=600, matched_seed_views=573):
            raise ValueError('Frozen observed roster changed')
        if (calibrated['flights'] != 160 or calibrated['source_components'] != 14
                or calibrated['cutoff_receipts'] != 40 or data.verified_archives):
            raise ValueError('Synthetic calibration or source access did not match the rehearsal')
    OUT.mkdir(exist_ok=True)
    pack.write_json(OUT / 'receipt.json', dict(result_source='synthetic_targets_on_frozen_real_proposals',
        input_binding=data.binding, rehearsal=pack.sha(Path(__file__)), zero_shot=counts, calibrated=calibrated,
        real_protocol_registered=False, real_future_index_loaded=False,
        real_future_coordinates_loaded=False, real_forecasts_scored=False, pseudo_effect_sizes_reported=False))
    print(dict(zero_shot=counts, calibrated=calibrated))


if __name__ == '__main__':
    run()
