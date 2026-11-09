"""Exercise registration and scoring with frozen proposals but no real label access."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_songdo_confirmation as runner

np, pack, protocol = runner.np, runner.pack, runner.protocol
OUT = runner.PUBLIC.parent / 'songdo_confirmation_rehearsal_v1'


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
    spec = dict(mode='zero_shot', approval_reference='temporary synthetic rehearsal only',
                site_roles={site: 'confirmation' for site in sorted(set(data.sites.values()))},
                easy_cut=.25, hard_cut=1.,
                release=dict(method='fixed_scores', threshold_source='synthetic rehearsal fixture',
                             thresholds={str(s): {a: 'all' if a == 'joint' else None for a in protocol.ARMS}
                                         for s in protocol.SEEDS}))
    with TemporaryDirectory() as path, redirect_stdout(io.StringIO()):
        work = Path(path)
        runner.execute(work, 'register', spec, data)
        r = runner.execute(work, 'confirm', spec, data)
        assert runner.execute(work, 'confirm', spec, data) == r
        # Do not report forecast effects of invented outcomes.
        counts = {k: r[k] for k in ('flights', 'source_components', 'flight_seed_views', 'matched_seed_views')}
        released = {k: r['released'][k]['edited'] for k in protocol.ARMS}
        if counts != dict(flights=200, source_components=17, flight_seed_views=600, matched_seed_views=573):
            raise ValueError('Frozen observed roster changed')
        if released != {a: 573 if a == 'joint' else 0 for a in protocol.ARMS} or data.verified_archives:
            raise ValueError('Fabricated release or source access did not match the rehearsal')
    OUT.mkdir(exist_ok=True)
    pack.write_json(OUT / 'receipt.json', dict(result_source='synthetic_targets_on_frozen_real_proposals',
        input_binding=data.binding, rehearsal=pack.sha(Path(__file__)), counts=counts,
        fabricated_release_counts=released, real_protocol_registered=False, real_future_index_loaded=False,
        real_future_coordinates_loaded=False, real_forecasts_scored=False, pseudo_effect_sizes_reported=False))
    print(counts)


if __name__ == '__main__':
    run()
