"""Exercise the secondary report with fabricated futures and frozen proposals."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rehearse_m3w_songdo_confirmation as prior
from scripts import report_m3w_songdo_secondary as report

runner = prior.runner
OUT = prior.OUT.parent/'songdo_secondary_rehearsal_v1'


def run():
    dataset = prior.FabricatedTargets()
    sites = sorted(set(dataset.sites.values()))
    summaries = {}
    with TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
        for mode in ('calibrated', 'zero_shot'):
            only = mode == 'zero_shot'
            path = Path(d)/mode
            path.mkdir()
            spec = dict(mode=mode, approval_reference='temporary fabricated-target rehearsal',
                        site_roles={s: 'calibration' if not only and s in 'ABCE' else 'confirmation' for s in sites},
                        cutoff_rule='not_used' if only else prior.protocol.CALIBRATION_CUTOFF_RULE,
                        release={'method': 'matched_only' if only else 'risk_passing_score_prefix'})
            runner.execute(path, 'register', spec, dataset)
            if not only:
                runner.execute(path, 'calibrate', spec, dataset)
            runner.execute(path, 'confirm', spec, dataset)
            r = report.read_finished(path)
            assert ('released' in r) != only
            counts = {k: r[k] for k in ('flights', 'components', 'flight_seed_views', 'matched_seed_views')}
            fields = sorted(r['matched']['joint']['metrics'])
            summaries[mode] = dict(**counts, matched_fields=fields, released_report='released' in r,
                                   primary_replayed=r['primary_unchanged'], per_flight_receipts=len(
                                       r['binding']['flight_score_sha256']))
        if dataset.verified_archives or summaries['calibrated']['flights'] != 160 or (
                summaries['zero_shot']['flights'], summaries['zero_shot']['matched_seed_views']) != (200, 573):
            raise ValueError('Synthetic rehearsal differs from the frozen proposal roster')
    result = dict(modes=summaries, input_binding=dataset.binding, source_kind='synthetic_targets_on_real_proposals',
                  real_protocol_registered=False, real_future_index_loaded=False,
                  real_future_coordinates_loaded=False, effect_sizes_exported=False,
                  original_2pct_unchanged=True, submission_ready=False)
    report.write(OUT/'receipt.json', result)
    print(summaries, flush=True)


if __name__ == '__main__':
    run()
