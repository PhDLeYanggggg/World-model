"""Exercise the Swiss metric adapter on frozen predictions with fake labels."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_swiss_roundabout_pilot as pilot
from src.evaluation import m3w_swiss_roundabout_readout as evaluator

source, np = pilot.source, pilot.np


def run():
    raw = {k: source.OUT/f'{k}_receipt.json' for k in ('prefix', 'forecast', 'matched')}
    prefix, forecast, matched = (pilot.read(raw[k]) for k in ('prefix', 'forecast', 'matched'))
    bindings = {k: source.digest(p) for k, p in raw.items()}
    if (forecast['binding']['prefix'] != bindings['prefix'] or matched['binding']['prefix'] != bindings['prefix']
            or matched['binding']['forecast'] != bindings['forecast']
            or any(r['roles_assigned'] or r['forecasts_scored'] or r['future_coordinates_parsed']
                   for r in (prefix, forecast, matched))):
        raise ValueError('Require unscored Swiss packets with aligned inputs')
    count = Counter()
    for name, f in forecast['files'].items():
        stem = Path(name).stem
        p, q = (pilot.LOCAL/f'{stem}.{k}.npz' for k in ('prefix', 'forecast'))
        if source.digest(p) != prefix['files'][name]['prefix_sha256'] or source.digest(q) != f['forecast_sha256']:
            raise ValueError('Past or forecast packets changed')
        with np.load(p, allow_pickle=False) as past, np.load(q, allow_pickle=False) as pred:
            off, cv, ols, med = past['offsets'], pred['cv'], pred['ols4'], pred['median4']
        size = np.diff(off)
        y = cv.copy()
        y[:, :, 0] += (.2+.05*(np.arange(len(cv)) % 23))[:, None]
        valid = np.ones(y.shape[:2], dtype=bool)
        valid[::11, 6:] = False
        valid[::53] = False
        y[~valid] = np.nan
        for seed in forecast['seeds']:
            tag = f'{stem}_{seed}'
            m = matched['packets'][tag]
            path = pilot.LOCAL/'matched'/f'{tag}.npz'
            if (source.digest(path) != m['sha256'] or m['identity']['member'] != name or m['identity']['seed'] != seed
                    or m['identity']['prefix_sha256'] != f['prefix_sha256']
                    or m['identity']['forecast_sha256'] != f['forecast_sha256']):
                raise ValueError('Unbound proposal')
            with np.load(path, allow_pickle=False) as z:
                rows = evaluator.score(cv, ols, med, y, valid, off, z, easy_cut=.6, hard_cut=1.)
                selected = z['query_index']
                if (rows['matched_queries'] != m['available'] or rows['coverage']['windows'] != m['original']
                        or not all(rows['full_roster'][a]['original_risk']['rows'] == len(cv) for a in evaluator.ARMS)):
                    raise ValueError('Matched support or full risk denominator changed')
            count.update(packets=1, matched=rows['matched_queries'], unmatched=rows['unmatched_queries'],
                         two_agent_matched=int((size[selected] == 2).sum()), two_agent_original=int((size == 2).sum()))
    if (count['matched'], count['matched']+count['unmatched'], count['packets']) != (
            matched['comparable_views'], matched['original_views'], len(matched['packets'])):
        raise ValueError('Full rehearsal did not cover all frozen banks')
    if count['two_agent_matched']:
        raise ValueError('The matched menu must retain the three-actor rule')
    out = source.OUT.parent/'swiss_roundabouts_readout_rehearsal_v1'/'receipt.json'
    source.write(out, dict(result_source='fresh_frozen_inputs_synthetic_outcomes_rehearsal',
        bindings=dict(bindings, driver=source.digest(Path(__file__)),
                      readout_code=source.digest(Path(evaluator.__file__))),
        checked=count, full_roster_used_for_original_risk=True, original_2pct=True,
        real_future_positions_read=False, real_accuracy_scored=False, real_roles_assigned=False, deployment_changed=False))
    print(dict(count), flush=True)


if __name__ == '__main__':
    run()
