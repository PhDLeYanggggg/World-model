"""Measure hindsight capacity when the original intervention mask cannot change."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_shortlist as study

ctx, action, screen, np = study.ctx, study.action, study.screen, study.np
OUT, LOCAL, read, write, sha = study.OUT, study.LOCAL, study.read, study.write, study.sha


def main():
    key, source, _, parent, _, _, _ = study.register()
    fp = OUT/'freeze.json'
    f = read(fp)
    if f['registration'] != key or not f['before_held_scoring']:
        raise ValueError('Require the complete frozen shortlist')
    data = ctx.load_features(source)
    out = {q: Counter() for q in ('ego_count', 'fixed_15pct')}
    for name, c in parent['cells'].items():
        for s, r in c['packets'].items():
            with screen.packet_file(ctx.study.LOCAL, r) as p, screen.packet_file(LOCAL, f['cells'][name]['packets'][s]) as z:
                selected = (z['old_mask'], screen.model.top_mask(z['scores'][0], int(np.ceil(.15*len(z['old_mask'])))))
                t = action.labels(data[s]['file'])
                wins = np.asarray([v[4][3][1:] for v in study.truth(p, t, c['cut'], z['shortlist'])])
                for (q, counter), mask in zip(out.items(), selected, strict=True):
                    k = wins[mask]
                    counter.update(count=int(mask.sum()), strict_top1=int(k[:, 0].sum()), strict_top3=int(k.any(1).sum()))
    if [(r['count'], r['strict_top1']) for r in out.values()] != [(526, 54), (1368, 107)]:
        raise ValueError('The fixed masks no longer match the margin control')
    result = dict(result_source='fresh_hindsight_diagnostic', counts=out, freeze=sha(fp), code=sha(Path(__file__)),
        same_intervention_records=True, external_outcomes_read=False, policy_unchanged=True)
    write(OUT/'fixed_mask_capacity.json', result, immutable=True)
    print(result, flush=True)


if __name__ == '__main__':
    main()
