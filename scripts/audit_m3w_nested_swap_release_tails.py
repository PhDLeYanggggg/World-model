"""Read held error tails and easy degradation for the frozen releases."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_nested_swap_release as run

np = run.np
ARMS = ('cv', *run.ARMS)


def measure(pred, cv, y, mask, rec, easy_cut):
    observed = mask.any(1)
    ade = np.divide(np.where(mask, np.linalg.norm(pred-y, axis=-1), 0.).sum(1),
                    mask.sum(1), out=np.zeros(len(pred)), where=observed)
    floor = np.divide(np.where(mask, np.linalg.norm(cv-y, axis=-1), 0.).sum(1),
                      mask.sum(1), out=np.zeros(len(cv)), where=observed)
    endpoint = mask[:, -1]
    easy = observed & (floor > 0.) & (floor <= easy_cut)
    signed, worst_p95 = [], []
    for name in np.unique(rec):
        group = rec == name
        take = group & easy
        if take.any():
            signed.append(float(100.*(ade[take]-floor[take]).sum()/floor[take].sum()))
        keep = group & observed
        if keep.any():
            worst_p95.append(float(np.quantile(ade[keep], .95)))
    return dict(ade=float(ade[observed].mean()), fde=float(np.linalg.norm(pred[endpoint, -1]-y[endpoint, -1], axis=-1).mean()),
                p95=float(np.quantile(ade[observed], .95)), worst_recording_p95=max(worst_p95),
                worst_easy_degradation=max(signed) if signed else None, easy_supported_recordings=len(signed))


def main():
    key, source, cells, _ = run.register()
    p, c, r = (run.OUT/k for k in ('freeze.json', 'calibration_freeze.json', 'readout.json'))
    f, freeze, result = map(run.read, (p, c, r))
    if not f['key'] == freeze['key'] == result['key'] == key or result['calibration'] != run.sha(c):
        raise ValueError('Require scored frozen releases')
    data, features = run.base.parent.load(source), run.run.old.prior.load_features(source)
    verified = {(x['identity']['held'], x['identity']['donor_cal'], x['identity']['seed']): x['arms']
                for x in result['by_direction']}
    rows = []
    for name, cell in cells.items():
        for held in cell['packets']:
            rule = freeze['rules'][f'{name}/{held}']
            identity = held, rule['calibration'], cell['seed']
            old = verified[identity]
            pp, zz = run.opened(held, name, cell, f)
            with pp as packet, zz as actions:
                cv, forecasts = run.candidates(packet, features[held], actions)
                d = data[held]
                rec = d['recording']
                use = np.isin(rec, packet['records'])
                a = dict(cv=cv[use])
                for arm in run.ARMS:
                    threshold = rule['arms'][arm]['threshold']
                    score = packet['additive_scores'] if arm == 'additive' else actions[arm][:, 1]
                    keep = np.ones(len(score), dtype=bool) if arm == 'additive' else actions[arm][:, 0] >= 0.
                    selected = keep & (score >= threshold) if threshold is not None else np.zeros(len(score), dtype=bool)
                    if int(selected.sum()) != old[arm]['released']:
                        raise ValueError('Frozen release count changed')
                    release = np.isin(rec[use], packet['records'][selected])
                    a[arm] = np.where(release[:, None, None], forecasts[arm][use], cv[use])
                rows.append(dict(roles=list(identity), arms={arm: measure(pred, a['cv'], d['future'][use],
                    d['valid'][use], rec[use], cell['cut'][0]) for arm, pred in a.items()}))
    if len(rows) != 126:
        raise ValueError('Required held directions changed')
    summary = {}
    for a in ARMS:
        v = [r['arms'][a] for r in rows]
        summary[a] = {f'direction_mean_{k}': float(np.mean([r[k] for r in v])) for k in ('ade', 'fde', 'p95')}
        for k in ('worst_recording_p95', 'worst_easy_degradation'):
            summary[a][k] = max(r[k] for r in v if r[k] is not None)
    run.write(run.OUT/'tail_audit.json', dict(readout=run.sha(r), calibration=run.sha(c), summary=summary,
        no_threshold_changes=True, external_outcomes_read=False), immutable=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
