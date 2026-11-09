"""Describe what the extra, verified joint witnesses actually selected."""

import collections
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as study


def main():
    read = study.read
    raw = read(study.PRIVATE / 'recording_receipt.json')
    bound = study.OUT / 'readout.json'
    results = read(bound)
    if raw['registration'] != results['registration'] or results['independent_confirmation']:
        raise ValueError('Require bound development solver receipts')
    names = (*study.motion.EXPERTS[1:], 'eqmotion', 'pct')
    rows = raw['cells']
    baselines = {(r['recording'], r['cap']): r for r in rows if r['seed'] is None}
    payload = {}
    for cap in study.CAPS:
        counts = collections.Counter()
        new = []
        for r in rows:
            if r['cap'] != cap or r['seed'] is None or r['status'] != 'verified':
                continue
            a = baselines[r['recording'], cap]['status']
            if a in ('unresolved', 'verified'):
                continue
            p = study.PRIVATE / f'{r["recording"]}_{r["seed"]}_{cap}.json'
            cell = read(p)
            if cell['witness'] != r['witness'] or cell['identity']['registration'] != raw['registration']:
                raise ValueError('Witness choice cache changed')
            arms = [names[j] for _, j in cell['actions']]
            if not set(arms) & {'eqmotion', 'pct'}:
                raise ValueError('A pure analytic witness contradicts the negative analytic bound')
            counts.update(arms)
            new.append(r['witness'])
        if len(new) != results['summary']['caps'][str(cap)]['confirmed_new_views']:
            raise ValueError('Confirmed witness total changed')
        bottleneck = [min(w['gain_pp']) for w in new]
        payload[str(cap)] = dict(views=len(new), actions=dict(counts),
            learned_fraction=(counts['eqmotion']+counts['pct'])/sum(counts.values()) if counts else None,
            median_bottleneck_pp=float(study.np.median(bottleneck)) if new else None,
            smallest_bottleneck_pp=min(bottleneck) if new else None,
            largest_bottleneck_pp=max(bottleneck) if new else None)
    study.once(study.OUT / 'confirmed_new_witnesses.json', study.stable(dict(
        parent=study.artifact(bound), scope='descriptive confirmed new witnesses only', results=payload,
        independent_confirmation=False, labels_hindsight_only=True)))
    print(payload)


if __name__ == '__main__':
    main()
