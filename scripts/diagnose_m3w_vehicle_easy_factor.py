"""Attribute failures of the frozen OpenDD easy-support policies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_easy_factor as exp

np, old = exp.np, exp.old


def risks(records):
    fields = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')
    a, e, s = [np.array([r[k] for r in records], dtype=float) for k in fields]
    return dict(undefined_all=~np.isfinite(a), undefined_easy=~np.isfinite(e),
                undefined_signed=~np.isfinite(s), excess_all=a > .02,
                excess_easy=e > .02, excess_signed=s > .02)


def main():
    key, source, _ = exp.register()
    frozen = old.read(exp.OUT / 'freeze.json')
    result = old.read(exp.OUT / 'readout.json')
    if result['freeze'] != exp.prior.sha(exp.OUT / 'freeze.json') or result['registration'] != key:
        raise ValueError('Readout identity changed')
    data = exp.parent.load(source)
    breakdown = {arm: {} for arm in exp.ARMS}
    unique = {arm: set() for arm in exp.ARMS}
    for cell in frozen['cells'].values():
        for site, ref in cell['directions'].items():
            d = data[site]
            packet = exp.LOCAL / ref['file']
            if exp.prior.sha(packet) != ref['sha256']:
                raise ValueError('Frozen packet changed')
            with np.load(packet, allow_pickle=False) as z:
                cv = z['cv']
                for arm in exp.ARMS:
                    rows, path, record, scores = [
                        z[f'{arm}_{k}'] for k in ('rows', 'paths', 'record', 'scores')]
                    t = cell['calibration'][site]['arms'][arm]['threshold']
                    matched = np.ones(len(rows), dtype=bool)
                    released = scores >= t if t is not None else ~matched
                    pred = cv.copy()
                    pred[rows] = path
                    r = exp.policy(pred, cv, cv, d['future'], d['valid'], d['recording'],
                                   easy_cut=cell['cut'][0], hard_cut=cell['cut'][1])
                    recs = [r['recordings'][k] for k in record]
                    flags = risks(recs)
                    supported = np.array([r['supported'] for r in recs], dtype=bool)
                    np.testing.assert_array_equal(supported, ~np.logical_or.reduce(list(flags.values())))
                    flags['risk_pass'] = supported
                    if any(r['selected'] != 1 for r in recs):
                        raise ValueError('Expected exactly one edit per matched recording')
                    unique[arm].update((site, str(v)) for v in record[released])
                    b = breakdown[arm].setdefault(site, {})
                    for mode, used in (('matched', matched), ('released', released)):
                        sums = b.setdefault(mode, {})
                        for k, v in dict(edits=used, **flags).items():
                            sums[k] = sums.get(k, 0) + int((used & v).sum())
    totals = {}
    for arm, sites in breakdown.items():
        totals[arm] = {}
        for mode, ref in (('matched', 'matched'), ('released', 'calibrated')):
            sums = {k: sum(v[mode][k] for v in sites.values()) for k in next(iter(sites.values()))[mode]}
            if (sums['edits'] != result['summary'][arm][f'{ref}_edits']
                    or sums['risk_pass'] != result['summary'][arm][f'{ref}_risk']):
                raise ValueError('Attribution does not reproduce the frozen readout')
            totals[arm][mode] = sums
    old.write(exp.OUT / 'diagnosis.json', dict(
        freeze=result['freeze'], readout=exp.prior.sha(exp.OUT / 'readout.json'),
        result_source='fresh_run', flags_overlap=True, summary=totals, by_site=breakdown,
        unique_released_recordings={k: len(v) for k, v in unique.items()},
        score_and_thresholds_changed=False, external_data_read=False))


if __name__ == '__main__':
    main()
