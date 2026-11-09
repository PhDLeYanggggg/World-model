"""Label-aware two-edit capacity at the exact frozen queries and radii."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_support as run
from src.evaluation.m3w_recording_switch_ceiling import moments

base, old, np, torch = run.base, run.old, run.np, run.torch


def witness(rows, risk, reference, gain, signed_cap, ols_offset):
    i, j = np.triu_indices(len(rows), 1)
    b = risk[i]+risk[j]
    u = gain[i]+gain[j]
    u[:, 1] += ols_offset
    ok = ((rows[i] != rows[j]) & ((reference[i]+reference[j]) > 0.)
          & (b <= [0., 0., signed_cap]).all(1) & (u > 0.).all(1))
    if not ok.any():
        return None
    k = int(np.where(ok, u.min(1), -np.inf).argmax())
    return np.array([i[k], j[k]])


def capacity(z, d, paths, cut):
    cv, ols = z['cv'], z['ols4']
    y, valid, w = d['future'], d['valid'], d['weight']
    rec, ix = np.unique(d['recording'], return_inverse=True)
    m = moments(cv, cv, y, valid, cut[0])
    easy_mass = np.bincount(ix, weights=m['ref_easy'])
    before = base.parent.paired(ols, cv, y, valid)[1].sum(1)
    ols_offset = np.bincount(ix, weights=before*w)
    result, totals = cv.copy(), []
    for key, (lo, hi) in zip(z['records'], z['bounds'], strict=True):
        q = slice(lo, hi)
        peak = np.linalg.norm(paths[q]-cv[q, None], axis=-1).max(2)
        row, family = np.nonzero(peak > 0.)
        row += lo
        p = paths[row, family]
        mm = moments(cv[row], p, y[row], valid[row], cut[0])
        risk = np.column_stack([mm[k] for k in ('all_excess', 'easy_excess', 'easy_signed')])
        gain = np.column_stack((base.parent.paired(cv[row], p, y[row], valid[row])[1].sum(1),
                                base.parent.paired(ols[row], p, y[row], valid[row])[1].sum(1)-before[row]))
        r = np.searchsorted(rec, key)
        selected = witness(row, risk, mm['ref_easy'], gain*w[row, None], .02*easy_mass[r], ols_offset[r])
        if selected is not None:
            result[row[selected]] = p[selected]
            totals.append(key)
    if totals:
        check = run.evaluate(result, dict(cv=cv, ols4=ols, records=np.array(totals)), d, cut)
        if not check['dual'].all():
            raise ValueError('Hindsight witness failed unchanged risk or gain scorer')
    return len(totals)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, source, _, _, forecasts = run.register()
    freeze = old.read(run.OUT / 'freeze.json')
    readout = old.read(run.OUT / 'readout.json')
    if key != freeze['registration'] or readout['freeze'] != base.prior.sha(run.OUT / 'freeze.json'):
        raise ValueError('Missing frozen scored comparison')
    data = base.parent.load(source)
    records = []
    for name, c in freeze['cells'].items():
        pair, seed = list(c['packets']), c['seed']
        tag = old.double_name(pair, seed)
        net = base.load_net(old.LOCAL / f'{tag}.pt', forecasts['double_excluded'][tag],
                            old.motion.motion.network(17), pair)
        for site, r in c['packets'].items():
            path = run.LOCAL / r['file']
            if base.prior.sha(path) != r['sha256']:
                raise ValueError('Frozen packet changed')
            d = data[site]
            cv, ols, _, paths, *_ = old.menu(d, old.forecast(net, d))
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(cv, z['cv'])
                np.testing.assert_array_equal(ols, z['ols4'])
                n = capacity(z, d, paths, c['cut'])
                records.append(dict(cell=name, site=site, eligible=len(z['records']), verified_witnesses=n))
        run.beat(phase='hindsight', cell=name, count=len(records))
    total = sum(x['verified_witnesses'] for x in records)
    eligible = sum(x['eligible'] for x in records)
    if (eligible != readout['summary']['joint']['matched_recordings']
            or total < max(v['matched_dual'] for v in readout['summary'].values())):
        raise ValueError('Hindsight capacity contradicts matched choices')
    old.write(run.OUT / 'hindsight.json', dict(verified_witnesses=total, eligible=eligible, views=records,
        readout=base.prior.sha(run.OUT / 'readout.json'), label_aware_not_deployable=True, changed_thresholds=False))


if __name__ == '__main__':
    main()
