"""Separate original risk failures from utility-only failures in rescued views."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_frontier as study


def report():
    r = study.run
    parent = study.OUT / 'readout.json'
    readout, key = r.read(parent), r.sha(study.OUT / 'registration.json')
    if readout['registration'] != key or readout['external_outcome_access'] or not readout['original_2pct']:
        raise ValueError('Require unchanged exchange-frontier receipts')
    totals, sites = Counter(), {}
    f = study.FLOORS['positive']
    for site, group in readout['by_site'].items():
        local = Counter()
        for c in group:
            for i in range(c['views']):
                row = r.read(study.PRIVATE / f"{site}_{c['seed']}_{i:04d}.json")
                if row['identity'] != dict(registration=key, site=site, seed=c['seed'], ordinal=i):
                    raise ValueError('Private frontier receipt changed')
                zero, one, all_edits = (row['results'][k] for k in ('zero', 'one', 'unrestricted'))
                if (one['lower_pp'] or 0.) > f and not row['anchor_dual']:
                    local['rescued_failed_risk' if zero['infeasible'] else 'rescued_failed_utility'] += 1
                if (all_edits['lower_pp'] or 0.) > f and (one['lower_pp'] or 0.) <= f:
                    needs_more = one['infeasible'] or (one['upper_pp'] is not None and one['upper_pp'] <= f)
                    local['confirmed_need_more_than_one' if needs_more else 'one_exchange_unresolved'] += 1
        totals.update(local)
        sites[site] = dict(local)
    s = readout['summary']['bounds']
    if (totals['rescued_failed_risk'] + totals['rescued_failed_utility'] != s['one']['positive'][0] - s['zero']['positive'][0]
            or totals['confirmed_need_more_than_one'] + totals['one_exchange_unresolved'] != s['unrestricted']['positive'][0] - s['one']['positive'][0]):
        raise ValueError('Failure categories do not partition the verified differences')
    out = dict(readout_sha256=r.sha(parent), summary=dict(totals), by_site=sites, original_2pct=True,
               external_outcome_access=False, hindsight_only=True)
    r.once(study.OUT / 'failure_decomposition.json', r.stable(out))
    return out


if __name__ == '__main__':
    print(report()['summary'], flush=True)
