"""Independent scalar aggregation of frozen label-support diagnostic reports."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_label_support_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(rows, value, seed=40091, draws=3000):
    per_site = {}
    for row in rows:
        v = value(row)
        if v is not None: per_site.setdefault(row['source'], []).append(v)
    sites = {k:math.fsum(v)/len(v) for k,v in sorted(per_site.items())}
    if len(sites) < 2:
        return dict(localities=sites,mean=None,CI95=None,status='insufficient_localities')
    a = list(sites.values())
    indices = np.random.default_rng(seed).integers(0,len(a),(draws,len(a)))
    means = [math.fsum(a[int(i)] for i in ix)/len(a) for ix in indices]
    return dict(localities=sites,mean=math.fsum(a)/len(a),CI95=np.quantile(means,[.025,.975]).tolist(),
                bootstrap_draws=draws,status='nominal_exposed_development_association_not_causal')


def check(a,b):
    if isinstance(a,dict):
        assert a.keys() == b.keys()
        return sum(check(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a) == len(b)
        return sum(check(x,y) for x,y in zip(a,b))
    if isinstance(a,float): assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9), (a,b)
    else: assert a == b, (a,b)
    return 1


def main():
    complete = json.loads((PUBLIC/'complete.json').read_text())
    replay = json.loads((PUBLIC/'replay.json').read_text())
    assert sha(PUBLIC/'summary.json') == complete['summary_sha256'] == replay['summary_sha256']
    assert replay['raw_and_readout_exact'] and replay['groups'] == complete['groups']
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for path,h in reg['bindings'].items(): assert sha(ROOT/path) == h
    summary = json.loads((PUBLIC/'summary.json').read_text())
    rows = []
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    assert len(rows) == 72 and len({r['source'] for r in rows}) == 12
    total = {}; checks = 0
    for key in ('unknown','one_to_three','four_to_eleven','all_twelve'):
        parts = [r['cohort']['strata'][key] for r in rows]
        result = {k:math.fsum(p[k] for p in parts) for k in parts[0] if k != 'easy_harm_ratio'}
        result['easy_harm_ratio'] = result['easy_harm']/result['easy_reference'] if result['easy_reference'] > 0 else None
        for p in parts:
            assert p['selected'] <= p['rows']
            assert p['harmful']+p['nonharmful'] == p['known_selected'] <= p['selected']
            assert p['leave_one_out_fragile_harmful'] <= p['leave_one_out_defined_harmful'] <= p['harmful']
            assert p['early_late_opposite_harmful'] <= p['early_late_defined_harmful'] <= p['harmful']
            assert p['easy_harm'] <= p['harm'] and p['easy_reference'] <= p['reference']
            checks += 5
        checks += check(result,summary['repeated_occurrence_strata'][key]); total[key] = result
    def share(r):
        p = r['cohort']['strata']; den=math.fsum(v['harm'] for v in p.values())
        return p['all_twelve']['harm']/den if den > 0 else None
    checks += check(interval(rows,share),summary['full_label_share_of_selected_harm'])
    for proxy,v in summary['contrasts'].items():
        for mode,expected in v.items():
            checks += check(interval(rows,lambda r:r['contrasts'][proxy][mode]),expected)
    # These counts were established before this diagnostic, not fitted to it.
    assert sum(v['rows'] for v in total.values()) == 596988
    assert sum(v['selected'] for v in total.values()) == 95455
    assert sum(v['harmful'] for v in total.values()) == 12733
    assert total['unknown']['selected'] == 918
    raw = json.loads((PUBLIC/'raw_label_receipt.json').read_text())
    assert sha(PUBLIC/'raw_label_receipt.json') == complete['raw_receipt_sha256']
    assert len(raw['records']) == 163 and sum(r['rows'] for r in raw['records']) == 318969
    assert sum(r['valid_label_boxes'] for r in raw['records']) == raw['valid_label_boxes']
    receipt = dict(independent_scalar_fields_checked=checks,parent_occurrence_anchors_checked=4,
        raw_records=163,raw_rows=318969,raw_masks_and_coordinates_replayed=True,
        full_inference_and_readout_replayed=True,summary_sha256=sha(PUBLIC/'summary.json'),
        complete_sha256=sha(PUBLIC/'complete.json'),replay_sha256=sha(PUBLIC/'replay.json'),
        independent_confirmation=False,training=False,deployment_changed=False)
    path=PUBLIC/'verification.json'; payload=json.dumps(receipt,indent=2)+'\n'
    if path.exists(): assert path.read_text() == payload
    else: path.write_text(payload)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
