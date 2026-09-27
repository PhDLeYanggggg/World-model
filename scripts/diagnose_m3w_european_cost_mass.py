"""Post-hoc absolute costs alongside registered relative contrasts; no new gate."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cost_mass as run
import numpy as np


def finite_mean(values):
    return float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None


def absolute_rows(rows, seeds):
    out = []
    for pair, producer, controller in sorted({(r['pair'],r['producer'],r['controller']) for r in rows}):
        group = sorted([r for r in rows if (r['pair'],r['producer'],r['controller']) ==
            (pair,producer,controller)],key=lambda r:r['seed'])
        assert [r['seed'] for r in group] == seeds
        sites = sorted(f['held'] for f in group[0]['folds'])
        assert all(sorted(f['held'] for f in r['folds']) == sites for r in group)
        for site in sites:
            folds = [next(f for f in r['folds'] if f['held'] == site) for r in group]
            keys = sorted(folds[0]['metrics'])
            assert all(sorted(f['metrics']) == keys for f in folds)
            for method in keys:
                for subset in ('all','envelope_positive'):
                    v = [f['metrics'][method][subset] for f in folds]
                    z = dict(pair=pair,producer=producer,controller=controller,locality=site,
                        method=method,subset=subset,seeds=len(seeds))
                    for key in ('rows','positive','harm_MSE','actual_harm_mean','predicted_harm_mean','harm_coverage'):
                        z['seed_mean_'+key] = finite_mean([m.get(key) for m in v])
                    z['all_seeds_supported'] = all(m.get('status') != 'not_estimable' for m in v)
                    out.append(z)
    return out


def main():
    cfg = json.loads((ROOT/run.CONFIG).read_text())
    doc = json.loads((run.PUBLIC/'readout.json').read_text())
    rows = absolute_rows(doc['rows'],cfg['seeds']); assert len(rows) == 864
    with (run.PUBLIC/'absolute_costs.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for family in cfg['pairs']:
        for method in sorted({r['method'] for r in rows}):
            z = [r for r in rows if r['pair'] == family and r['method'] == method and r['subset'] == 'envelope_positive']
            assert len(z) == 24
            vals = [r['seed_mean_harm_MSE'] for r in z]
            summary.append(dict(pair=family,method=method,dependent_locality_views=24,
                missing=sum(v is None for v in vals),MSE_min=min(vals) if None not in vals else None,
                MSE_median=float(np.median(vals)) if None not in vals else None,
                MSE_max=max(vals) if None not in vals else None))
    run.immutable_json(run.PUBLIC/'absolute_cost_summary.json',dict(
        posthoc_descriptive=True,new_gate=False,rows=864,summary=summary,
        averaging='three seed means within each fixed locality; coverage is mean of seed ratios, not pooled ratio'))
    lines = ['# Absolute Cost Context','','Post-hoc descriptive context; no model selection or replacement gate.',
        'All864 locality/method/subset rows are retained in absolute_costs.csv. Unknown fields stay empty.',
        'Means average three seeds within locality. Repeated assignments are dependent.',
        'Coverage is the mean of seed-specific ratios, not the ratio of pooled means.',
        'Large relative losses can result from small comparator MSE; the registered ratios remain unchanged.',
        'These expected-cost squared errors are not trajectory FDE/ADE or physical-safety measurements.','',
        '| Family/method | MSE min / median / max across24 dependent locality views | Missing |','|---|---|---:|']
    for r in summary:
        lines.append(f"| {r['pair']}/{r['method']} | {[r[k] for k in ('MSE_min','MSE_median','MSE_max')]} | {r['missing']} |")
    (run.PUBLIC/'absolute_cost_context.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
