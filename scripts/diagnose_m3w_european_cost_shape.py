"""Absolute, seed-averaged locality costs; no additional selection rule."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_cost_shape as run
from scripts.diagnose_m3w_european_cost_mass import absolute_rows
from scripts.report_m3w_european_cost_mass import summarize


def main():
    cfg=json.loads((ROOT/run.CONFIG).read_text())
    doc=json.loads((run.PUBLIC/'readout.json').read_text())
    rows=absolute_rows(doc['rows'],cfg['seeds']); assert len(rows)==1440
    with (run.PUBLIC/'absolute_costs.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for pair in cfg['pairs']:
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['pair']==pair and r['method']==method and r['subset']=='envelope_positive']
            assert len(selected)==24
            summary.append(dict(pair=pair,method=method,dependent_locality_views=24,
                MSE=summarize([r['seed_mean_harm_MSE'] for r in selected]),
                coverage=summarize([r['seed_mean_harm_coverage'] for r in selected])))
    run.immutable_json(run.PUBLIC/'absolute_cost_summary.json',dict(rows=1440,summary=summary,new_gate=False,
        averaging='three seed means within locality;coverage is mean of ratios,not pooled ratio'))
    lines=['# Absolute Expected-Cost Context','','All1440 locality/method/subset rows are retained in absolute_costs.csv.',
        'Three seeds averaged within locality;repeated assignments are dependent. Missing support is not dropped.',
        'Coverage is the mean of seed ratios,not a pooled ratio. These are expected-cost squared errors,not trajectory ADE/FDE.',
        'Large relative losses can reflect small comparator MSE;registered metrics and gates are unchanged.','',
        '| Family/method | MSE min / median / max | Coverage min / median / max |','|---|---|---|']
    for r in summary:
        lines.append(f"| {r['pair']}/{r['method']} | {[r['MSE'][k] for k in ('minimum','median','maximum')]} | {[r['coverage'][k] for k in ('minimum','median','maximum')]} |")
    (run.PUBLIC/'absolute_cost_context.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
