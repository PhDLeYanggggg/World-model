"""Absolute, seed-averaged locality costs; no additional selection rule."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_cost_shape as run
from scripts.diagnose_m3w_european_cost_mass import absolute_rows
from scripts.report_m3w_european_cost_mass import summarize


def transport_counts(items):
    out=dict(views=len(items),missing=0,fit_and_held_improved=0,fit_only=0,held_only=0,neither=0)
    for a,b,c,d in items:
        if any(v is None for v in (a,b,c,d)):
            out['missing']+=1; continue
        fit,held=a<b,c<d
        out['fit_and_held_improved' if fit and held else 'fit_only' if fit else 'held_only' if held else 'neither']+=1
    return out


def fitting_transport(rows):
    held={r['group']+'_'+r['pair']+'_'+f['held']:f for r in rows for f in r['folds']}
    items={}
    for path in sorted((run.PRIVATE/'readouts').glob('*/models.json')):
        r=json.loads(path.read_text())
        parent=json.loads((run.previous.PRIVATE/'readouts'/r['tag']/'models.json').read_text())
        h=held[r['tag']]['metrics']
        for arm in ('cost_only','cap_aux','shuffled_aux'):
            for mode in ('shape_L2','shape_mass'):
                for reference in ('raw','scaled','mass'):
                    for j,(col,component) in enumerate(((1,'H_all'),(3,'H_easy'))):
                        key=(r['pair'],arm,mode,reference,component)
                        a=r['models'][arm+'_'+mode]['components'][j]['MSE']
                        b=parent['details'][arm]['equal_locality'][reference][component]['MSE']
                        c,d=[h[arm+'_'+m]['envelope_positive'].get('component_MSE') for m in (mode,reference)]
                        items.setdefault(key,[]).append((a,b,c[col] if c else None,d[col] if d else None))
    result=[]
    for key,values in sorted(items.items()):
        assert len(values)==72
        result.append(dict(zip(('pair','arm','mode','reference','component'),key),**transport_counts(values)))
    return result


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
    transport=fitting_transport(doc['rows']); assert len(transport)==72
    run.immutable_json(run.PUBLIC/'fitting_transport.json',dict(posthoc_descriptive=True,new_gate=False,rows=transport,
        caveat='Fitting equal-locality MSE versus held-locality row MSE;dependent view counts,not inferential samples'))
    lines=['# Fitting Gains and Held-Locality Transport','',
        'Post-hoc descriptive counts,not another selection criterion or gate.',
        'Each row includes72 dependent seed/locality views. Fitting is equal-locality weighted;held is row-weighted within its locality.',
        'Both compare the same supported positive-envelope cost target within their own role. Equalities count as no improvement.',
        'These counts do not identify the cause of a failed transport or establish independent confirmation.','',
        '| Family/arm/new/reference/component | Both improve | Fit only | Held only | Neither | Missing |','|---|---:|---:|---:|---:|---:|']
    for r in transport:
        name='/'.join(r[k] for k in ('pair','arm','mode','reference','component'))
        lines.append('| '+name+' | '+' | '.join(str(r[k]) for k in ('fit_and_held_improved','fit_only','held_only','neither','missing'))+' |')
    (run.PUBLIC/'fitting_transport.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
