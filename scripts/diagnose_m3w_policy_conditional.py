"""Posthoc accounting of frozen cross-source calibration; never changes actions."""
import argparse
from collections import defaultdict
import json
import numpy as np

from scripts import run_m3w_policy_conditional_calibration as run


def accounting(z, calibrator):
    p,y,e=z['prediction'],z['targets'],z['envelope']
    known=np.isfinite(y).all(1);den=float(y[known,2].sum())
    raw,support,keep=[z['action_'+a] for a in ('raw','support_only','conditional')]
    masks=dict(raw=raw,unsupported_removed=raw & ~support,penalty_removed=support & ~keep,retained=keep)
    if not np.array_equal(raw,masks['unsupported_removed'] | masks['penalty_removed'] | keep):
        raise ValueError('Removal partition does not reconstruct original actions')
    ids=run.api.cells(p,e);offset=np.array([calibrator['table'][str(i)]['offset'] for i in ids])
    estimated=run.api.excess(p)+offset*e[:,None]
    observed=run.api.excess(y)
    result={}
    for name,mask in masks.items():
        at=mask & known;unknown=mask & ~known
        def percent(value):return float(100*value/den) if den>0 else None
        result[name]=dict(rows=int(mask.sum()),known=int(at.sum()),unknown=int(unknown.sum()),
            known_benefit_pp=percent(y[at,0].sum()),known_harm_pp=percent(y[at,1].sum()),
            known_net_utility_pp=percent((y[at,0]-y[at,1]).sum()),
            unknown_envelope_pp=percent(e[unknown].sum()),
            predicted_easy_excess_pp=percent(estimated[at,1].sum()),
            observed_easy_excess_pp=percent(observed[at,1].sum()),
            positive_easy_excess_error_pp=percent(np.maximum(observed[at,1]-estimated[at,1],0).sum()))
    cells=[]
    for c in np.unique(ids[raw]):
        table=calibrator['table'][str(c)];row=dict(cell=int(c),calibration=table)
        for name,mask in (('raw',raw),('retained',keep)):
            at=mask & known & (ids==c) & (e>0)
            mass=float(e[at].sum())
            residual=run.api.excess(y[at])-run.api.excess(p[at])
            row[name]=dict(known=int(at.sum()),unknown=int((mask & ~known & (ids==c)).sum()),
                target_residual_per_envelope=(residual.sum(0)/mass).tolist() if mass>0 else None,
                easy_observed_excess=float(observed[at,1].sum()),
                easy_estimated_excess=float(estimated[at,1].sum()))
        cells.append(row)
    return dict(partition=result,cells=cells,denominator=den,posthoc_not_inference=True)


def build():
    run.admission()
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    complete=json.loads((run.PUBLIC/'complete.json').read_text())
    if complete['decision_freeze_sha256']!=run.sha(run.PUBLIC/'decision_freeze.json'):
        raise ValueError('Diagnostic must follow unchanged completed readout')
    records=[]
    for ref in freeze['receipts']:
        p=run.ROOT/ref['path'];d=json.loads(p.read_text())
        if run.sha(p)!=ref['sha256']:raise ValueError('Changed frozen calibrator')
        cache=run.ROOT/d['cache']['path']
        if run.sha(cache)!=d['cache']['sha256']:raise ValueError('Changed frozen decisions')
        with np.load(cache,allow_pickle=False) as z:
            result=accounting(z,d['calibrator'])
        records.append(dict(context=d['context'],source=d['source'],seed=d['seed'],result=result))
    summary={}
    for mask in ('raw','unsupported_removed','penalty_removed','retained'):
        row={}
        for key in records[0]['result']['partition'][mask]:
            if key in ('rows','known','unknown'):
                row[key]=sum(r['result']['partition'][mask][key] for r in records)
            else:
                by_source=defaultdict(list)
                for r in records:by_source[r['source']].append(r['result']['partition'][mask][key])
                row[key]=float(np.mean([np.mean(v) for v in by_source.values()]))
        summary[mask]=row
    transfer=[]
    for r in records:
        for c in r['result']['cells']:
            v=c['retained'];bias=v['target_residual_per_envelope']
            if not v['known']:continue
            transfer.append(dict(context=r['context'],source=r['source'],seed=r['seed'],cell=c['cell'],
                calibration_easy_residual=c['calibration']['signed_residual'][1],
                applied_easy_offset=c['calibration']['offset'][1],
                target_raw_easy_residual=c['raw']['target_residual_per_envelope'][1],
                target_retained_easy_residual=bias[1],
                observed_easy_excess=v['easy_observed_excess'],estimated_easy_excess=v['easy_estimated_excess'],
                known=v['known'],unknown=v['unknown']))
    harmful=[r for r in transfer if r['observed_easy_excess']>1e-12]
    out=dict(result_source='fresh_run_posthoc_frozen_decision_accounting',
        decision_freeze_sha256=run.sha(run.PUBLIC/'decision_freeze.json'),
        completed_summary_sha256=run.sha(run.PUBLIC/'summary.json'),groups=records,
        mean_locality_partition_pp=summary,
        retained_cell_accounting=dict(known_cells=len(transfer),observed_easy_excess_positive=len(harmful),
            positive_with_zero_offset=sum(r['applied_easy_offset']==0 for r in harmful),
            positive_despite_positive_offset=sum(r['applied_easy_offset']>0 for r in harmful),
            details=transfer),
        actions_changed=False,independent_roles_read=False,deployment_changed=False)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    out=build();path=run.PUBLIC/'failure_accounting.json'
    if a.check:
        if json.loads(path.read_text())!=out:raise ValueError('Posthoc accounting failed replay')
    else:run.once(path,out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('groups','retained_cell_accounting')},indent=2))
    print(json.dumps({k:v for k,v in out['retained_cell_accounting'].items() if k!='details'},indent=2))


if __name__=='__main__':main()
