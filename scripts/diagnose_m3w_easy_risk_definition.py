"""Fixed post-decision decomposition before registering a new target experiment."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_easy_allocation_guarded import load
from scripts.run_m3w_easy_allocation import causal_view, ARMS
from scripts.run_m3w_native_forecast import immutable_json,file_digest,assert_current
from src.evaluation.m3w_easy_risk_definition import sums
import numpy as np
import torch


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();cfg,data,context,parent,tr,eq,identity=pack
    previous_path=ROOT/cfg['reports']/'analysis.json';previous=json.loads(previous_path.read_text())
    assert file_digest(previous_path)=='73dc1e392efe1bc6b61569d474e09158284118da13bebc9625ce317e078a6956'
    records=[];bins=[]
    for key in tr:
        seed=int(key.rsplit('seed',1)[1]);site=key.rsplit('_seed',1)[0]
        for action in cfg['actions']:
            values,pred,scale,cut=causal_view(pack,key,action);ids=values['ids']
            source=next(r for r in parent['outcome_archives'] if r['path'].endswith(f'{action}_seed{seed}.npz'))
            assert file_digest(ROOT/source['path'])==source['sha256']
            with np.load(ROOT/source['path'],allow_pickle=False) as z:
                b,e,k=z['cv'][ids],z['candidate_ade'][ids],z['complete'][ids]
            decisions=next(r for r in previous['outcome_archives'] if r['path'].endswith(f'{action}_seed{seed}.npz'))
            assert file_digest(ROOT/decisions['path'])==decisions['sha256']
            with np.load(ROOT/decisions['path'],allow_pickle=False) as z:choices=z['choices'][ids].copy()
            q=values[action+'__moments'][:,0]*values[action+'__distance']
            r=values[action+'__moments'][:,1]*cut
            for p in ('net_stop','strict_stop','pointwise','aggregate_selected','aggregate_population','aggregate_joint'):
                records.append(dict(view=key,site=site,seed=seed,action=action,policy=p,cutoff=cut,
                    **sums(b,e,k,choices[:,ARMS.index(p)],q,r,cut)))
            ratio=np.divide(q,r,out=np.full(len(q),np.inf),where=r>0)
            breaks=[0,.02,.1,.5,1,np.inf]
            for low,high in zip(breaks[:-1],breaks[1:]):
                use=values[action+'__net_stop']&(ratio>=low)&(ratio<high)
                bins.append(dict(view=key,action=action,lower=low,upper=None if np.isinf(high) else high,
                    **sums(b,e,k,use,q,r,cut)))
            print(json.dumps(dict(view=key,action=action,complete_rows=int(k.sum()))),flush=True)
    assert_current(identity)
    out=ROOT/'outputs/publication_readiness_2026_09/easy_risk_definition_diagnosis_v1'
    result=dict(result_source='fresh_postdecision_development_diagnosis_no_policy_change',
        parent_analysis_sha256=file_digest(previous_path),
        source_bindings={p:file_digest(ROOT/p) for p in ('scripts/diagnose_m3w_easy_risk_definition.py',
            'src/evaluation/m3w_easy_risk_definition.py','tests/test_m3w_easy_risk_definition.py')},
        records=records,causal_ratio_bins=bins,independent_calibration=False,
        new_training=False,threshold_selection=False,external_readout=False,
        deployment=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(out/'analysis.json',result)
    with (out/'reliability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    def val(x):return 'undefined' if x is None else f'{x:.3f}'
    lines=['# Easy-Risk Definition and Reliability Diagnosis','',
        'Fresh post-decision development analysis on frozen outputs, not calibration or threshold selection.',
        'Only complete futures contribute observed costs. Incomplete selections remain separately counted.',
        'Positive harm does not subtract improvements; net harm does. A negative net ratio means improvement.',
        'No policy changes or new external readout. Four exposed SDD sites, obs8/pred12 annotation pixels.','',
        '| View / predictor | Strict predicted positive selected ratio % | Observed positive selected ratio % | Observed net selected ratio % | Predicted / observed harm | Benefit / positive harm % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        if r['policy']=='strict_stop':
            lines.append('| '+r['view']+' / '+r['action']+' | '+' | '.join(val(r[k]) for k in (
                'predicted_positive_selected_ratio','observed_positive_selected_ratio','observed_net_selected_ratio',
                'predicted_to_observed_harm','benefit_cancels_positive_harm_percent'))+' |')
    lines+=['','## Interpretation Boundaries','',
        'Scene-balanced training draws are uniform within supported complete rows, not hard-label oversampling.',
        'Complete-label selection and cross-scene shift remain; this cannot identify unconditional risk on missing labels.',
        'Selected-set ratios in this table do not equal population easy degradation. Both denominators are in the CSV.',
        'Ratios pooled across sites would mix pixel scales and are not reported. Zero denominators remain undefined.',
        'Use this diagnostic to register a falsifiable target comparison, not to tune a threshold or promote a policy.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,policy_rows=len(records),bin_rows=len(bins),analysis_sha256=file_digest(out/'analysis.json'))))


if __name__=='__main__':main()
