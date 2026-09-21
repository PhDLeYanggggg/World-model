"""Verify cost summaries and replay fixed first/middle/last original OOF batches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('transformer','eqmotion'), required=True)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Use native arm64 Python')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import numpy as np
    import torch
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.evaluation.m3w_frozen_runtime import prepare_runtime, RelocatedCodeContract
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/'configs/m3w_cost_head_fit_forensics_v1.json').read_text())
    parent = json.loads((ROOT/cfg['parent_config']).read_text())
    protocol = json.loads((ROOT/parent['protocol']).read_text())
    if file_digest(ROOT/parent['protocol']) != parent['protocol_file_sha256']:
        raise ValueError('Protocol changed')
    runtime = prepare_runtime(ROOT,protocol,ROOT/parent['output']/'frozen_runtime',
        revision=parent['historical_model_revision'],install=True)
    from src.world_model.m3w_neural_gain_harm import read_verified_oof_cache
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, collate_forecasts, load_verified_forecaster, risk_features
    private = ROOT/cfg['private_output']/args.family
    public = ROOT/cfg['output']
    completion = json.loads((private/'completion.json').read_text())
    report_path = public/(args.family+'.json')
    if file_digest(report_path) != completion['report_sha256']:
        raise ValueError('Completed diagnosis changed')
    report = json.loads(report_path.read_text())
    for p,digest in report['source_bindings'].items():
        if file_digest(ROOT/p) != digest:
            raise ValueError('Source binding changed')
    for r in completion['receipts']:
        if file_digest(ROOT/r['path']) != r['sha256']:
            raise ValueError('Diagnostic receipt or array changed')
    checked_summaries, checks = 0, []
    # Independent scalar reductions, without the production diagnosis helper.
    for record in report['results']:
        with np.load(private/f"seed{record['seed']}_{record['head']}.npz",allow_pickle=False) as a:
            y,p = a['target'].astype(np.float64),a['predicted'].astype(np.float64)
            masks = {'all':np.ones(len(y),bool)}
            if 'raw_ridge' in a.files:
                raw=a['raw_ridge']
                np.testing.assert_array_equal(np.maximum(raw,0),a['predicted'])
                masks['ridge_harm_clipped_to_zero']=raw[:,1]<0
                masks['ridge_benefit_clipped_to_zero']=raw[:,0]<0
            for name,rule in protocol['development_evaluation']['policies'].items():
                # Preserve the original float32 membership calculation.
                q=a['predicted']
                masks['eligible_'+name]=(q[:,0]-q[:,1]>=rule['min_predicted_gain'])&(q[:,1]<=rule['max_agent_predicted_harm'])
            for name,mask in masks.items():
                expected=record['conditional'][name]
                if int(mask.sum()) != expected['rows']:
                    raise ValueError('Conditional count changed')
                if mask.any():
                    actual = {'target_harm_mean':y[mask,1].sum()/mask.sum(),
                        'predicted_harm_mean':p[mask,1].sum()/mask.sum(),
                        'actual_net_gain_mean':(y[mask,0].sum()-y[mask,1].sum())/mask.sum(),
                        'predicted_net_gain_mean':(p[mask,0].sum()-p[mask,1].sum())/mask.sum(),
                        'harm_mse':np.square(p[mask,1]-y[mask,1]).sum()/mask.sum()}
                    for key,value in actual.items():
                        np.testing.assert_allclose(value,expected[key],rtol=1e-12,atol=1e-12)
                checked_summaries+=1
    device=cfg['families'][args.family]
    study=ROOT/f'data/stage_cvpr2027_experiments/8to12_{args.family}_v6'
    began=time.monotonic()
    for seed in cfg['seeds']:
        artifacts=[json.loads((study/f'seed{seed}_{name}/artifact.json').read_text())
                   for name in ('full','hold0','hold1','hold2','ridge','neural_cost')]
        contract=RelocatedCodeContract(protocol,ROOT,artifacts,runtime=runtime)
        groups=read_verified_oof_cache(contract,study/f'seed{seed}_ridge')
        for fold,group in enumerate(groups):
            names=sorted({r['recording_id'] for r in group['identities']})
            contract.assert_prediction_use(group['predictor_id'],names,purpose='oof_risk_training')
            dataset=ContractForecastDataset(contract,names,purpose='fit',baseline_name=group['baseline_name'])
            if len(dataset)!=len(group['targets']):
                raise ValueError('Complete OOF label population changed')
            model=load_verified_forecaster(contract,group['predictor_id'],device=device)
            size=128
            starts=sorted({0,((len(dataset)//2)//size)*size,((len(dataset)-1)//size)*size})
            for start in starts:
                end=min(start+size,len(dataset))
                batch=collate_forecasts([dataset[i] for i in range(start,end)])
                if batch['identities']!=group['identities'][start:end]:
                    raise ValueError('Original batch identity alignment changed')
                inputs={k:v.to(device) for k,v in batch['inputs'].items()}
                with torch.no_grad():
                    candidate=model(inputs)
                    features=risk_features(inputs,candidate).cpu().numpy()
                np.testing.assert_array_equal(features,group['features'][start:end])
                b=inputs['baseline'].cpu().numpy().astype(np.float64)
                c=candidate.cpu().numpy().astype(np.float64)
                y=batch['target'].numpy().astype(np.float64)
                if not batch['target_mask'].all() or not inputs['request_mask'].all():
                    raise ValueError('Expected original complete requested paths')
                gain=np.linalg.norm(b-y,axis=-1).mean(1)-np.linalg.norm(c-y,axis=-1).mean(1)
                independently_rebuilt=np.column_stack((np.maximum(gain,0),np.maximum(-gain,0))).astype(np.float32)
                np.testing.assert_array_equal(independently_rebuilt,group['targets'][start:end])
                checks.append(dict(seed=seed,fold=fold,batch_start=start,rows=end-start,
                    exact_feature_replay=True,exact_independent_target_replay=True,device=device))
                print(json.dumps(dict(family=args.family,**checks[-1],elapsed_seconds=time.monotonic()-began)),flush=True)
    result=dict(result_source='fresh_run_original_OOF_batch_verification_cached_verified_inputs',
        family=args.family,analysis_sha256=file_digest(report_path),code_sha256=file_digest(Path(__file__)),
        independent_scalar_summaries=checked_summaries,original_batch_checks=checks,
        exact_replayed_rows=sum(c['rows'] for c in checks),elapsed_seconds=time.monotonic()-began,
        new_training=False,new_test_data=False,policy_selection=False,
        scope='fixed_first_middle_last_original_batch_per_fit_fold_not_all_raw_rows')
    path=public/(args.family+'_verification.json')
    if path.exists():
        old=json.loads(path.read_text())
        if {k:v for k,v in old.items() if k!='elapsed_seconds'}!={k:v for k,v in result.items() if k!='elapsed_seconds'}:
            raise ValueError('Existing verification changed')
    else:
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='verification_complete',family=args.family,rows=result['exact_replayed_rows'],
        summaries=checked_summaries)),flush=True)


if __name__ == '__main__':
    main()
