"""Paired matched-coverage contrasts, never policy selection or calibration."""
import json
import os
from pathlib import Path
import platform
import sys

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_source_intervention as study
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg,data,identity,designs,qmask = study.load()
    analysis = json.loads((study.PUBLIC/'analysis.json').read_text())
    verified = json.loads((study.PUBLIC/'verification.json').read_text())
    if verified['analysis_sha256']!=digest(study.PUBLIC/'analysis.json'):
        raise ValueError('Verified decision readout required')
    n = len(data['sites'])
    states = {}
    for key,design,ti in designs:
        if ti['kind']!='complement':
            continue
        held = design['held_ids']
        p = study.prediction(key,held)
        ade,_ = native_errors(p.astype(float)+data['origin'][held,None],data['target_eval'][held],
                              data['valid'][held],np.ones(len(held)))
        floor = np.asarray(data['baseline_ade'][held,design['baseline_index']])
        for arm in reg['arms']:
            path = study.PRIVATE/'decisions'/(key+'_'+arm+'.npz')
            receipt = json.loads(path.with_suffix('.json').read_text())
            if digest(path)!=receipt['sha256']:
                raise ValueError('Decision bytes changed')
            with np.load(path,allow_pickle=False) as z:
                ids = z['ids'].copy()
                np.testing.assert_array_equal(ids,held[qmask[held]])
                row = {k:z[k].copy() for k in ('matched','matched_nonzero',*study.CONTROL_ARMS)}
            keys = np.column_stack((data['recordings'][ids],data['frames'][ids]))
            unique,inverse = np.unique(keys,axis=0,return_inverse=True)
            query_differences = 0
            for i in range(len(unique)):
                use = inverse==i
                if np.any(row['matched'][use])!=np.all(row['matched'][use]):
                    raise ValueError('Inconsistent query matching flag')
                if row['matched'][use].all():
                    counts = [int(row[k][use].sum()) for k in ('independent','unary_exact','joint_exact')]
                    if len(set(counts))!=1 or np.any(row['matched_nonzero'][use]!=(counts[0]>0)):
                        raise ValueError('Actual decisions violate claimed equal-count control')
                query_differences += int(np.any(row['unary_exact'][use]!=row['joint_exact'][use]))
            s = states.setdefault((ti['seed'],arm),dict(
                costs={k:np.full(n,np.nan) for k in ('independent','unary_exact','joint_exact','joint')},
                masks=np.zeros(n,bool),nonzero=np.zeros(n,bool),seen=np.zeros(n,bool),
                query_count=0,changed_joint_vs_unary_queries=0))
            if s['seen'][ids].any():
                raise ValueError('Repeated query readout')
            s['seen'][ids] = True
            s['masks'][ids],s['nonzero'][ids] = row['matched'],row['matched_nonzero']
            for k in s['costs']:
                s['costs'][k][ids] = np.where(row[k],ade[qmask[held]],floor[qmask[held]])
            s['query_count'] += len(unique)
            s['changed_joint_vs_unary_queries'] += query_differences
    roster = sorted(identity['folds'])
    results = {}
    for (seed,arm),s in states.items():
        if not np.array_equal(s['seen'],qmask):
            raise ValueError('Joint population changed')
        comparisons = {}
        for name,left,right,mask in [
            ('joint_vs_independent_full_population','joint','independent',qmask),
            ('joint_exact_vs_independent_matched_nonzero','joint_exact','independent',s['nonzero']),
            ('joint_exact_vs_unary_matched_nonzero','joint_exact','unary_exact',s['nonzero'])]:
            comparisons[name] = paired_scene_metrics(s['costs'][left][mask],s['costs'][right][mask],
                data['sites'][mask],expected_scenes=roster,dataset='EuropeanSquares_released_detector_tracks',
                coordinate_unit='image_pixel',bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
        results[f'{seed}_{arm}'] = dict(comparisons=comparisons,
            query_count=s['query_count'],changed_joint_vs_unary_queries=s['changed_joint_vs_unary_queries'],
            matched_nonzero_rows=int(s['nonzero'].sum()),actual_count_checks_passed=True)
    out = dict(result_source='fresh_run',input_provenance='cached_verified',new_training=False,
        new_model_inference=False,new_policy_selection=False,analysis_sha256=digest(study.PUBLIC/'analysis.json'),
        verifier_sha256=digest(Path(__file__)),results=results,independent_confirmation=False)
    immutable_json(study.PUBLIC/'paired_joint_contrasts.json',out)
    lines = ['# Paired Joint-Decision Contrasts','',
        'Fresh arithmetic on hash-verified frozen predictions and decisions; no refit, threshold change or reserved readout.',
        'Every claimed matched query was rechecked from its actual binary decisions, not just a solver-success label.',
        '', '| Seed / cost head | Comparison | Equal-locality ADE gain (%) | Conditional 95% CI |',
        '|---|---|---:|---|']
    for key,v in results.items():
        for name,m in v['comparisons'].items():
            gain,ci = m['equal_scene_gain_percent'],m['scene_bootstrap_ci95']
            g = 'undefined' if gain is None else f'{gain:.6f}'
            c = 'undefined' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'
            lines.append(f'| {key} | {name} | {g} | {c} |')
    lines += ['', 'Nonzero matching is fixed by predictions before target costs. Undefined ratios indicate missing supported localities or zero reference, not a zero improvement.',
        'The unrestricted joint/independent comparison can differ in coverage; only the exact-count contrasts isolate geometry at the same intervention count.',
        'All intervals are source-development diagnostics with shared predictors, not independent confirmation or a physical-safety claim.','']
    (study.PUBLIC/'paired_joint_contrasts.md').write_text('\n'.join(lines))
    print(json.dumps({k:{name:m['equal_scene_gain_percent'] for name,m in v['comparisons'].items()} for k,v in results.items()}))


if __name__=='__main__':
    main()
