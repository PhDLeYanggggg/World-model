"""Matched-budget real CNN appearance probes on frozen fit-scene folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',required=True,type=Path)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--report-dir',required=True,type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--only-arm',choices=('geometry','current_rgb','past_rgb'))
    parser.add_argument('--only-fold',type=int)
    parser.add_argument('--only-seed',type=int)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use arm64 .venv-pytorch, not Rosetta/Intel Conda')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe, fit_probe, pixel_delta_to_native
    from src.evaluation.m3w_experiment_contract import ExperimentContract,file_digest
    from src.evaluation.m3w_stationary_scene_context import forecast_metrics
    from src.evaluation.m3w_stationary_start_probe import score_probabilities
    from scripts.summarize_m3w_stationary_probe import group_balanced_brier
    from scripts.run_m3w_stationary_start_probe import atomic_json

    reg = json.loads(args.registration.read_text())
    for p,h in reg['bindings'].items():
        if file_digest(ROOT/p) != h:
            raise ValueError('Changed source: '+p)
    parent = ExperimentContract(json.loads((ROOT/reg['parent_protocol']).read_text()),ROOT)
    if parent.digest != reg['parent_protocol_sha256']:
        raise ValueError('Parent changed')
    cache_file = args.cache/'past_inputs.npz'
    receipt = json.loads((args.cache/'receipt.json').read_text())
    if receipt['registration_sha256'] != file_digest(args.registration) or receipt['cache_sha256'] != file_digest(cache_file):
        raise ValueError('Input cache provenance changed')
    source = ROOT/reg['source_cache']
    if file_digest(source) != reg['source_cache_sha256']:
        raise ValueError('Label-source cache changed')
    with np.load(cache_file,allow_pickle=False) as a:
        x, images, mask, xy, h = [a[k].copy() for k in ('geometry','rgb','mask','image_xy','homography')]
        rows = json.loads(str(a['rows_json']))
        frame_ids = a['frame_ids'].copy()
    with np.load(source,allow_pickle=False) as a:
        native, scale, y = [a[k].copy() for k in ('native','parent_scale','start')]
        original_rows = json.loads(str(a['rows_json']))
    for i,r in enumerate(rows):
        if (any(r[k] != original_rows[i][k] for k in r) or r['data_role']!='fit'
                or parent.protocol['assignments'][r['recording_id']]!='fit'
                or frame_ids[i,-1] != r['frame_id'] or np.any(frame_ids[i] > r['frame_id'])):
            raise ValueError('Input/label alignment or data role mismatch')
    output,reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments') or not reports.is_relative_to(ROOT) or reports.exists():
        raise ValueError('Ignored local output and new report directory required')
    identity = {'registration_sha256':file_digest(args.registration),'cache_sha256':file_digest(cache_file),
        'source_labels_sha256':file_digest(source),'torch':str(torch.__version__),'numpy':np.__version__,
        'runtime':'arm64_cpu_threads4_interop1_workers0','parent_protocol_sha256':parent.digest}
    if args.resume:
        if json.loads((output/'run_identity.json').read_text()) != identity:
            raise ValueError('Run identity changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'run_identity.json',identity)
    started = time.monotonic()
    data = {'images':torch.from_numpy(images.astype(np.float32)/255-.5),
        'mask':torch.from_numpy(mask.astype(np.float32)), 'image_xy':torch.from_numpy(xy),
        'homography':torch.from_numpy(h),'target_native':torch.from_numpy(native),
        'parent_scale':torch.from_numpy(scale),'start':torch.from_numpy(y.astype(np.float32))}
    folds = np.array([r['fit_fold'] for r in rows])
    easy = parent.protocol['development_evaluation']['easy_threshold']
    trials,partial = [],[]
    for fold in sorted(set(folds)):
        fold = int(fold)
        if args.only_fold is not None and fold != args.only_fold:
            continue
        train,held = np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        if {rows[i]['physical_scene'] for i in train} & {rows[i]['physical_scene'] for i in held}:
            raise ValueError('Scene overlap')
        mean,std = x[train].mean(0),np.maximum(x[train].std(0),1e-6)
        data['geometry'] = torch.from_numpy(np.clip((x-mean)/std,-10,10).astype(np.float32))
        prior = float((y[train].sum()+1)/(len(train)+2))
        groups = [(original_rows[i]['recording_id'],original_rows[i]['agent_id'],original_rows[i]['first_row']) for i in held]
        for seed in reg['seeds']:
            if args.only_seed is not None and seed != args.only_seed:
                continue
            for arm in reg['arms']:
                if args.only_arm is not None and arm != args.only_arm:
                    continue
                name = f'fold{fold}_seed{seed}_{arm}'
                checkpoint,trial_path,pred_path = output/(name+'.pt'),output/(name+'.json'),output/(name+'.npz')
                trial_id = {'run':identity,'fold':fold,'seed':seed,'arm':arm,'normalization_mean':mean.tolist(),
                            'normalization_std':std.tolist(),'train_rows':train.tolist()}
                if trial_path.exists():
                    old = json.loads(trial_path.read_text())
                    if (old['trial_identity'] != trial_id or old['checkpoint_sha256'] != file_digest(checkpoint)
                            or old['prediction_sha256'] != file_digest(pred_path)):
                        raise ValueError('Saved model/result identity changed')
                    trials.append({**old,'result_source':'cached_verified'})
                    continue
                torch.manual_seed(seed)
                model = PastAppearanceProbe(x.shape[1])
                with torch.no_grad():
                    model.start.bias.fill_(float(np.log(prior/(1-prior))))
                def beat(step,loss,elapsed):
                    entry = {'pid':os.getpid(),'state':'training','trial':name,'step':step,
                             'target_steps':reg['training']['updates'],'loss':loss,'trial_fit_seconds':elapsed}
                    atomic_json(output/'heartbeat.json',entry)
                    print(json.dumps(entry),flush=True)
                fit = fit_probe(model,data,train,reg['training'],trial_id,checkpoint,
                                stop_at=args.stop_after,heartbeat=beat)
                if not fit['complete']:
                    partial.append({'trial':name,'step':fit['step'],'held_evaluation':'not_run_incomplete_budget'})
                    continue
                model.eval()
                predictions,probabilities = [],[]
                with torch.no_grad():
                    for start in range(0,len(held),32):
                        ids = held[start:start+32]
                        delta,logit = model(data['geometry'][ids],data['images'][ids],data['mask'][ids],arm)
                        p = pixel_delta_to_native(delta,data['image_xy'][ids],data['homography'][ids]).numpy()
                        predictions.append(p); probabilities.append(torch.sigmoid(logit).numpy())
                prediction,probability = np.concatenate(predictions),np.concatenate(probabilities)
                switch = (probability>=reg['diagnostic_probability_gate']) & mask[held].all(1)
                guarded = prediction*switch[:,None,None]
                def score(p):
                    return forecast_metrics(p,native[held],scale[held],easy,groups)
                np.savez(pred_path,held=held,prediction=prediction,probability=probability,guarded=guarded)
                trial = {'trial_identity':trial_id,'result_source':'fresh_training_or_resumed_fixed_budget',
                    'fold':fold,'seed':seed,'arm':arm,'train_rows':len(train),'held_rows':len(held),
                    'steps':fit['step'],'fit_seconds':fit['fit_seconds'],
                    'parameters':sum(p.numel() for p in model.parameters()),
                    'loss_first':fit['losses'][0],'loss_last':fit['losses'][-1],
                    'classification':score_probabilities(y[held],probability,prior=prior),
                    'run_balanced_brier':group_balanced_brier(y[held],probability,prior,groups),
                    'agent_balanced_brier':group_balanced_brier(y[held],probability,prior,
                        [(rows[i]['recording_id'],rows[i]['agent_id']) for i in held]),
                    'cv':score(np.zeros_like(prediction)), 'trajectory_unrestricted':score(prediction),
                    'trajectory_fixed_gate':score(guarded),'fixed_gate_switch_rate':float(switch.mean()),
                    'held_full_image_support':int(mask[held].all(1).sum()),
                    'checkpoint_sha256':file_digest(checkpoint),'prediction_sha256':file_digest(pred_path),
                    'new_deployment':False}
                atomic_json(trial_path,trial); trials.append(trial)
                print(json.dumps({'completed_trial':name,'unrestricted_gain_pct':trial['trajectory_unrestricted']['gain_vs_cv_pct'],
                    'guarded_gain_pct':trial['trajectory_fixed_gate']['gain_vs_cv_pct'],
                    'brier_lift':trial['classification']['brier_lift_over_prior']}),flush=True)
    complete = len(trials)==18 and not partial
    report = {'identity':identity,'result_source':'fresh_training_and_held_fit_evaluation_or_hash_verified_resume',
        'input_cache_report':receipt,'complete_registered_budget':complete,'trials':trials,'partial':partial,
        'fresh_completed_evaluations':sum(t['result_source']!='cached_verified' for t in trials),
        'cached_verified_models':sum(t['result_source']=='cached_verified' for t in trials),
        'elapsed_invocation_seconds':time.monotonic()-started,'future_input':False,
        'held_model_or_threshold_selection':False,'new_primary_metric':False,'new_deployment':False,
        'stage5c_executed':False,'smc_enabled':False}
    reports.mkdir(parents=True)
    atomic_json(reports/'metrics.json',report)
    atomic_json(output/'heartbeat.json',{'pid':os.getpid(),'state':'complete' if complete else 'partial_stopped',
        'completed_trials':len(trials),'partial':partial,'elapsed_seconds':time.monotonic()-started})
    if complete:
        hashes = {f'fold{t["fold"]}_seed{t["seed"]}_{t["arm"]}':t['checkpoint_sha256'] for t in trials}
        if (output/'completion.json').exists():
            if json.loads((output/'completion.json').read_text())['checkpoint_hashes'] != hashes:
                raise ValueError('Original completion checkpoint identities changed')
        else:
            atomic_json(output/'completion.json',{'report_sha256':file_digest(reports/'metrics.json'),
                'checkpoint_hashes':hashes})
    print(json.dumps({'complete':complete,'trials':len(trials),'partial':partial}),flush=True)


if __name__ == '__main__':
    main()
