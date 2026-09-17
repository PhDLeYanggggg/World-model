"""Replay fixed checkpoints and publish aggregate appearance ablations, not winners."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args = parser.parse_args()
    if platform.system()=='Darwin' and platform.machine()!='arm64':
        raise SystemExit('Use native arm64 interpreter')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='4'
    import numpy as np
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe,pixel_delta_to_native
    from src.evaluation.m3w_experiment_contract import file_digest
    from scripts.run_m3w_stationary_start_probe import atomic_json

    report=json.loads(args.metrics.read_text())
    reg_path=ROOT/'configs/m3w_past_appearance_probe.json'
    reg=json.loads(reg_path.read_text())
    if file_digest(reg_path)!=report['identity']['registration_sha256'] or not report['complete_registered_budget']:
        raise ValueError('Incomplete/changed registered run')
    for p,sha in reg['bindings'].items():
        if file_digest(ROOT/p)!=sha:
            raise ValueError('Source changed: '+p)
    completion=json.loads((args.study/'completion.json').read_text())
    if completion['report_sha256']!=file_digest(args.metrics):
        raise ValueError('Use original complete report, not a resume inventory')
    if file_digest(args.cache)!=report['identity']['cache_sha256']:
        raise ValueError('Past input cache changed')
    with np.load(args.cache,allow_pickle=False) as a:
        x=a['geometry'].copy()
        images=torch.tensor(a['rgb'].astype(np.float32)/255-.5)
        mask=torch.tensor(a['mask'].astype(np.float32))
        xy=torch.tensor(a['image_xy']); h=torch.tensor(a['homography'])
    public, max_difference=[],0.0
    for trial in report['trials']:
        name=f'fold{trial["fold"]}_seed{trial["seed"]}_{trial["arm"]}'
        checkpoint,predictions=args.study/(name+'.pt'),args.study/(name+'.npz')
        if (file_digest(checkpoint)!=trial['checkpoint_sha256']
                or file_digest(predictions)!=trial['prediction_sha256']
                or completion['checkpoint_hashes'][name]!=trial['checkpoint_sha256']):
            raise ValueError('Checkpoint/predictions changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['step']!=reg['training']['updates'] or state['identity']!=trial['trial_identity']:
            raise ValueError('Model budget or identity mismatch')
        model=PastAppearanceProbe(x.shape[1]); model.load_state_dict(state['model']); model.eval()
        mean=np.asarray(state['identity']['normalization_mean'])
        std=np.asarray(state['identity']['normalization_std'])
        geometry=torch.tensor(np.clip((x-mean)/std,-10,10).astype(np.float32))
        with np.load(predictions,allow_pickle=False) as a:
            held,saved_p,saved_prob,saved_guard=[a[k].copy() for k in ('held','prediction','probability','guarded')]
        p,prob=[],[]
        with torch.no_grad():
            for start in range(0,len(held),32):
                ids=held[start:start+32]
                delta,logit=model(geometry[ids],images[ids],mask[ids],trial['arm'])
                p.append(pixel_delta_to_native(delta,xy[ids],h[ids]).numpy())
                prob.append(torch.sigmoid(logit).numpy())
        p,prob=np.concatenate(p),np.concatenate(prob)
        guard=p*((prob>=reg['diagnostic_probability_gate']) & mask[held].bool().all(1).numpy())[:,None,None]
        difference=max(float(np.max(np.abs(p-saved_p))),float(np.max(np.abs(prob-saved_prob))),
                       float(np.max(np.abs(guard-saved_guard))))
        if difference>1e-10:
            raise ValueError(f'Saved inference failed replay: {name}, {difference}')
        max_difference=max(max_difference,difference)
        public.append({k:v for k,v in trial.items() if k!='trial_identity'})
    grouped={}
    for fold in (0,1):
        grouped[str(fold)]={}
        for arm in reg['arms']:
            rows=[t for t in public if t['fold']==fold and t['arm']==arm]
            if len(rows)!=3:
                raise ValueError('Missing registered seed')
            grouped[str(fold)][arm]={
                'unrestricted_gain_pct_mean':float(np.mean([t['trajectory_unrestricted']['gain_vs_cv_pct'] for t in rows])),
                'guarded_gain_pct_mean':float(np.mean([t['trajectory_fixed_gate']['gain_vs_cv_pct'] for t in rows])),
                'guarded_easy_absolute_harm_mean':float(np.mean([t['trajectory_fixed_gate']['easy_absolute_harm'] for t in rows])),
                'brier_lift_mean':float(np.mean([t['classification']['brier_lift_over_prior'] for t in rows])),
                'switch_rate_mean':float(np.mean([t['fixed_gate_switch_rate'] for t in rows]))}
    output=args.report_dir.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use a new workspace report directory')
    output.mkdir(parents=True)
    summary={'result_source':'fresh_18_model_training_with_cached_verified_checkpoint_replay',
        'registration_sha256':file_digest(reg_path),'source_complete_report_sha256':file_digest(args.metrics),
        'summary_code_sha256':file_digest(Path(__file__)),
        'completed_models':len(public),'updates_per_model':reg['training']['updates'],
        'total_model_updates':sum(t['steps'] for t in public),
        'fit_seconds_sum':sum(t['fit_seconds'] for t in public),
        'source_input_report':report['input_cache_report'],'trials':public,'seed_means':grouped,
        'replayed_checkpoints':len(public),'max_prediction_replay_difference':max_difference,
        'held_outcomes_select_model_or_threshold':False,'full_benchmark_or_confirmation':False,
        'new_deployment':False,'stage5c_executed':False,'smc_enabled':False}
    atomic_json(output/'metrics.json',summary)
    lines=['# Past-Appearance Forecast Ablation','',
        'All 18 fixed-budget models completed; 3 seeds, 2 held fit scenes, 3 input arms.',
        'No winner chosen. Positive values indicate improvement over zero/CV on this stationary subset.', '',
        '| Held source | Arm | Mean unrestricted gain % | Mean guarded gain % | Mean guarded easy absolute harm | Mean Brier lift | Mean switch rate |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for fold,groups in grouped.items():
        for arm,m in groups.items():
            lines.append(f'| {"ETH" if fold=="0" else "Hotel"} | {arm} | {m["unrestricted_gain_pct_mean"]:.5f} | '
                f'{m["guarded_gain_pct_mean"]:.5f} | {m["guarded_easy_absolute_harm_mean"]:.6f} | '
                f'{m["brier_lift_mean"]:.6f} | {m["switch_rate_mean"]:.4%} |')
    lines += ['', 'Easy percentage ratios are undefined when the CV floor is zero. Absolute harm uses the unchanged parent normalization.',
              'Native ADE/FDE, every seed, losses and missing-image support remain in metrics.json.',
              'Fit-only, historically exposed and adaptive research; no independent scene CI or official model improvement claim.',
              'No physical seconds/meters/pose labels, Stage5C, SMC or deployment.', '']
    (output/'results.md').write_text('\n'.join(lines))
    print(json.dumps({'completed_models':len(public),'max_replay_difference':max_difference,'seed_means':grouped}),flush=True)


if __name__=='__main__':
    main()
