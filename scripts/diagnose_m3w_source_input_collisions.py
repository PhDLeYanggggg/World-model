"""Post-hoc training-only exact-input audit; no model fit or held forecasts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_cost_deferral import load_config,build_data
from scripts.run_m3w_source_start_probe import array_hash
import numpy as np
import torch
from src.evaluation.m3w_input_collisions import input_keys,collision_summary
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-registration',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reg=load_config(args.training_registration)
    data,train,weights,scale,_,_=build_data(reg)
    assert np.allclose(weights,np.ones(len(train))/len(train),rtol=0,atol=1e-15)
    keys=[]
    for start in range(0,len(train),128):
        features,frame=data.dynamics_inputs(train[start:start+128],training=True)
        geometry,_,coverage=features
        keys.extend(input_keys(geometry.numpy(),coverage.numpy(),*(x.numpy() for x in frame)))
    keys=np.asarray(keys)
    loc=train-data.nmain
    target=data.loss_targets(train).numpy()
    summary=collision_summary(keys,target)
    private=ROOT/'data/stage_cvpr2027_experiments/source_deferral_transfer_v1'
    public=ROOT/'outputs/publication_readiness_2026_09/source_deferral_transfer_v1'
    path=private/'training_input_fingerprints.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            np.testing.assert_array_equal(old['ids'],train)
            np.testing.assert_array_equal(old['input_sha256'],keys)
    else:
        np.savez(path,ids=train,input_sha256=keys)
    details=summary.pop('duplicate_group_details')
    result=dict(result_source='fresh_run_posthoc_training_only_exact_input_diagnostic',
        training_registration_sha256=file_digest(args.training_registration),
        training_ids_input_target_sha256=array_hash(train,keys,target),
        private_fingerprint_sha256=file_digest(path),input_includes=['geometry','coverage','radius','rotation','support'],
        rgb_excluded_because_mask_arm_zeroes_it=True,labels_not_used_for_grouping=True,
        no_near_neighbor_or_tolerance_grouping=True,uniform_training_weights_verified=True,
        unsupported_training_rows=int((~data.support[loc]).sum()),
        model_or_policy_changed=False,held_rows_scored=0,main_rows_scored=0,
        population_bayes_or_irreducibility_claim=False,**summary)
    json_write(public/'training_input_collision_audit.json',result)
    json_write(private/'training_input_collision_groups.json',details)
    lines=['# Exact Observed-Input Collision Audit','','## Material Passport','',
        'Post-hoc, training-only analysis after the fixed source readout. No model fitting,',
        'held forecast, label-based input construction or policy change. It concerns the',
        'current mask-only predictor, not a hypothetical richer multimodal observation.','',
        '| Quantity | Value |','| --- | ---: |']
    for k,v in summary.items():
        lines.append(f'| {k} | {v} |')
    lines+=['','## Exact Scope and Argument','',
        'Keys contain the actual geometry, every coverage-mask element, radius, rotation',
        'and support flag. Floating signed zero is canonicalized; no approximate',
        'clustering, PCA or learned metric is used. RGB is excluded only because the',
        'current mask-only model explicitly replaces it with zero. The keys include',
        'the restoration frame so identical hidden inputs with different restored',
        'outputs are not incorrectly merged. Future labels are read after grouping.','',
        'For a group with identical effective inputs and uniform weights, any',
        'deterministic current-input predictor must return one common trajectory.',
        'If at least half the full target paths are exactly zero, stationary CV is',
        'an empirical minimizer of mean Euclidean ADE on that group: at each step,',
        'triangle inequality gives total cost at q at least baseline cost plus',
        '(number_zero - number_nonzero)*norm(q). Only conflicting duplicate groups',
        'are counted for this sufficient condition. Zero may be a nonunique minimizer.','',
        'This is a finite-training-cohort statement, not a Bayes-risk or real-world',
        'unpredictability theorem. It is sensitive to observed schema and precision;',
        'exact duplicates cannot describe near-duplicate ambiguity. Small collision',
        'coverage would reject exact aliasing as the main explanation, not prove',
        'that current features contain enough learnable signal. New RGB/video inputs',
        'could distinguish rows that this mask-only schema cannot.','',
        'Offline supplied annotations, pixel/past-normalized raw-frame task only.',
        'No metric/seconds/true-3D/foundation claim. Stage5C and SMC remain off.','']
    (public/'training_input_collision_audit.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
