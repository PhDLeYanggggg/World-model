"""Training-source support and label-ablation verification; never a tuning pass."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from scripts.run_m3w_sdd_auxiliary import Corpus
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_auxiliary_mechanism import load_mechanism_registration, permuted_batch
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_sdd_auxiliary import predict
from src.world_model.m3w_track_event_sampling import training_event_labels, EVENT_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reg,parent=load_mechanism_registration(ROOT,args.registration)
    data=Corpus(parent); data.set_fold(0)
    manifest=json.loads(data.manifest_path.read_text())
    keys=np.concatenate([np.load(data.manifest_path.parent/r['recording']/'query_keys.npy')
                         for r in manifest['records']])
    tracks=np.array([str(r)+':'+str(a) for r,a in zip(data.record_ids,keys[:,1])])
    scales=[]; box_scales=[]
    for record in manifest['records']:
        folder=data.manifest_path.parent/record['recording']
        scales.append(np.load(folder/'scale.npy'))
        boxes=np.load(folder/'annotation_boxes.npy')[np.load(folder/'image_rows.npy')]
        box_scales.append(np.median(np.linalg.norm(boxes[:,:,2:]-boxes[:,:,:2],axis=-1),axis=1))
    scales=np.concatenate(scales); box_scales=np.concatenate(box_scales)
    valid=data.aux['valid']; complete=valid.all(1)
    events=np.full(len(valid),-1,int)
    events[complete]=training_event_labels(data.aux['geometry'][complete],data.aux['target'][complete])
    support={name:dict(rows=int((events==j).sum()),
                      recording_local_tracks=len(np.unique(tracks[events==j])),
                      recordings=len(np.unique(data.record_ids[events==j])))
             for j,name in enumerate(EVENT_NAMES)}
    travel=np.linalg.norm(data.aux['target'].astype(float),axis=-1).max(1)*scales
    thresholded_start=(events==1)&(box_scales>0)&(travel/np.maximum(box_scales,1e-12)>=.5)
    thresholded=dict(rows=int(thresholded_start.sum()),
        recording_local_tracks=len(np.unique(tracks[thresholded_start])),
        definition='exact_static_then_max_future_displacement_at_least_half_past_median_box_diagonal',
        original_census_population='all_60_recordings_vs_current_original_train40_only')
    old_counts={'windows':0,'tracks':0}
    for record in manifest['records']:
        old_path=ROOT/'data/stage_cvpr2027_experiments/sdd_state_support'/record['recording']/'receipt.json'
        old=json.loads(old_path.read_text())
        if old['annotation_sha256']!=record['annotation_sha256']:
            raise ValueError('Original support census source differs')
        counts=old['support_by_stride_and_type']['12'].get('Pedestrian',{})
        old_counts['windows']+=counts.get('exact_static_to_movement_windows',0)
        old_counts['tracks']+=counts.get('exact_static_to_movement_tracks',0)
    assert old_counts==dict(windows=thresholded['rows'],tracks=thresholded['recording_local_tracks'])
    thresholded['same_train40_source_census_counts_match']=True
    def quantiles(values):
        return dict(zip(('min','p25','median','p75','p95','max'),
                        map(float,np.quantile(values,[0,.25,.5,.75,.95,1])))) if len(values) else None
    main_ids=data.train
    main_events=training_event_labels(data.main['geometry'][main_ids],data.main['targets'][main_ids])
    scale_audit={}
    for name,targets,scale,label,baseline in (
            ('sdd_complete',data.aux['target'][complete],scales[complete],events[complete],data.aux['baseline'][complete]),
            ('main_fold0_training_only',data.main['targets'][main_ids],data.main['scale'][main_ids],main_events,
             data.main['baselines'][main_ids,data.cv])):
        distance=np.linalg.norm(targets.astype(float),axis=-1).mean(1)
        cv_error=np.linalg.norm(baseline.astype(float)-targets,axis=-1).mean(1)
        scale_audit[name]=dict(rows=len(targets),scale_floor_rows=int(np.isclose(scale,1e-3,rtol=1e-6,atol=0).sum()),
            mean_normalized_displacement=quantiles(distance),
            static_moves_normalized_displacement=quantiles(distance[label==1]),
            static_moves_log1p_loss_sensitivity_at_CV=quantiles(1/(1+cv_error[label==1])),
            moving_log1p_loss_sensitivity_at_CV=quantiles(1/(1+cv_error[label>=2])),
            sensitivity_is_scalar_dlog1p_dADE_not_parameter_gradient=True,
            source_or_local_scale=quantiles(scale),
            units='past-normalized displacement; native scales must not be compared as common physical units')
    probes=np.unique(np.linspace(0,len(valid)-1,128,dtype=int))
    torch.manual_seed(501); model=OfflineVisualForecast(476).eval()
    with torch.no_grad():
        model.output.weight.fill_(.01)
    mapping_checks={}; sampled_support={}
    for seed in reg['seeds']:
        path=ROOT/reg['output']/'donors'/f'seed{seed}.npy'
        donor=np.load(path,allow_pickle=False)
        np.testing.assert_array_equal(np.sort(donor),np.arange(len(valid)))
        np.testing.assert_array_equal(data.record_ids[donor],data.record_ids)
        np.testing.assert_array_equal(valid[donor],valid)
        exact=[]
        for modality in reg['modalities']:
            original=data.batch(probes,modality,auxiliary=True,training=True)
            changed=permuted_batch(original,probes,donor,data.aux)
            for key in original:
                if key != 'target':
                    torch.testing.assert_close(original[key],changed[key],rtol=0,atol=0,equal_nan=True)
            with torch.no_grad():
                np.testing.assert_array_equal(predict(model,original,modality).numpy(),
                                              predict(model,changed,modality).numpy())
            exact.append(modality)
        mapping_checks[str(seed)]=dict(mapping_sha256=file_digest(path),all_rows_verified=len(donor),
                                      real_input_rows_checked=len(probes),exact_prediction_modalities=exact)
        checkpoint=ROOT/parent['output']/'checkpoints'/f'sdd_aux_geometry_seed{seed}_fold0.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['step'] != 6000 or state['identity']['seed'] != seed:
            raise ValueError('Original source sampling checkpoint incomplete')
        draws=state['draw_counts']['auxiliary']
        if draws.sum()!=128000:
            raise ValueError('Unexpected source sampling budget')
        sampled_support[str(seed)]={name:dict(draws=int(draws[events==j].sum()),
            unique_rows=int(((events==j)&(draws>0)).sum()),
            unique_recording_local_tracks=len(np.unique(tracks[(events==j)&(draws>0)])))
            for j,name in enumerate(EVENT_NAMES)}
        sampled_support[str(seed)]['partial_or_absent']=dict(draws=int(draws[~complete].sum()),
                                                           unique_rows=int(((~complete)&(draws>0)).sum()))
    report=dict(result_source='fresh_run_training_source_diagnostic_cached_verified_source',
        registration_sha256=file_digest(args.registration),full_source_rows=len(valid),
        complete_label_rows=int(complete.sum()),partial_or_absent_rows=int((~complete).sum()),
        support=support,half_box_start_proxy=thresholded,scale_audit=scale_audit,
        source_draw_support_by_seed=sampled_support,mapping_checks=mapping_checks,
        source_train_only=True,main_held_labels_evaluated=False,new_optimizer_updates=0,
        event_definition='existing_exact_static_late_stop_45_degree_turn_proxy_on_complete_labels',
        complete_label_subset_is_diagnostic_not_a_training_filter=True,
        independent_track_or_window_claim=False)
    json_write(ROOT/reg['reports']/'source_support.json',report)
    lines=['# Source Event Support and Permutation Checks','',
        'Training-source diagnostic only, zero optimizer updates; held main labels are not evaluated.',
        'Exact-static/stop/turn proxies are not human intent labels. Partial support is not silently discarded.','',
        '| Complete-label event | Windows | Recording-local tracks | Recordings |',
        '| --- | ---: | ---: | ---: |']
    for name,s in support.items():
        lines.append(f"| {name} | {s['rows']} | {s['recording_local_tracks']} | {s['recordings']} |")
    lines+=['',f"Partial/absent-label windows: {int((~complete).sum()):,}; retained in training, unclassified in this complete-label event diagnostic.",
        f"The stricter historical half-box-displacement start proxy has {thresholded['rows']} train-only windows from {thresholded['recording_local_tracks']} recording-local tracks.",
        'The exact-nonzero training category and half-box census proxy are different labels; counts are not interchangeable.',
        'Scale-floor and target-magnitude diagnostics compare admitted SDD with main fold-0 training rows only; see JSON.',
        'A fixed 0.001 native-coordinate floor is not a verified pixel-to-local-unit calibration. No normalization change is made.',
        'Counts of actual source draws/unique event rows per seed are in source_support.json.',
        'Permutations preserve all 229,333 row identities as a bijection within recording and target-support strata.',
        'For each seed, 128 real source rows have unchanged inputs and exact predictions in all three modalities.',
        'Donor singleton/same-track counts remain in input_checks.json; no complete independence claim.',
        'Source event support is measured, not used to alter this registered training run.','']
    (ROOT/reg['reports']/'source_support.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
