"""Bind completed source checks into fit-only admission, without training."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_dut_background_bridge as bridge
from src.evaluation.m3w_moving_development_admission import (
    KIND,eligible_train_nodes,group_folds,load_admission,require_fit_record)

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_development_admission_v1'


def main():
    base=bridge.base;source=base.intake.cached_source();moving=base.intake.PUBLIC
    bp=bridge.PUBLIC;read=lambda path:json.loads(path.read_text())
    result=read(bp/'analysis.json');verify=read(bp/'verification.json')
    if (not verify['all6874receipts_and_graph_exact'] or not verify['all28raw_and_prepared_features_reextracted_exact']
            or verify['analysis_sha256']!=base.sha(bp/'analysis.json')
            or result['DUT_DroneCrowd_cross_reservation_components']):
        raise ValueError('Complete consistent source bridge and resolve reservation conflicts before admission')
    structural=read(moving/'analysis.json');replay=read(moving/'verification.json')
    if (not structural['structural_support_ready'] or structural['quality_flags']
            or replay['analysis_sha256']!=base.sha(moving/'analysis.json') or not replay['all_recording_structures_exact']):
        raise ValueError('Complete clean structural intake/replay required')
    metadata=read(ROOT/source['publisher_metadata']['path'])
    if metadata['sha']!='0867fef0a8885fceb4253bbeab394f3d4dd85f5c' or metadata['cardData']['license']!='cc-by-nc-4.0':
        raise ValueError('Require the reviewed pinned non-commercial source card')
    splits=read(moving/'split_interpretation.json')['protocols']['extended']['splits']
    train=[base.moving_node(c) for c in splits['train']['clips']]
    kept,excluded=eligible_train_nodes(train,result['conservative_components'])
    previous=read(bridge.parent.version.PUBLIC/'analysis.json')
    for node in previous['insufficient_background_moving_nodes']:
        if node in kept:excluded[node]=['unresolved_source_background'];kept.pop(node)
    summaries={base.moving_node(row['clip']):row for row in structural['recordings']}
    records={}
    for node,index in kept.items():
        item=summaries[node];clip=item['clip']
        matches=[v for name,v in source['members'].items() if base.intake.clip_key(name,'annotations')==clip]
        if len(matches)!=1:raise ValueError('Exact source annotation required')
        raw=matches[0];group=f'moving_source_component_{index:03d}'
        records[node]=dict(clip=clip,source_annotation_path=raw['path'],source_annotation_sha256=raw['sha256'],
            publisher_scene_label=item['publisher_scene_label'],exclusion_group=group,
            exclusion_group_is_verified_physical_site=False,allowed_scientific_roles=['fit'],
            dataset='MovingDroneCrowd++',rows=item['rows'],recording_local_agents=item['agents'],
            complete_native8_12_windows=item['window_support']['obs8_pred12_native_stride1']['target_agent_windows'],
            image_inputs_admitted=False,predictive_error_used_for_assignment=False)
    folds,support=group_folds(records,'m3w_moving_development_v1_fixed_structural_balance')
    files=(Path(__file__),ROOT/'src/evaluation/m3w_moving_development_admission.py',
        ROOT/'tests/test_m3w_moving_development_admission.py',PUBLIC/'source_use.md',
        moving/'source_manifest.json',moving/'analysis.json',moving/'verification.json',
        moving/'split_interpretation.json',moving/'split_verification.json',
        bridge.parent.version.PUBLIC/'analysis.json',bridge.parent.version.PUBLIC/'verification.json',
        bp/'analysis.json',bp/'verification.json',bp/'image_manifest.json',bp/'registration.json',
        ROOT/'outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json')
    bindings={str(p.relative_to(ROOT)):base.sha(p) for p in files}
    for value in (source['publisher_metadata'],*source['outside'].values()):bindings[value['path']]=value['sha256']
    admission=dict(kind=KIND,status='admitted_for_development_fit_only',result_source='fresh_source_role_decision_with_cached_verified_evidence',
        bindings=bindings,source_reservation_sha256=base.sha(ROOT/base.REGISTRY),recordings=records,
        allowed_scientific_roles=['fit'],allowed_data_roles=['supervised_training','representation_pretraining'],
        source_scope='eligible_publisher_train_clips_only',excluded_publisher_train=excluded,
        all_other_clips_predictive_admission=False,independent_confirmation_claimed=False,
        grouping_unit='conservative_source_components_not_verified_physical_sites',
        proposed_within_fit_threefold_groups=folds,proposed_fold_complete_window_support=support,
        fold_assignment_uses_predictive_errors=False,producer_protocol_still_required_before_training=True,
        observation_steps=8,prediction_steps=12,observation_stride=1,
        time_unit='native_annotation_step_not_verified_raw_video_frame',coordinate_unit='image_pixel',
        effective_seconds=None,metric_calibration_verified=False,source_use='attributed_local_noncommercial_no_raw_redistribution',
        future_inputs_allowed=False,future_masks_allowed_as_inputs=False,central_velocity_allowed=False,
        train_only_endpoint_goals_required_if_future_experiment_uses_goals=True,
        new_model_training=False,forecast_errors_read=False,calibration_arrays_opened=False,confirmation_arrays_opened=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    base.once(PUBLIC/'admission.json',base.stable(admission))
    checked=load_admission(ROOT,PUBLIC/'admission.json')
    for node,row in records.items():require_fit_record(checked,node,'fit',ROOT/row['source_annotation_path'])
    summary=dict(recordings=len(records),rows=sum(v['rows'] for v in records.values()),
        complete_native8_12_windows=sum(v['complete_native8_12_windows'] for v in records.values()),
        conservative_fit_groups=len(folds),fold_complete_window_support=support,
        excluded_publisher_train=len(excluded),source_use_and_role_checks_passed=True,
        admission_sha256=base.sha(PUBLIC/'admission.json'),new_model_training=False,
        predictive_errors_read=False,independent_confirmation=False,stage5c_executed=False,smc_enabled=False)
    base.once(PUBLIC/'verification.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
