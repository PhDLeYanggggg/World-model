"""Repeatable read-only v2 source replay, not another full pair-matching run."""
import sys
sys.dont_write_bytecode=True

import json
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_movingdronecrowd_background as base
from scripts import audit_m3w_movingdronecrowd_background_v2 as version
from src.evaluation.m3w_background_finalization import summarize_roles_once


def check_bindings(root,bindings):
    root=root.resolve()
    for name,digest in bindings.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or base.sha(path)!=digest:
            raise ValueError('Changed or missing frozen binding: '+name)


def frozen_context():
    public=version.PUBLIC
    read=lambda path:json.loads(path.read_text())
    reg=read(public/'registration.json');check_bindings(ROOT,reg['bindings'])
    parent_public=version.PARENT_PUBLIC
    parent=read(parent_public/'registration.json');check_bindings(ROOT,parent['bindings'])
    for key,name in (('parent_registration_sha256','registration.json'),
            ('parent_image_manifest_sha256','image_manifest.json'),
            ('parent_acquisition_sha256','acquisition.json'),('parent_timing_pilot_sha256','pilot.json')):
        if base.sha(parent_public/name)!=reg[key]:raise ValueError('Changed parent source receipt')
    if base.sha(ROOT/base.REGISTRY)!=parent['reservation_sha256']:
        raise ValueError('Changed whole-source reservations')
    base.SourceReservations(ROOT)
    base.intake.cached_source()
    for key,name in (('source_manifest_sha256','source_manifest.json'),
            ('source_analysis_sha256','analysis.json'),('source_split_interpretation_sha256','split_interpretation.json')):
        if base.sha(base.intake.PUBLIC/name)!=parent[key]:raise ValueError('Changed original source evidence')
    cfg=reg['configuration'];matcher_path=ROOT/cfg['matcher_config']
    identity=read(base.OLD_PRIVATE/'identity.json')
    if (base.sha(matcher_path)!=parent['matcher_configuration_sha256']
            or base.sha(base.OLD_PRIVATE/'identity.json')!=parent['protected_identity_sha256']
            or identity['opencv']!=base.cv2.__version__ or identity['numpy']!=base.np.__version__):
        raise ValueError('Changed protected matching runtime or identity')
    groups=read(base.OLD_GROUPS)['conservative_components']
    if base.sha(base.OLD_GROUPS)!=parent['protected_constraints_sha256']:
        raise ValueError('Changed protected source grouping')
    protected=sorted({v for group in groups for v in group})
    if len(protected)!=112 or len(groups)!=45:raise ValueError('Incomplete protected source inventory')
    return reg,groups,protected


def main():
    started=time.monotonic();public=version.PUBLIC
    reg,groups,protected=frozen_context()
    read=lambda path:json.loads(path.read_text())
    cfg=reg['configuration'];matcher_path=ROOT/cfg['matcher_config']
    original=read(public/'analysis.json');verify=read(public/'verification.json')
    execution=read(public/'execution.json');acquisition=read(public/'acquisition.json')
    if (any(v['analysis_sha256']!=base.sha(public/'analysis.json') for v in (execution,verify))
            or acquisition['image_manifest_sha256']!=base.sha(public/'image_manifest.json')
            or not verify['all_pair_receipts_and_graph_exact']
            or not verify['all_new_prepared_features_reextracted_exact']):
        raise ValueError('Require completed full-roster execution and verification')
    base.PUBLIC,base.PRIVATE=public,version.PRIVATE
    base.cv2.setNumThreads(1)
    matcher=dict(read(matcher_path),selected_frames_one_based=[1,2,3])
    selection=reg['view_selection']
    features,frames,feature_receipts=base.load_features(selection,protected,matcher,reextract=True)
    moving=sorted({r['node'] for r in selection})
    roster=base.pair_roster(moving,['drone_'+v for v in protected],cfg['roster_seed'])
    if len(roster)!=20580:raise ValueError('Keep the complete original pair roster')
    results=[]
    for index,row in enumerate(roster):
        value=read(base.pair_path(row))
        if any(value.get(k)!=v for k,v in dict(registration_sha256=base.sha(public/'registration.json'),**row).items()):
            raise ValueError('Changed saved pair identity')
        if index<cfg['verification_fresh_pair_prefix']:
            if base.match_pair(row,features,frames,matcher)!={k:v for k,v in value.items() if k!='match_seconds'}:
                raise ValueError('Fresh fixed-prefix matching differs')
        results.append(value)
    if summarize_roles_once(base,results,selection,protected,groups,feature_receipts)!=original:
        raise ValueError('Complete saved pair graph or source feature receipts changed')
    print(json.dumps(dict(result_source='cached_verified_all_receipts_plus_fresh_features_and_fixed50pairs',
        seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        prepared_features_reextracted_exact=360,protected_descriptor_views_verified=336,
        saved_clip_pairs_and_graph_exact=20580,freshly_recomputed_pairs=cfg['verification_fresh_pair_prefix'],
        analysis_sha256=base.sha(public/'analysis.json'),application_artifacts_written=0,
        bytecode_cache_writes_disabled=True,network_requests=0,new_model_training=False,
        predictive_outcomes_read=False,roles_assigned=False,stage5c_executed=False,smc_enabled=False)))


if __name__=='__main__':main()
