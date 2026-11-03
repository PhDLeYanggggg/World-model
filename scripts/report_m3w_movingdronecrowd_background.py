"""Summarize completed source evidence without granting predictive admission."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_movingdronecrowd_background as parent_audit


def candidate_support(recordings,candidates,groups):
    lookup={parent_audit.moving_node(r['clip']):r for r in recordings}
    if len(lookup)!=len(recordings) or not set(candidates)<=lookup.keys():
        raise ValueError('Require unique, source-traceable candidate recordings')
    selected=[lookup[node] for node in candidates]
    containing=[g for g in groups if set(g)&set(candidates)]
    if any(sum(node in group for group in containing)!=1 for node in candidates):
        raise ValueError('Conservative components must partition candidate clips')
    name='obs8_pred12_native_stride1'
    fields=('target_agent_windows','past_visible_agent_instances',
            'history_complete_agent_instances','history_complete_but_future_incomplete_instances',
            'queries_with_targets','nonoverlapping_raw_intervals_with_targets')
    return dict(clips=len(selected),publisher_scene_labels=len({r['publisher_scene_label'] for r in selected}),
        conservative_components_touching_candidates=len(containing),
        rows=sum(r['rows'] for r in selected),recording_local_tracks=sum(r['agents'] for r in selected),
        native_obs8_pred12={k:sum(r['window_support'][name][k] for r in selected) for k in fields},
        native_endpoint_support={h:{k:sum(r['endpoint_support'][h][k] for r in selected)
            for k in ('endpoint_label_windows','complete_interval_windows')} for h in ('10','25','50','100')},
        candidate_counts_are_admission=False,components_are_verified_physical_sites=False,
        windows_are_independent_samples=False,future_masks_used_as_predictive_inputs=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version2',action='store_true');args=parser.parse_args()
    audit=parent_audit
    if args.version2:
        from scripts import audit_m3w_movingdronecrowd_background_v2 as audit
    public=audit.PUBLIC
    read=lambda name:json.loads((public/name).read_text())
    result,verify,execution,images,acquisition,pilot=[read(name) for name in
        ('analysis.json','verification.json','execution.json','image_manifest.json','acquisition.json','pilot.json')]
    if (not verify['all_pair_receipts_and_graph_exact'] or not verify['all_new_prepared_features_reextracted_exact']
            or verify['analysis_sha256']!=audit.sha(public/'analysis.json') or result['clip_pairs']!=20580):
        raise ValueError('Require the completed full roster and prescribed replay')
    sources=json.loads((audit.intake.PUBLIC/'split_interpretation.json').read_text())['protocols']['extended']['splits']['train']['clips']
    linked=set(result['moving_nodes_linked_to_protected']);unresolved=set(result['insufficient_background_moving_nodes'])
    candidates=sorted(audit.moving_node(key) for key in sources if audit.moving_node(key) not in linked|unresolved)
    groups=result['conservative_exclusion_components']
    source_analysis_path=audit.intake.PUBLIC/'analysis.json'
    source_analysis=json.loads(source_analysis_path.read_text())
    support=candidate_support(source_analysis['recordings'],candidates,groups)
    scene_by_node={v['node']:v['clip'].split('/')[0] for v in images['views']}
    nodes=sorted(scene_by_node)
    same_label_pairs=sum(scene_by_node[a]==scene_by_node[b]
        for i,a in enumerate(nodes) for b in nodes[i+1:])
    same_label_supported=sum(v['kind']=='within_moving' and scene_by_node[v['left']]==scene_by_node[v['right']]
                             for v in result['supported_pairs'])
    mixed=[g for g in groups if any(v.startswith('moving_') for v in g) and any(v.startswith('drone_') for v in g)]
    cache_bytes=sum(v['bytes'] for v in images['views'])
    bounds=sum(v['head_centers_outside_image'] for v in images['views'])
    payload=dict(result_source='verified_scalar_readout_not_new_matching',
        analysis_sha256=audit.sha(public/'analysis.json'),verification_sha256=audit.sha(public/'verification.json'),
        moving_clips_linked_to_protected=len(linked),moving_insufficient_background=len(unresolved),
        publisher_train_clips=64,remaining_unadmitted_publisher_train_candidates=len(candidates),
        remaining_unadmitted_candidate_nodes=candidates,mixed_source_exclusion_components=len(mixed),
        remaining_unadmitted_candidate_support=support,source_structure_sha256=audit.sha(source_analysis_path),
        within_same_publisher_label_pairs=same_label_pairs,
        supported_within_same_publisher_label_pairs=same_label_supported,
        publisher_label_support_is_geographic_recall=False,
        source_roles_assigned=False,model_training=False,physical_independence_certified=False,
        stage5c_executed=False,smc_enabled=False)
    lines=['# Moving-Drone Source-Overlap Results','', '## Material Passport','',
        'Fresh_run: specified new source-view acquisition/extraction and complete',
        'new pair roster, with the fixed pilot reused. Cached_verified: declared',
        'parent/landscape reuse where applicable, plus protected descriptors,',
        'all saved receipts/graph, all360new prepared-raster re-extractions and a',
        'fresh fixed50-pair prefix. This is not a second full download or second',
        '20,580-pair matching run. No model training or forecast-error evaluation.','',
        '## Acquisition And Resource Cost','',
        f'New source views:360; protected cached views:336. Current acquisition',
        f'invocation transfers{images["network_payload_range_bytes_this_invocation"]:,}archive range bytes,',
        f'with{images["new_views_this_invocation"]}new and{images["cached_views_this_invocation"]}exact cached views.',
        'These are invocation counts, not an invented total for any failed requests.',
        f'Prepared view cache:{cache_bytes:,}bytes; no raw JPEG/full archive retained.',
        f'Head centers outside image bounds in these audit views:{bounds}; none are silently removed.',
        f'Acquisition PID{acquisition["pid"]}: {acquisition["seconds"]:.3f}s, peak RSS{acquisition["peak_RSS_bytes"]:,}bytes.',
        f'Fixed timing pilot estimated{pilot["estimated_full_matching_seconds"]:.1f}s for all pairs; this was an estimate.',
        f'Completion process PID{execution["pid"]}: {execution["seconds"]:.3f}s, peak RSS{execution["peak_RSS_bytes"]:,}bytes.',
        f'Its fresh/cached pairs:{execution["new_pairs"]}/{execution["cached_pairs"]}; do not erase the pilot or reused work.']
    if 'original_matching_exit_code' in execution:
        lines += [f'Original matching PID{execution["original_matching_pid"]} saved all pairs after',
            f'{execution["original_matching_loop_seconds"]:.3f}s, with{execution["original_matching_peak_RSS_bytes"]:,}bytes peak RSS,',
            f'{execution["original_matching_new_pairs"]}new pairs and{execution["original_cached_pilot_pairs"]}cached pilot pairs.',
            f'It then exited{execution["original_matching_exit_code"]} on duplicate summary-key assembly, not matching.',
            'The registered forward adapter recovers that work without changing matching or grouping;',
            'its short completion time is not presented as the original full matching time.',
            'See completion_failure.md and completion_repair_registration.json.']
    lines+=['',
        '## Complete Roster','', '| Scope | Strong multi-view candidates | Ambiguous candidates | Unsupported |',
        '|---|---:|---:|---:|']
    for name,counts in result['status_counts'].items():
        lines.append(f'| {name} | {counts["strong_multiview"]} | {counts["ambiguous"]} | {counts["unsupported"]} |')
    lines+=['',f'All20,580clip pairs complete;{result["frame_pairs_attempted"]:,}frame pairs attempted.',
        f'Conservative components touching both sources:{len(mixed)}. New moving',
        f'clips connected to protected source:{len(linked)}. Insufficient-background',
        f'new clips:{len(unresolved)}. Of64publisher-train clips,{len(candidates)}remain',
        'unadmitted candidates after these known-link/background exclusions.',
        'This is not a physical-site count, role grant, independence certificate',
        'or positive transfer result. Preserve all weak, publisher-scene and old',
        '45group constraints; do not reassign a protected clip to create training data.',
        f'Of {same_label_pairs} within-publisher-label pairs, {same_label_supported} have image support.',
        'The publisher label is not an exact shared-field-of-view ground truth;',
        'this ratio is not measured geographic recall. It illustrates why the',
        'publisher priors must remain even when sparse image matching is unsupported.','',
        '## Remaining Candidate Support, Not Admitted Data','',
        f'The remaining publisher-train candidates contain {support["rows"]:,} rows,',
        f'{support["recording_local_tracks"]:,} recording-local tracks across',
        f'{support["publisher_scene_labels"]} publisher labels and',
        f'{support["conservative_components_touching_candidates"]} conservative components.',
        'Neither label nor component count is a verified number of independent sites.','',
        '| Native 8/12 support | Count |','|---|---:|']
    for key,value in support['native_obs8_pred12'].items():
        lines.append(f'| {key} | {value:,} |')
    lines+=['','| Native endpoint offset | Endpoint labels | Complete intervals |',
            '|---|---:|---:|']
    for horizon,value in support['native_endpoint_support'].items():
        lines.append(f'| +{horizon} | {value["endpoint_label_windows"]:,} | {value["complete_interval_windows"]:,} |')
    lines+=['','These overlapping windows are structural availability counts, not',
        'independent samples, measured forecasting success or new data roles.',
        'Keep all current visible neighbors and future labels/masks separate;',
        'target-complete windows are not permission to filter inference neighbors.','',
        '## Verification','',
        f'Verification PID{verify["pid"]}: {verify["seconds"]:.3f}s, peak RSS{verify["peak_RSS_bytes"]:,}bytes.',
        f'All360new prepared features reproduce exactly; all336protected descriptors',
        f'and20,580pair receipts/graph are checked. Freshly recomputed pairs:{verify["freshly_recomputed_pair_prefix"]},',
        'the predetermined prefix only, not all pairs. Source checks prove the',
        'implemented replay and coverage, not geographic or scientific independence.','',
        '## Limits And Next Action','',
        'The first-view inspection is recorded in overview_review.md. The later',
        'within-source inspection and empty cross-source panel scope are recorded',
        'in pair_review.md. The visual review remains model-assisted self-audit,',
        'not human gold. Unsupported image matching does not prove a different',
        'place; sparse views, darkness, motion and parallax can hide overlap.',
        'DUT imagery is not screened in this roster. No complete producer-chain',
        'admission or calibration exchangeability follows from this experiment.',
        'Images and middle/last annotation masks are source-audit evidence only,',
        'not future predictive inputs. Existing112DroneCrowd confirmation and',
        'DUT reservations,2%risk criteria and negative model results remain.',
        'New source remains diagnostic until source-use and role/producer admission.',
        'Image/native-step evidence is not metric/seconds, true3D, foundation',
        'or a neural dynamics contribution. Stage5C/SMC remain off; goal unmet.']
    (public/'readout.md').write_text('\n'.join(lines)+'\n')
    payload['reporter_sha256']=audit.sha(Path(__file__))
    (public/'readout_sources.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload))


if __name__=='__main__':main()
