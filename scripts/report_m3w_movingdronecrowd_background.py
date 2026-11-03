"""Summarize completed source evidence without granting predictive admission."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_movingdronecrowd_background as parent_audit


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
    mixed=[g for g in groups if any(v.startswith('moving_') for v in g) and any(v.startswith('drone_') for v in g)]
    cache_bytes=sum(v['bytes'] for v in images['views'])
    bounds=sum(v['head_centers_outside_image'] for v in images['views'])
    payload=dict(result_source='verified_scalar_readout_not_new_matching',
        analysis_sha256=audit.sha(public/'analysis.json'),verification_sha256=audit.sha(public/'verification.json'),
        moving_clips_linked_to_protected=len(linked),moving_insufficient_background=len(unresolved),
        publisher_train_clips=64,remaining_unadmitted_publisher_train_candidates=len(candidates),
        remaining_unadmitted_candidate_nodes=candidates,mixed_source_exclusion_components=len(mixed),
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
        f'with{images["new_views_this_invocation"]}new and{images["cached_views_this_invocation"]}resumed cached views.',
        'These are invocation counts, not an invented total for any failed requests.',
        f'Prepared view cache:{cache_bytes:,}bytes; no raw JPEG/full archive retained.',
        f'Head centers outside image bounds in these audit views:{bounds}; none are silently removed.',
        f'Acquisition PID{acquisition["pid"]}: {acquisition["seconds"]:.3f}s, peak RSS{acquisition["peak_RSS_bytes"]:,}bytes.',
        f'Fixed timing pilot estimated{pilot["estimated_full_matching_seconds"]:.1f}s for all pairs; this was an estimate.',
        f'Complete matching process PID{execution["pid"]}: {execution["seconds"]:.3f}s, peak RSS{execution["peak_RSS_bytes"]:,}bytes.',
        f'Its fresh/cached pairs:{execution["new_pairs"]}/{execution["cached_pairs"]}; do not erase the pilot or reused work.','',
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
        '45group constraints; do not reassign a protected clip to create training data.','',
        '## Verification','',
        f'Verification PID{verify["pid"]}: {verify["seconds"]:.3f}s, peak RSS{verify["peak_RSS_bytes"]:,}bytes.',
        f'All360new prepared features reproduce exactly; all336protected descriptors',
        f'and20,580pair receipts/graph are checked. Freshly recomputed pairs:{verify["freshly_recomputed_pair_prefix"]},',
        'the predetermined prefix only, not all pairs. Source checks prove the',
        'implemented replay and coverage, not geographic or scientific independence.','',
        '## Limits And Next Action','',
        'Inspect source contact sheets and candidate correspondences; any visual',
        'review must name the exact figures inspected and remain self-audited,',
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
