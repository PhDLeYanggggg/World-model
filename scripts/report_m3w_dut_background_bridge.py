"""Derived source summary and private audit contacts, never predictive inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts import audit_m3w_dut_background_bridge as audit
from scripts.render_m3w_movingdronecrowd_background import font,save_contact


def render(pairs=False):
    public,private=audit.PUBLIC,audit.PRIVATE
    manifest=json.loads((public/'image_manifest.json').read_text())
    if manifest['registration_sha256']!=audit.sha(public/'registration.json'):raise ValueError('Changed image identity')
    dut={row['node']:row for row in manifest['views']}
    output=private/'visual_review';output.mkdir(exist_ok=True)
    receipt=public/('pair_contact_sources.json' if pairs else 'overview_contact_sources.json')
    if receipt.exists():raise ValueError('Preserve completed contact manifests')
    parent_public=audit.parent.version.PUBLIC
    moving={(v['node'],v['slot']):v for v in json.loads((parent_public/'image_manifest.json').read_text())['views']}
    old_path=ROOT/'outputs/publication_readiness_2026_09/dronecrowd_image_audit_v1/source_manifest.json'
    old_identity=json.loads((audit.base.OLD_PRIVATE/'identity.json').read_text())
    if audit.sha(old_path)!=old_identity['image_manifest_sha256']:raise ValueError('Changed protected source images')
    old={(v['scene_id'],v['frame_one_based']):v for v in json.loads(old_path.read_text())['frames']}
    outputs=[];typeface=font()

    def image(node,slot):
        if node.startswith('dut_'):row=dut[node];label=node+' GSOC background; time unknown'
        elif node.startswith('moving_'):
            row=moving[node,slot];label=node+f' image {row["image_frame_one_based"]}'
        else:
            frame=(1,150,300)[slot-1];row=old[node.removeprefix('drone_'),frame]
            path=ROOT/'external_data/DroneCrowd_images'/row['local_name']
            if audit.sha(path)!=row['sha256']:raise ValueError('Changed old source PNG/JPEG')
            with Image.open(path) as source:im=source.convert('RGB')
            return im,node+f' image {frame}',row['sha256']
        path=ROOT/row['path']
        if audit.sha(path)!=row['sha256']:raise ValueError('Changed prepared source view')
        with np.load(path,allow_pickle=False) as arrays:im=Image.fromarray(arrays['gray'].copy()).convert('RGB')
        return im,label,row['sha256']

    def save(canvas,name,sources):
        path=output/name;save_contact(canvas,path)
        outputs.append(dict(path=str(path.relative_to(ROOT)),sha256=audit.sha(path),sources=sources))

    if not pairs:
        nodes=sorted(dut)
        for page in range(0,len(nodes),12):
            canvas=Image.new('RGB',(960,508),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),'DUT author background composites: source audit only, NOT temporal frames',font=typeface,fill='black')
            sources=[]
            for cell,node in enumerate(nodes[page:page+12]):
                im,label,digest=image(node,1);x=(cell%4)*240;y=28+(cell//4)*160
                canvas.paste(im.resize((240,135)),(x,y));draw.text((x+2,y+138),node,font=typeface,fill='black')
                sources.append(dict(node=node,slot=1,actual_temporal_frame=None,view_sha256=digest))
            save(canvas,f'overview_{page//12+1:02d}.png',sources)
        selection='all28author_backgrounds_sorted_by_recording'
    else:
        analysis=json.loads((public/'analysis.json').read_text());cases=[]
        for kind in ('moving_to_dut','drone_to_dut','within_dut'):
            cases += sorted((r for r in analysis['supported_pairs'] if r['kind']==kind),key=lambda r:(r['left'],r['right']))[:8]
        for i,case in enumerate(cases,1):
            evidence=case['support'][0];canvas=Image.new('RGB',(960,340),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),case['kind']+' single-background candidate; no geographic certification',font=typeface,fill='black')
            sources=[]
            for column,side in enumerate(('left','right')):
                node=case[side];slot=evidence[side+'_slot'];im,label,digest=image(node,slot)
                canvas.paste(im.resize((480,270)),(column*480,28))
                draw.text((column*480+3,302),label,font=typeface,fill='black')
                sources.append(dict(node=node,slot=slot,view_sha256=digest))
            save(canvas,f'candidate_{i:02d}.png',sources)
        selection='first8lexical_supported_pairs_per_kind_single_image_pair_only'
    audit.once(receipt,dict(renderer_sha256=audit.sha(Path(__file__)),image_manifest_sha256=audit.sha(public/'image_manifest.json'),
        selection_rule=selection,outputs=outputs,visual_review_performed=False,figures_private_not_committed=True,
        source_audit_only=True,predictive_inputs=False,geographic_independence_certified=False))
    print(json.dumps(dict(contact_sheets=len(outputs),manifest=str(receipt))))


def report():
    public=audit.PUBLIC;read=lambda name:json.loads((public/name).read_text())
    result,execution,verification,acquisition,pilot,images=[read(n) for n in (
        'analysis.json','execution.json','verification.json','acquisition.json','pilot.json','image_manifest.json')]
    if (result['clip_pairs']!=6874 or not verification['all6874receipts_and_graph_exact']
            or not verification['all28raw_and_prepared_features_reextracted_exact']
            or verification['analysis_sha256']!=audit.sha(public/'analysis.json')):
        raise ValueError('Complete actual source matching and replay first')
    splits=json.loads((audit.base.intake.PUBLIC/'split_interpretation.json').read_text())
    candidates=[audit.base.moving_node(c) for c in splits['protocols']['extended']['splits']['train']['clips']]
    linked=set(result['moving_nodes_connected_to_reserved_sources'])
    remaining=sorted(set(candidates)-linked)
    values=dict(result_source='verified_scalar_readout_not_new_matching',
        analysis_sha256=audit.sha(public/'analysis.json'),verification_sha256=audit.sha(public/'verification.json'),
        moving_candidates_remaining_after_known_source_links=remaining,
        moving_candidate_count=len(remaining),moving_nodes_connected_to_reserved_sources=len(linked),
        reservation_conflicts=len(result['DUT_DroneCrowd_cross_reservation_components']),
        source_roles_assigned=False,new_model_training=False,predictive_outcomes_read=False,
        negative_matches_certify_independence=False,stage5c_executed=False,smc_enabled=False)
    rows=['# DUT Background Bridge Results','','## Evidence And Cost','',
        'Fresh_run: all 28 pinned author background PNGs, new features and complete',
        '6,874-pair roster, with fixed pilot reuse. Cached_verified: 696 parent views,',
        'all saved pair receipts/graph, 28 raw/prepared feature re-extractions and',
        'the fixed 60-pair fresh prefix. No model training or predictive-error readout.','',
        f'Acquisition PID {acquisition["pid"]}: {acquisition["seconds"]:.3f} s, peak RSS {acquisition["peak_RSS_bytes"]:,} bytes.',
        f'Successful image payload: {acquisition["successful_image_payload_bytes"]:,} bytes; no video acquired.',
        f'Pilot: {pilot["clip_pairs"]} pairs, {pilot["seconds"]:.3f} s; full-time projection {pilot["estimated_full_matching_seconds"]:.1f} s.',
        f'Full matching PID {execution["pid"]}: {execution["seconds"]:.3f} s, peak RSS {execution["peak_RSS_bytes"]:,} bytes.',
        f'Fresh/cached pairs in that invocation: {execution["new_pairs"]}/{execution["cached_pairs"]}.',
        f'Verification PID {verification["pid"]}: {verification["seconds"]:.3f} s, peak RSS {verification["peak_RSS_bytes"]:,} bytes.','',
        '## Complete Scope','','| Scope | Conservative supported pairs | Unsupported |','|---|---:|---:|']
    for name,counts in result['status_counts'].items():rows.append(f'| {name} | {counts["ambiguous"]} | {counts["unsupported"]} |')
    rows += ['',f'Actual image comparisons attempted: {result["frame_pairs_attempted"]:,}.',
        f'Moving clips linked to a reserved source: {len(linked)}.',
        f'Components joining DUT and DroneCrowd reservations: {values["reservation_conflicts"]}.',
        f'Publisher-train candidates remaining after known links: {len(remaining)} of 64.',
        'These are not newly assigned training roles or independently verified sites.','',
        '## Image Semantics And Limits','',
        'DUT provides one cropped GSOC background per recording. No duplicate temporal',
        'views are fabricated. Strong point-level support remains a conservative',
        'ambiguous recording link; it is not multi-temporal confirmation. All original',
        'publisher/89 parent components and both author-described DUT site priors remain.',
        'The quarantined intersection_04 is included only for source identity, not prediction.','',
        'Background aggregation time and raw-frame mapping are unverified. These',
        'composites are prohibited as causal predictive image inputs. DUT masks are',
        'full-image masks because raw trajectory-to-background projection is not verified.',
        f'Decoded source dimensions: {sorted({tuple(v["source_dimensions"]) for v in images["views"]})}.',
        f'Maximum registered resize-axis ratio: {max(v["axis_resize_scale_ratio"] for v in images["views"]):.4f}.',
        'Cropping, composite artifacts, perspective and sparse features limit negative',
        'matching. Unsupported pairs do not prove different physical sites or IID sampling.',
        'Inspect the private contacts and record the exact self-audit scope separately.','',
        '## Next Action','',
        'Resolve any supported cross-reservation component before role admission. If',
        'none is found, bind the completed source screens, use conditions and complete',
        'producer exclusions in a separate new-development admission manifest, then run',
        'the decisive causal-context experiment. Do not read reserved outcomes to choose',
        'that protocol. Existing 2% risk rules and all negative model results stand.',
        'No metric/seconds, true3D, foundation, deployment or submission promotion.',
        'Stage5C and SMC remain off; the full research goal remains unmet.']
    (public/'readout.md').write_text('\n'.join(rows)+'\n')
    values['reporter_sha256']=audit.sha(Path(__file__))
    (public/'readout_sources.json').write_text(json.dumps(values,sort_keys=True,indent=2)+'\n')
    print(json.dumps(values))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--contacts',action='store_true');group.add_argument('--pairs',action='store_true')
    args=parser.parse_args()
    if args.contacts or args.pairs:render(args.pairs)
    else:report()
