"""Decode declared past frame indices for a local, fit-only visual admission audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import ExperimentContract,file_digest
from src.evaluation.m3w_past_video_alignment import native_to_image_xy,past_frame_indices,requested_crop_box
from scripts.run_m3w_stationary_start_probe import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    reg=json.loads(args.registration.read_text())
    for name,sha in reg['bindings'].items():
        if file_digest(ROOT/name)!=sha:
            raise ValueError('Frozen visual audit binding changed: '+name)
    sys.path.insert(0,str(ROOT/reg['decoder_path']))
    import av
    if av.__version__!=reg['decoder_version']:
        raise ValueError('Different decoder version')
    contract=ExperimentContract(json.loads((ROOT/reg['parent_protocol']).read_text()),ROOT)
    if contract.digest!=reg['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    source_report=json.loads((ROOT/reg['scene_report']).read_text())
    cache=ROOT/reg['scene_cache']
    if file_digest(cache)!=source_report['cache_sha256']:
        raise ValueError('Stationary row source changed')
    with np.load(cache,allow_pickle=False) as a:
        rows=json.loads(str(a['rows_json']))
    chosen={}
    for row in sorted(rows,key=lambda r:(r['recording_id'],r['agent_id'],r['frame_id'])):
        if row['data_role']!='fit' or contract.protocol['assignments'][row['recording_id']]!='fit':
            raise ValueError('Only frozen fit scenes may be inspected')
        chosen.setdefault((row['recording_id'],row['agent_id']),row)
    output,reports=args.output.resolve(),args.report_dir.resolve()
    if not output.is_relative_to(ROOT) or not reports.is_relative_to(ROOT) or output.exists() or reports.exists():
        raise ValueError('Use new workspace-local output/report directories')
    output.mkdir(parents=True)
    started=time.monotonic()
    results,local_records={},[]
    for rid in sorted({k[0] for k in chosen}):
        reader,_=contract.open_recording(rid,purpose='fit')
        source_path=ROOT/reader.metadata['files'][0]['path']
        if file_digest(source_path)!=reader.metadata['files'][0]['sha256']:
            raise ValueError('Canonical annotation source changed')
        directory=source_path.parent
        source={(int(p[0]),int(p[1])):p[2:4] for p in reader.points}
        h=np.loadtxt(directory/'H.txt')
        requests=defaultdict(list)
        scene_rows=[r for (recording,_),r in chosen.items() if recording==rid]
        for row in scene_rows:
            times=past_frame_indices(row['frame_id'],row['frame_id']+np.arange(-7,1)*row['native_frame_step'])
            for label,frame in [('first_past',int(times[0])),('current',int(times[-1]))]:
                native=source[(frame,row['agent_id'])]
                pixel=native_to_image_xy(native[None],h,projected_axes=reg['projected_axes'])[0]
                requests[frame].append({'agent_id':row['agent_id'],'current_frame':row['frame_id'],
                    'requested_frame':frame,'history_endpoint':label,'native_xy':native.tolist(),'image_xy':pixel.tolist()})
        local=output/rid
        local.mkdir()
        c=av.open(str(directory/'video.avi'))
        stream=c.streams.video[0]
        rates={'average_rate':str(stream.average_rate),'base_rate':str(stream.base_rate),
               'time_base':str(stream.time_base),'header_frames':stream.frames,
               'width':stream.width,'height':stream.height,'codec':stream.codec_context.name}
        bounds={}
        for axes in ('xy','row_col'):
            p=native_to_image_xy(reader.points[:,2:4],h,projected_axes=axes)
            inside=(p[:,0]>=0)&(p[:,0]<stream.width)&(p[:,1]>=0)&(p[:,1]<stream.height)
            bounds[axes]={'inside_rows':int(inside.sum()),'total_rows':len(p),'inside_fraction':float(inside.mean())}
        found=set(); decoded=0; full_support=point_inside=0; maximum=max(requests)
        for index,frame in enumerate(c.decode(video=0)):
            decoded=index+1
            if index in requests:
                found.add(index)
                im=frame.to_image()
                im.save(local/f'frame_{index:06d}.png')
                for r in requests[index]:
                    if index>r['current_frame']:
                        raise ValueError('Future frame request')
                    crop=requested_crop_box(r['image_xy'],im.width,im.height)
                    crop_name=None
                    if crop is not None:
                        crop_name=f"agent_{r['agent_id']:04d}_{r['history_endpoint']}.png"
                        im.crop(crop['box']).save(local/crop_name)
                        full_support+=int(crop['full_support']); point_inside+=int(crop['point_inside'])
                    local_records.append({'recording_id':rid,**r,'decoded_index':index,'encoded_pts':frame.pts,
                        'encoded_time_base':str(frame.time_base),'crop':crop,'crop_file':crop_name})
            if index%1000==0:
                atomic_json(output/'heartbeat.json',{'pid':os.getpid(),'state':'decoding_prefix','recording':rid,
                    'decoded_frames':decoded,'target_max_index':maximum,'elapsed_seconds':time.monotonic()-started})
            if index>=maximum:
                break
        c.close()
        if found!=set(requests):
            raise ValueError('Requested source frame unavailable')
        results[rid]={'video_sha256':file_digest(directory/'video.avi'),'annotation_sha256':file_digest(source_path),
            'H_sha256':file_digest(directory/'H.txt'),'stream':rates,'projection_bounds':bounds,
            'selected_agents':len(scene_rows),'requested_observed_images':sum(map(len,requests.values())),
            'unique_requested_frames':len(requests),'decoded_prefix_frames':decoded,'all_requested_frames_decoded':True,
            'full_inspection_crop_support':full_support,'annotation_point_inside_image_requests':point_inside,
            'all_requested_indices_past_or_current':True,'image_annotation_alignment_independently_verified':False,
            'physical_capture_clock_verified':False,'visual_training_approved_by_this_audit':False}
        print(json.dumps({'recording':rid,**results[rid]}),flush=True)
    atomic_json(output/'local_frame_records.json',local_records)
    report={'result_source':'fresh_run_local_video_prefix_decode_and_source_axis_replay',
        'registration_sha256':file_digest(args.registration),'decoder_version':av.__version__,
        'decoder_libraries':av.library_versions,'records':results,'elapsed_seconds':time.monotonic()-started,
        'source_axes':'upstream_plotting_row_col_not_a_fit_to_held_error',
        'clock_status':'ETH_annotation_source_15_assumption_conflicts_with_encoded_25; Hotel_consistent_not_independent_clock_verification',
        'future_labels_used_as_inputs':False,'new_model_training':False,'images_used_as_model_inputs':False,
        'development_calibration_confirmation_opened':False,'parent_protocol_changed':False,
        'seconds_or_metric_claim_allowed':False,'physical_or_human_gold_labels':False,
        'visual_model_training_status':'not_run_alignment_and_modality_admission_not_yet_established',
        'stage5c_executed':False,'smc_enabled':False}
    reports.mkdir(parents=True)
    atomic_json(reports/'audit.json',report)
    atomic_json(output/'completion.json',{'report_sha256':file_digest(reports/'audit.json'),
        'local_frame_record_sha256':file_digest(output/'local_frame_records.json')})
    atomic_json(output/'heartbeat.json',{'pid':os.getpid(),'state':'complete','elapsed_seconds':time.monotonic()-started})
    lines=['# Local Past-Video Alignment Audit','',
        'Fresh prefix decoding on fit-only sources. No new model, metric or training modality is admitted.',
        'Requests retain annotation frame indices; source index alignment is inspected, not inferred from playback seconds.', '',
        '| Source | Encoded rate | Selected agents | Past/current image requests | Unique frames | Full crop support | Old xy bounds | Upstream row/column bounds |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for rid,a in results.items():
        lines.append(f"| {rid} | {a['stream']['average_rate']} | {a['selected_agents']} | {a['requested_observed_images']} | {a['unique_requested_frames']} | {a['full_inspection_crop_support']} | {a['projection_bounds']['xy']['inside_fraction']:.4%} | {a['projection_bounds']['row_col']['inside_fraction']:.4%} |")
    lines+=['','The source plotting implementation reverses H-inverse output axes. This corrects an availability interpretation,',
        'not any frozen world-coordinate forecast or its scores. Hotel still has out-of-image annotations.',
        'Upstream ETH timing uses15 whereas actual encoded streams report25. Native frame index extraction does not resolve the capture clock.',
        'All requested images were decoded at past/current indices. Decode/index checks do not certify annotation causality,',
        'body detection, per-agent registration, physical scale or sensor-as-of availability.',
        'Raw frames/crops/agent identifiers are local only, excluded from public aggregate reports and Git.',
        'No development/calibration/confirmation labels, Stage5C, SMC, metric/seconds claim or visual-model training.', '']
    (reports/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
