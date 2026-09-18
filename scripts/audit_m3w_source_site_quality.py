"""Descriptive annotation and past-neighbor support; never filter registered rows."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_site_probe import SiteCorpus, load_config
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write

PIXEL_ROUNDOFF_TOLERANCE = 1e-6


def at_most_one_pixel(value):
    # Targets were cached as float32; restoration can turn 1 into 1.000000047.
    return np.asarray(value) <= 1+PIXEL_ROUNDOFF_TOLERANCE


def error_mass_slices(ade, ratio):
    ade,ratio=map(lambda a:np.asarray(a,dtype=float),(ade,ratio))
    if (ade.ndim!=1 or ade.shape!=ratio.shape or not np.isfinite(ade).all()
            or not np.isfinite(ratio).all() or np.any(ade<0) or np.any(ratio<0)):
        raise ValueError('Finite nonnegative aligned error and relative displacement required')
    total=float(ade.sum())
    bins=[('unchanged',ratio==0),('positive_below_0.1_box',(ratio>0)&(ratio<.1)),
          ('0.1_to_below_0.5_box',(ratio>=.1)&(ratio<.5)),('at_least_0.5_box',ratio>=.5)]
    return [dict(bin=name,rows=int(m.sum()),window_fraction=float(m.mean()),
                 cv_ade_error_mass_fraction=float(ade[m].sum()/total) if total>0 else None)
            for name,m in bins]


def restored_label_extent(target, scale, current_box):
    target,scale,current_box=map(lambda a:np.asarray(a,dtype=float),(target,scale,current_box))
    if (target.shape!=(len(scale),12,2) or scale.ndim!=1 or current_box.shape!=(len(scale),4)
            or not all(np.isfinite(a).all() for a in (target,scale,current_box)) or np.any(scale<=0)):
        raise ValueError('Finite normalized targets, positive scales and current boxes required')
    extent=current_box[:,2:]-current_box[:,:2]
    diagonal=np.linalg.norm(extent,axis=1)
    if np.any(extent<0) or np.any(diagonal<=0):
        raise ValueError('Positive ordered annotation boxes required')
    maximum=np.linalg.norm(target,axis=-1).max(1)*scale
    return maximum,maximum/diagonal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();reg=load_config(args.registration);data=SiteCorpus(reg)
    receipts=json.loads(data.manifest_path.read_text())['records']
    n=len(data.sid);flags=np.zeros((n,8,3),np.uint8)
    maximum=np.zeros(n);ratio=np.zeros(n);ade=np.zeros(n);passed=0
    for record in np.unique(data.record_ids[data.sid]):
        loc=np.flatnonzero(data.record_ids[data.sid]==record);global_rows=data.sid[loc]
        receipt=receipts[int(record)];directory=data.manifest_path.parent/receipt['recording']
        def read(name):
            path=directory/(name+'.npy')
            if file_digest(path)!=receipt['arrays'][name+'.npy']:
                raise ValueError('Unverified quality audit array')
            return np.load(path,mmap_mode='r',allow_pickle=False)
        local=data.local_ids[global_rows];image_rows=data.images[int(record)]['image_rows'][local]
        queries=read('query_keys')[local];crops=read('crop_keys')[image_rows]
        if (not np.array_equal(crops[:,:,0],queries[:,0,None]+12*np.arange(-7,1))
                or not np.all(crops[:,:,1]==queries[:,1,None])):
            raise ValueError('Source past-only frame/agent alignment changed')
        flags[loc]=read('source_flags')[image_rows]
        scale=read('scale')[local]
        maximum[loc],ratio[loc]=restored_label_extent(data.aux['target'][global_rows],
            scale,read('annotation_boxes')[image_rows[:,-1]])
        if np.count_nonzero(data.aux['baseline'][global_rows]):
            raise ValueError('Exact stationary CV must equal the current-position origin')
        ade[loc]=np.linalg.norm(data.aux['target'][global_rows].astype(float),axis=-1).mean(1)*scale
        passed+=len(loc)
    positive=data.y[data.nmain:].astype(bool)
    np.testing.assert_array_equal(positive,maximum>0)
    x=data.x[data.nmain:];neighbor=x[:,230:294].reshape(n,8,8).astype(bool)
    results=[]
    for name,m in [('all_source',np.ones(n,bool))]+[(s,data.source_sites==s) for s in reg['sites']]:
        p=m&positive
        results.append(dict(site=name,rows=int(m.sum()),positives=int(p.sum()),
            positive_maximum_annotation_pixel_displacement_quantiles=np.quantile(maximum[p],[0,.25,.5,.75,1]).tolist(),
            positive_maximum_over_current_annotation_box_diagonal_quantiles=np.quantile(ratio[p],[0,.25,.5,.75,1]).tolist(),
            positive_maximum_at_most_one_annotation_pixel=int((p&at_most_one_pixel(maximum)).sum()),
            positive_maximum_below_tenth_box_diagonal=int((p&(ratio<.1)).sum()),
            positive_maximum_at_least_half_box_diagonal=int((p&(ratio>=.5)).sum()),
            histories_any_lost=int(flags[m,:,0].any(1).sum()),
            histories_any_occluded=int(flags[m,:,1].any(1).sum()),
            histories_any_generated=int(flags[m,:,2].any(1).sum()),
            generated_past_frame_fraction=float(flags[m,:,2].mean()),
            histories_any_neighbor=int(neighbor[m].any((1,2)).sum()),
            histories_current_neighbor=int(neighbor[m,:,-1].any(1).sum()),
            varying_geometry_columns=int((np.ptp(x[m],axis=0)>0).sum()),
            cv_ade_annotation_pixels=float(ade[m].mean()),
            annotation_change_count_vs_cv_error_mass=error_mass_slices(ade[m],ratio[m])))
    result=dict(result_source='fresh_run_descriptive_quality_audit_on_cached_verified_inputs',
        registration_sha256=file_digest(args.registration),source_assignment_sha256=data.assignment_hash,
        rows_with_exact_eight_past_frame_agent_joins=passed,quantile_levels=[0,.25,.5,.75,1],
        records=results,thresholds_are_descriptive_not_selection=True,
        one_pixel_bin_roundoff_tolerance=PIXEL_ROUNDOFF_TOLERANCE,
        half_box_diagonal_not_identical_to_historical_half_box_census=True,
        future_labels_used_only_for_supervision_audit=True,future_inputs_used=False,
        rows_filtered=False,training_or_primary_changed=False,independent_confirmation=False,
        annotation_quality='offline_auto_silver_not_intention_gold')
    reports=ROOT/reg['reports'];json_write(reports/'annotation_quality_audit.json',result)
    lines=['# Annotation-Change and Past-Context Audit','',
        'Descriptive follow-up; no labels, row eligibility, training or thresholds changed.',
        'The positive label is any future annotation-coordinate change, not a verified intention event.',
        'Displacements below are restored to annotation pixels, never meters or seconds.','',
        '| Site | Positive rows | Median maximum future pixel displacement | Median fraction of current box diagonal | Positive change <=1px | Positive change <0.1 box diagonal | Any occluded past frame | Any generated past frame | Any past neighbor |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['site']} | {r['positives']} | {r['positive_maximum_annotation_pixel_displacement_quantiles'][2]:.6f} | {r['positive_maximum_over_current_annotation_box_diagonal_quantiles'][2]:.6f} | {r['positive_maximum_at_most_one_annotation_pixel']} | {r['positive_maximum_below_tenth_box_diagonal']} | {r['histories_any_occluded']} | {r['histories_any_generated']} | {r['histories_any_neighbor']} |")
    lines+=['',f'All{passed:,} supervised source histories have exact eight-frame/agent past joins at stride12.',
        'Generated/occluded flags come from dataset annotation metadata; absence of a generated flag is not a guarantee of online sensor provenance.',
        'Nonzero coordinate change can include rounding/interpolation/body-box change; this audit does not establish which mechanism caused an event.',
        'The one-pixel bin allows1e-6annotation-pixel roundoff after float32 target restoration; this does not change the binary training label.',
        'The current-box diagonal differs from the historical median-past-box denominator, even when aggregate counts agree. Do not silently exchange definitions.',
        'Neighbor presence does not prove interaction intent. Main box labels are not opened or guessed.',
        'No main protocol change, sealed-role access, new deployment, Stage5C or SMC.','']
    lines += ['## Label Frequency Versus Trajectory Error Mass','',
        'Supplementary source-only diagnosis. Exact stationary histories give a zero-displacement CV rollout.',
        'The table partitions source CV ADE error mass, not neural prediction performance or the main primary endpoint.',
        '| Source displacement slice | Windows | Fraction of all stationary windows | Fraction of stationary CV ADE error mass |',
        '| --- | ---: | ---: | ---: |']
    for b in results[0]['annotation_change_count_vs_cv_error_mass']:
        lines.append(f"| {b['bin']} | {b['rows']} | {100*b['window_fraction']:.3f}% | {100*b['cv_ade_error_mass_fraction']:.3f}% |")
    lines+=['','Window-pooled source diagnostic only; per-site values are retained in the JSON.',
        'These descriptive slices are not new training labels, sampling weights, deployment thresholds or a replacement metric.','']
    (reports/'annotation_quality_audit.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
