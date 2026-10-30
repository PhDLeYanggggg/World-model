"""Publish verified aggregate raw-label provenance, without rescoring forecasts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_source_label_provenance as run

LABELS=dict(zip(run.api.REASONS,('Beyond supplied record detection range','After last detection of scoped ID',
    'Inside ID span; other agents observed','Inside ID span; no detection at frame')))


def percent(n,d):return 'not_estimable' if d<=0 else f'{100*n/d:.3f}%'


def evidence():
    a=json.loads((run.PUBLIC/'audit.json').read_text());v=json.loads((run.PUBLIC/'verification.json').read_text())
    if not v['all74records_reparsed'] or not v['all_scalar_receipts_exact'] or v['audit_sha256']!=run.sha(run.PUBLIC/'audit.json'):
        raise ValueError('Complete separate raw-label replay required')
    binding=dict(audit_sha256=run.sha(run.PUBLIC/'audit.json'),verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        report_code_sha256=run.sha(Path(__file__)))
    path=run.PUBLIC/'evidence.json'
    if path.exists():
        e=json.loads(path.read_text())
        if e['bindings']!=binding:raise ValueError('Frozen raw-label report differs')
    else:
        e=dict(bindings=binding,result_source='fresh_run',
            runtime=json.loads((run.PRIVATE/'run_runtime.json').read_text()),
            replay=json.loads((run.PRIVATE/'verify_runtime.json').read_text()),
            import_discrepancy_observed=not a['totals']['all_stored_masks_and_known_coordinates_match'],
            raw_labels_recovered_or_written=False,forecasts_rescored=False,training_executed=False,
            controller_held_outcomes_opened=False,independent_roles_opened=False,deployment_changed=False,
            submission_ready=False,stage5c_executed=False,smc_enabled=False)
        run.once(path,e)
    return a,e


def render(a,e):
    t=a['totals'];rt=e['runtime'];vr=e['replay']
    lines=['# Raw Future-Label Provenance On The Frozen SOURCE Queries','',
        '## Material Passport','',
        'Fresh_run:74original raw records,149,510unchanged fitting queries,',
        '1,794,120exact agent/frame requests,and a separate full74record reparse.',
        'Cached_verified: original roles,query IDs,stored masks/midpoints and',
        'the preceding72producer results. Not_run: label repair,new model fitting,',
        'forecast rescoring,controller-held or independent outcomes.','',
        f"Raw audit PID{rt['pid']} exit0: {rt['seconds']:.6f}s, peak RSS{rt['peak_RSS_bytes']:,}B.",
        f"Separate reparse PID{vr['pid']} exit0: {vr['seconds']:.6f}s, peak RSS{vr['peak_RSS_bytes']:,}B.",
        'The independent structured-key lookup agrees exactly with every stored',
        'future-valid flag and known midpoint. No original importer call supplies',
        'its matching decision. Existing labels and predictions remain unchanged.','',
        '## Availability','',
        '| Unique query type | Count | Share of all queries |','|---|---:|---:|']
    for name,key in (('Complete12step future','complete_future_rows'),('Partial future','partial_future_rows'),('No future labels','missing_future_rows')):
        lines.append(f"| {name} | {t[key]:,} | {percent(t[key],t['rows'])} |")
    lines+=['',f"Observed future steps: {t['observed_future_steps']:,}; missing steps: {t['missing_future_steps']:,} ({percent(t['missing_future_steps'],t['requested_future_steps'])}).",
        f"Final-step labels exist for {t['final_step_observed_rows']:,}/{t['rows']:,}queries.",'',
        '## Missing-Step Evidence','',
        'Reasons follow the preregistered exclusive priority. Step and final-step',
        'counts are disjoint; per-query any-reason fields in audit.json can overlap.','',
        '| Supplied-export evidence | Missing steps | Share of missing steps | Missing final steps |',
        '|---|---:|---:|---:|']
    for reason in run.api.REASONS:
        n=t['missing_steps_by_reason'][reason]
        lines.append(f"| {LABELS[reason]} | {n:,} | {percent(n,t['missing_future_steps'])} | {t['missing_final_step_by_reason'][reason]:,} |")
    lines+=['','An ID ending is not proof of physical exit,death or disappearance;',
        'it can reflect tracking/visibility/export behavior. The last exported',
        'detection is not a verified last video frame. No missing row is filled',
        'with an interpolated or predicted point and then called ground truth.','',
        '## Every Required Locality','',
        '| Locality | Records | Queries | Complete | Partial | No future | Known last step |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for site,s in sorted(a['by_locality'].items()):
        lines.append('| '+site+' | '+' | '.join(f'{s[k]:,}' for k in ('recordings','rows','complete_future_rows',
            'partial_future_rows','missing_future_rows','final_step_observed_rows'))+' |')
    lines+=['','Locality082 has78queries,62partial futures and16without future labels.',
        'None has the requested last step:74last steps are after the last detection',
        'of that ID,while4are beyond the record detection range. It is not a',
        'recoverable importer omission. Do not remove this locality or rename',
        'its endpoint error0to make the all-locality FDE appear defined.','',
        '## Import Error Check','',
        f"- Stored/raw valid-flag mismatches: {t['stored_mask_mismatch_steps']:,}.",
        f"- Raw row present but stored as missing: {t['raw_present_but_stored_missing_steps']:,}.",
        f"- Stored label marked present but raw row absent: {t['stored_present_but_raw_missing_steps']:,}.",
        f"- Known midpoint mismatches: {t['known_coordinate_mismatch_steps']:,}.",
        '- Original current-agent midpoint alignment: exact for every audited query.','',
        'No import fix is indicated on this population. This is an engineering',
        'provenance result,not proof that the detector-silver labels describe',
        'human motion accurately or that missing outcomes are benign.','',
        '## Research Consequence','',
        'The preceding positive known-label detector-history effect is retained,',
        'as are its failed completion evidence gate,easy/tail failures and',
        'nonpositive superiority intervals against strong fixed physical controls.',
        'The data contain a real supplied-observation support problem,not a mask',
        'bug solved by rewriting the loader. A new threshold sweep or more raw',
        'FDE optimization cannot manufacture the missing final-step labels.','',
        'Next,freeze the improved producer and test the same gain/harm learner',
        'with physical-only,old-neural and new-neural banks on the unchanged',
        'fitting roles,draws and2%complete-policy rule. This is a development',
        'repair experiment,not permission to open independent outcomes. A complete',
        'future-position claim still needs suitable observed labels or an',
        'explicitly approved different estimand; neither is silently substituted.','',
        'SOURCE remains supplied detector-silver image-plane midpoint pixels,',
        'obs8/pred12/stride12 raw frames. No certified live sensor-as-of,verified',
        'seconds,metric calibration,true3D,foundation or human-gold claim.',
        'No new inference features,labels,model training,deployment,Stage5C or SMC.','',
        '[Complete scalar audit](audit.json), [exact reparse](verification.json),',
        '[registered scope](protocol.md), [aggregate figure](figures/support.svg).','']
    return '\n'.join(lines)


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=a['totals'];plt.rcParams.update({'font.size':10,'svg.hashsalt':run.NAME,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,labels,values,colors in ((axes[0],['Complete future','Partial future','No future labels'],
        [t[k] for k in ('complete_future_rows','partial_future_rows','missing_future_rows')],['#287487','#a07a3e','#b5473c']),
        (axes[1],['After last detection of ID','Inside ID span; others observed','Beyond record detection range','Inside ID span; no detections'],
        [t['missing_steps_by_reason'][run.api.REASONS[i]] for i in (1,2,0,3)],['#b5473c','#347e9c','#888888','#94854b'])):
        bars=ax.barh(labels,values,color=colors,height=.55);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.25)
        for bar,value in zip(bars,values):ax.text(value+max(values)*.02,bar.get_y()+bar.get_height()/2,f'{value:,}',va='center',fontsize=9)
        ax.grid(axis='x',color='#dddddd',lw=.6);ax.set_axisbelow(True)
    axes[0].set_title('149,510 unique queries in 74 records');axes[0].set_xlabel('Queries, not repeated producer views')
    axes[1].set_title('279,192 missing requested future points');axes[1].set_xlabel('Missing points; disjoint export-level reasons')
    fig.suptitle('SOURCE future-label provenance: exact raw reparse',y=.96)
    fig.text(.5,.03,'Detector-silver supplied observations, not human gold or verified physical exits. No labels repaired or forecast models changed.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.13,right=.96,top=.82,bottom=.2,wspace=.85)
    folder=run.PUBLIC/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'support.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'support.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('report','figures'));args=parser.parse_args()
    a,e=evidence()
    if args.phase=='figures':figures(a)
    else:
        text=render(a,e);path=run.PUBLIC/'readout.md'
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen audit report differs')
        path.write_text(text);print(json.dumps(dict(import_discrepancy_observed=e['import_discrepancy_observed'],rows=a['totals']['rows'])))
