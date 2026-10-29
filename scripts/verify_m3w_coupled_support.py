"""Separate aggregate checks, not independent scientific replication."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts import run_m3w_coupled_support as run
from scripts.verify_m3w_selected_analogue import require, compare


def verify(rows, summary, expected):
    keys=[(r['context'],r['source'],r['seed']) for r in rows]
    require(len(keys)==len(set(keys))==72 and set(keys)==set(expected),'Complete72 required')
    banks=targets=0
    for r in rows:
        require(set(r['scores'])==set(run.api.ARMS),'Six arms required')
        o=r['scores']['origin'];d=r['diagnostics']
        require(o['rows']==o['known']+o['unknown'],'Roster accounting changed')
        require(d['local_supported']+d['local_unsupported']==o['rows'],'Local support accounting failed')
        require(0<=d['nested_supported']<=d['local_supported'],'Nested support invalid')
        require(0<=d['alpha_zero']<=o['rows'] and 0<=d['alpha_one']<=o['rows']-d['alpha_zero'],'Alpha counts invalid')
        require(d['alpha_one']<=d['alpha_sum']<=o['rows']-d['alpha_zero']+1e-10,'Alpha sum invalid')
        require(d['nested_loss_after']<=d['nested_loss_before']+1e-8,'Donor blend optimum failed')
        for name in ('overlap','row_leverage_overlap'):
            require(sum(d[name].values())==d['local_supported'],'Overlap loses query rows')
        for name,q in d['quantiles'].items():
            require(q is None or (len(q)==5 and q==sorted(q) and all(math.isfinite(v) for v in q)), 'Invalid diagnostic quantiles')
            if q is None:continue
            if name in ('alpha','mean_alpha','shuffled_alpha','nested_supported'):
                require(0<=q[0]<=q[-1]<=1,'Alpha/support quantile invalid')
            if name in ('effective_donors','record_ESS','distinct_records'):
                require(1-1e-9<=q[0]<=q[-1]<=32+1e-9,'ESS out of bounds')
        for a,s in r['scores'].items():
            require(all(s[k]==o[k] for k in ('rows','known','unknown')),'Cohort changed')
            require(s['supported']+s['unsupported']==s['rows'],'Support changed')
            require(s['supported_known']+s['supported_unknown']==s['supported'],'Known support changed')
            require(s['supported']==(d['local_supported'] if a=='independent_cost' else o['rows']),'Fallback support concealed')
            if a in run.api.JOINT_ARMS:
                require(d['arms'][a]['easy_exceeds_all']==0 and s['benefit_harm_envelope_exceedances']==0,'Joint bounds failed')
            for t in ('all','easy'):
                m=s[t]['full_cohort_MSE'];h=d['arms'][a][t]
                require(m is None or (math.isfinite(m) and m>=0),'Invalid risk MSE')
                require(s['unsupported']==0 or m is None,'Unsupported arm has finite primary')
                require(h['events']==s[t]['events'],'Event accounting changed')
                compare(h['predicted_harm_on_events']+h['predicted_harm_on_non_events'],s[t]['predicted_harm'])
                targets+=1
        for pair in (('alpha_zero','alpha_positive'),('one_reference_record','multiple_reference_records'),
                     ('record_ESS_below2','record_ESS_atleast2')):
            left,right=(d['slices'][k] for k in pair)
            total=o['rows'] if pair[0]=='alpha_zero' else d['local_supported']
            require(left['rows']+right['rows']==total,'Slice partition changed')
            for part in (left,right):require(part['known']+part['unknown']==part['rows'],'Slice knownness changed')
            if total==o['rows']:
                for a in run.api.ARMS:
                    for t in ('all','easy'):
                        if r['scores'][a][t]['full_cohort_MSE'] is not None:
                            compare((left['signed_SSE'][a][t]+right['signed_SSE'][a][t])/o['known'],r['scores'][a][t]['full_cohort_MSE'])
        for ref in r['references']:
            source,local=json.loads(ref['recording'])
            require(source==r['source']==ref['query_source'] and local==ref['local_recording'],'Recording namespace failed')
            require(ref['same_recording_donors']==ref['unselected_donors']==0,'Forbidden donor')
            require(sum(ref['known_donors_by_source'].values())==ref['known_donors'],'Source count failed')
            require(ref['supported']==(ref['known_donors']>=32),'Support threshold changed')
            banks+=1
        require(sum(ref['query_rows'] for ref in r['references'])==o['rows'],'Reference roster changed')
    means={a:{t:statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
        if all(r['scores'][a][t]['full_cohort_MSE'] is not None for r in rows) else None
        for t in ('all','easy')} for a in run.api.ARMS}
    compare(means,summary['equal_view_MSE'])
    uniform={t:statistics.fmean(r['prior_uniform_MSE'][t] for r in rows) for t in ('all','easy')}
    compare(uniform,summary['prior_uniform_MSE'])
    wins=0
    for context,seed in sorted({(r['context'],r['seed']) for r in rows}):
        group=[r for r in rows if (r['context'],r['seed'])==(context,seed)]
        require(len(group)==4,'Four sources per pair required')
        if all(means[a]['easy'] is not None for a in ('origin','independent_cost',run.api.PRIMARY)):
            v={a:statistics.fmean(r['scores'][a]['easy']['full_cohort_MSE'] for r in group)
               for a in ('origin','independent_cost',run.api.PRIMARY)}
            wins+=v[run.api.PRIMARY]<min(v['origin'],v['independent_cost'])-1e-12
    n,o,c=(means[a] for a in (run.api.PRIMARY,'origin','independent_cost'))
    checks=dict(all_views_local_support=all(r['diagnostics']['local_supported']==r['scores']['origin']['rows'] for r in rows),
        easy_MSE_beats_origin=n['easy'] is not None and n['easy']<o['easy']-1e-12,
        easy_MSE_beats_independent=n['easy'] is not None and c['easy'] is not None and n['easy']<c['easy']-1e-12,
        easy_MSE_beats_uniform=n['easy'] is not None and n['easy']<uniform['easy']-1e-12,
        all_MSE_not_worse_than_independent=n['all'] is not None and c['all'] is not None and n['all']<=c['all']+1e-12,
        all_MSE_not_worse_than_uniform=n['all'] is not None and n['all']<=uniform['all']+1e-12,
        at_least12_of18_pair_wins=wins>=12,joint_moments_valid=True)
    compare(checks,summary['magnitude_screen']);compare(wins,summary['pair_wins'])
    require(summary['magnitude_screen_pass']==all(checks.values()),'Screen result changed')
    assignment={a:n['easy'] is not None and n['easy']<means[a]['easy']-1e-12 for a in ('record_mean_blend','shuffled_alpha_blend')}
    compare(assignment,summary['conditional_assignment_screen'])
    for k,v in summary['occurrences'].items():compare(v,sum(r['scores'][run.api.PRIMARY][k] for r in rows))
    for k in ('whole_pipeline_OOF','held_inference','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'):
        require(summary[k] is False,'Evidence boundary changed')
    return dict(arm_targets=targets,reference_banks=banks,blend_objectives=72,screen_conditions=8,
                magnitude_screen_pass=all(checks.values()),conditional_assignment_screen=assignment)


def execute(check=False):
    reg,_,_=run.admission();done=json.loads((run.PUBLIC/'complete.json').read_text());rows=[]
    require(done['registration_sha256']==run.sha(run.REG) and done['summary_sha256']==run.sha(run.PUBLIC/'summary.json'),'Freeze changed')
    prior=json.loads((run.parent.PUBLIC/'complete.json').read_text())
    oldrefs={Path(e['path']).name:e for e in prior['groups']}
    for entry in done['groups']:
        path=run.ROOT/entry['path']
        require(not path.is_symlink() and path.resolve().is_relative_to((run.PUBLIC/'groups').resolve()) and run.sha(path)==entry['sha256'],'Group changed')
        row=json.loads(path.read_text());oldpath=run.parent.PUBLIC/'groups'/path.name
        require(row['registration_sha256']==run.sha(run.REG),'Registration changed')
        require(run.sha(oldpath)==oldrefs[path.name]['sha256']==row['parent_group_sha256'],'Parent changed')
        old=json.loads(oldpath.read_text());compare(row['references'],old['references'])
        for a,b in (('origin','origin'),('independent_cost','local_signed32')):
            compare(row['scores'][a],old['scores'][b]);require(row['prediction_sha256'][a]==old['prediction_sha256'][b],'Control changed')
        compare(row['prior_uniform_MSE'],{t:old['scores']['selected_nearest32'][t]['full_cohort_MSE'] for t in ('all','easy')})
        rows.append(row)
    result=verify(rows,json.loads((run.PUBLIC/'summary.json').read_text()),run.pooled.previous.previous.parent.expected(reg))
    result.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        verifier_sha256=run.sha(Path(__file__)),prior_controls=144,result_source='cached_verified_separate_receipt_arithmetic',
        new_inference=False,independent_replication=False)
    path=run.PUBLIC/'verification.json'
    if check:require(result==json.loads(path.read_text()),'Verification replay differs')
    else:run.once(path,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
