"""Separate policy arithmetic and concise fixed-roster evidence report."""
import argparse
import json
from pathlib import Path
import statistics
from scripts import run_m3w_easy_constrained as run
from scripts.report_m3w_shared_train_risk import check_policy,equal_number
from scripts.verify_m3w_selected_analogue import require,compare


def validate_row(row):
    d=row['diagnostics'];s=row['scores'];p=row['policy'];n=s['origin']['rows']
    require(set(s)==set(run.api.ARMS)==set(p['policies']),'Arm roster changed')
    require(0<=d['donor_conflicts']<=d['nested_supported']<=d['local_supported']<=n,'Support/conflict accounting')
    require(d['donor_easy_constraint_violations']==0,'Donor easy constraint failed')
    require(d['channel_loss']['after'][1]<=d['channel_loss']['before'][1]+1e-8,'Easy loss increased')
    for a,v in s.items():
        require(all(v[k]==s['origin'][k] for k in ('rows','known','unknown')),'Query cohort changed')
        require(v['supported']==n and v['unsupported']==0,'Fallback coverage concealed')
        require(d['arms'][a]['easy_exceeds_all']==v['benefit_harm_envelope_exceedances']==0,'Moment constraints failed')
        check_policy(p['policies'][a])
        for rec in p['recordings'].values():check_policy(rec[a])
        for key in ('selected_count','selected_unknown','selected_known_harm_mass','selected_known_reference_mass',
                    'selected_known_easy_harm_mass','selected_known_easy_reference_mass'):
            equal_number(p['policies'][a][key],sum(rec[a][key] for rec in p['recordings'].values()))
    require(p['rows']==p['known']+p['unknown'] and not p['inference_label_filter'],'Full policy cohort changed')
    require(p['matched_risk_required'] is False,'Matched risk wrongly made safety gate')
    for other,m in p['matched'].items():
        check_policy(m['left']);check_policy(m['right'])
        require(m['left']['selected_count']==m['right']['selected_count']==m['count'],'Unequal matched actions')
        require(m['empty']==(m['count']==0),'Empty control concealed')
        if m['empty']:
            require(m['empty_risk_status']=='not_estimable' and m['right']['easy_selected_risk_upper'] is None,'Empty risk called safe')
        for mode in ('full','matched'):
            key=other+'_'+mode;pair=p['pairs'][key];den=p['full_known_reference_mass']
            equal_number(pair['lower_mass'],pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(pair['upper_mass'],pair['known_difference_mass']+pair['unknown_exchanged_envelope_mass'])
            equal_number(p['contrasts'][key],100*pair['lower_mass']/den if den>0 else None)
    for left,right in (('donor_conflict','no_donor_conflict'),('reduced_alpha','unchanged_alpha'),('alpha_zero','alpha_positive')):
        l,r=d['strata'][left],d['strata'][right]
        require(l['rows']+r['rows']==n and l['known']+r['known']==s['origin']['known'],'Stratum dropped queries')
        for a in run.api.ARMS:
            for t in ('all','easy'):
                equal_number((l['signed_SSE'][a][t]+r['signed_SSE'][a][t])/s['origin']['known'],s[a][t]['full_cohort_MSE'])


def describe(rows,summary):
    fmt=lambda x:'not estimable' if x is None else f'{x:.8g}'
    data=dict(donor={k:sum(r['diagnostics'][k] for r in rows) for k in
        ('donor_conflicts','reduced_alpha','alpha_zero','alpha_one','alpha_sum','local_supported','nested_supported')},
        losses={k:[sum(r['diagnostics']['channel_loss'][k][j] for r in rows) for j in range(2)]
            for k in ('before','parent_after','after','easy_only_after')},sources={},contexts={},seeds={})
    data['policy_failure_modes']={}
    for a in run.api.ARMS:
        policies=[r['policy']['policies'][a] for r in rows]
        exceeds=lambda p,k:p[k] is not None and p[k]>.02+1e-12
        data['policy_failure_modes'][a]=dict(
            known_all_over_budget=sum(exceeds(p,'known_positive_risk') for p in policies),
            known_easy_over_budget=sum(exceeds(p,'known_easy_positive_risk') for p in policies),
            completion_all_over_budget=sum(exceeds(p,'all_selected_risk_upper') for p in policies),
            completion_easy_over_budget=sum(exceeds(p,'easy_selected_risk_upper') for p in policies),
            easy_degradation_upper_over_budget=sum(exceeds(p,'easy_degradation_upper') for p in policies),
            unknown_only_all_failure=sum(not exceeds(p,'known_positive_risk') and
                p['known_positive_risk'] is not None and exceeds(p,'all_selected_risk_upper') for p in policies),
            unknown_only_easy_failure=sum(not exceeds(p,'known_easy_positive_risk') and
                p['known_easy_positive_risk'] is not None and exceeds(p,'easy_selected_risk_upper') for p in policies))
    data['paired_completion']={}
    for key in rows[0]['policy']['pairs']:
        values={k:[100*r['policy']['pairs'][key][k]/r['policy']['full_known_reference_mass']
            if r['policy']['full_known_reference_mass']>0 else None for r in rows]
            for k in ('lower_mass','known_difference_mass','upper_mass')}
        data['paired_completion'][key]={k:statistics.fmean(v) if None not in v else None
                                        for k,v in values.items()}
    lines=['# Donor-Easy Constraint: Method and Actual Policy Readout','','## Material Passport','',
        'Fresh local fits and fixed-policy re-evaluation on exposed TRAIN only. No independent test,',
        'deployment upgrade, new neural forecaster, metric/seconds or physical-safety claim.',
        'Full-policy 2% budget unchanged; empty count-matched risks are not estimable. Old failures retained.',
        'Risk MSE is secondary, not a new paper-readiness gate.','',
        '## Donor Objective Conflict','', '```json',json.dumps(data['donor'],indent=2),'```','',
        '| Donor objective | All loss sum | Easy loss sum |','|---|---:|---:|']
    for name,value in data['losses'].items():lines.append(f'| {name} | {fmt(value[0])} | {fmt(value[1])} |')
    lines+=['','These sums reuse overlapping donor sets; they are training objective diagnostics, not independent observations.',
        'A preserved donor loss does not certify query or deployment risk.','',
        '## Actual Full Policy','',
        '| Arm | Selected | Unknown selected | Budget views | Budget record views | Undefined easy views | Known easy violations |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,s in summary['policy_readout']['safety'].items():
        lines.append(f"| {a} | {s['selected']} | {s['selected_unknown']} | {s['within_budget_views']}/{s['required_views']} | {s['within_budget_record_views']}/{s['required_record_views']} | {s['undefined_easy_views']} | {s['known_easy_violations']} |")
    lines+=['','Budget support checks all/easy positive harm/reference and signed easy degradation separately from utility.',
        'All recorded scenes stay; undefined full-policy risk does not pass. No population guarantee follows.','',
        '## Primary Policy Utility','', '| Comparison | Equal-view completion-lower change(pp) |','|---|---:|']
    for k,v in summary['policy_readout']['paired_completion_lower_utility_pp'].items():lines.append(f'| {k} | {fmt(v)} |')
    lines+=['', 'Empty matched views: '+json.dumps(summary['policy_readout']['empty_matched_views'])+'.',
        'Paired unknown outcomes cancel on shared actions; exchanged unknown rows retain envelope bounds.',
        'Empty matched utility contributes zero where its full reference denominator exists; not positive evidence.',
        'These are fitting partial-label utility diagnostics, not independent test ADE/FDE improvements.', '',
        '## Observed Versus Unresolved Risk', '',
        'Counts overlap. Unknown-only failures have a defined known-label ratio within budget,',
        'but selected missing labels make the completion upper bound exceed budget.', '',
        '```json',json.dumps(data['policy_failure_modes'],indent=2),'```', '',
        '## Paired Partial-Label Intervals', '',
        '| Contrast | Lower (pp) | Known component (pp) | Upper (pp) |',
        '|---|---:|---:|---:|']
    for key,v in data['paired_completion'].items():
        lines.append(f"| {key} | {fmt(v['lower_mass'])} | {fmt(v['known_difference_mass'])} | {fmt(v['upper_mass'])} |")
    lines+=['', 'These are deterministic completion bounds, not sampling confidence intervals.',
        'A negative lower bound alone does not prove the true policy effect is negative.', '',
        '## Secondary Risk Estimation','', '| Arm | All signed-risk MSE | Easy signed-risk MSE |','|---|---:|---:|']
    for a,m in summary['equal_view_MSE'].items():lines.append(f"| {a} | {fmt(m['all'])} | {fmt(m['easy'])} |")
    lines+=['','## Complete Source/Context/Head-Seed Breakdown','',
        '| Slice | Original easy MSE | Primary easy MSE | Primary full utility vs original(pp) |','|---|---:|---:|---:|']
    for key,outkey in (('source','sources'),('context','contexts'),('seed','seeds')):
        for value in sorted({r[key] for r in rows}):
            group=[r for r in rows if r[key]==value]
            m={a:{t:statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in group) for t in ('all','easy')} for a in run.api.ARMS}
            pol=run.policy.summarize(group);data[outkey][str(value)]=dict(views=len(group),MSE=m,policy=pol)
            lines.append(f"| {key}:{value} | {fmt(m['origin']['easy'])} | {fmt(m[run.api.PRIMARY]['easy'])} | {fmt(pol['paired_completion_lower_utility_pp']['origin_full'])} |")
    occurrences=sum(r['scores']['origin']['rows'] for r in rows)
    lines+=['','No source deletion, best-arm substitution or query/test threshold tuning.',
        f'{occurrences} repeated query occurrences, not independent samples; head seeds are not new forecaster seeds.',
        'obs8/pred12 rawstride12, pixel/local detector-silver. Stage5C and SMC remain off.']
    return data,'\n'.join(lines)+'\n'


def execute(check=False):
    reg,_,_=run.admission();done=json.loads((run.PUBLIC/'complete.json').read_text())
    require(done['registration_sha256']==run.sha(run.REG) and done['summary_sha256']==run.sha(run.PUBLIC/'summary.json'),'Freeze changed')
    prior=json.loads((run.parent.PUBLIC/'complete.json').read_text());refs={Path(r['path']).name:r for r in prior['groups']};rows=[]
    for ref in done['groups']:
        p=run.ROOT/ref['path'];require(run.sha(p)==ref['sha256'],'Group changed')
        r=json.loads(p.read_text());oldpath=run.parent.PUBLIC/'groups'/p.name
        require(run.sha(oldpath)==refs[p.name]['sha256']==r['parent_group_sha256'],'Parent changed')
        old=json.loads(oldpath.read_text());compare(r['references'],old['references'])
        for a,b in (('origin','origin'),('parent_blend','nested_support_blend')):
            compare(r['scores'][a],old['scores'][b]);require(r['prediction_sha256'][a]==old['prediction_sha256'][b],'Prior control changed')
        validate_row(r);rows.append(r)
    expected=run.pooled.previous.previous.parent.expected(reg)
    require({(r['context'],r['source'],r['seed']) for r in rows}==set(expected) and len(rows)==72,'Complete roster changed')
    summary=json.loads((run.PUBLIC/'summary.json').read_text())
    for a in run.api.ARMS:
        for t in ('all','easy'):equal_number(summary['equal_view_MSE'][a][t],statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows))
    for k,v in summary['policy_readout']['paired_completion_lower_utility_pp'].items():
        vals=[r['policy']['contrasts'][k] for r in rows];equal_number(v,statistics.fmean(vals) if None not in vals else None)
    data,md=describe(rows,summary);data.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),
        reporter_sha256=run.sha(Path(__file__)),verification='separate_arithmetic_not_independent_replication',
        result_source='cached_verified',views=72,arm_target_checks=864,prior_control_checks=144)
    path,table=run.PUBLIC/'readout.json',run.PUBLIC/'readout.md'
    if check:require(data==json.loads(path.read_text()) and md==table.read_text(),'Report replay differs')
    else:
        run.once(path,data)
        if table.exists():require(table.read_text()==md,'Existing report changed')
        else:table.write_text(md)
    print(json.dumps(dict(views=72,policy_checks=sum(6*(1+len(r['policy']['recordings']))+4 for r in rows),
        donor_conflicts=data['donor']['donor_conflicts'],budget_supported=summary['policy_readout']['primary_full_roster_budget_supported'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
