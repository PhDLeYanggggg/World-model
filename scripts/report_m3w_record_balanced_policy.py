"""Reconcile full and matched TRAIN scores without changing frozen decisions."""
import argparse
import json
import statistics

from scripts import run_m3w_record_balanced_policy as run
from scripts.report_m3w_shared_train_risk import check_policy, equal_number


def validate_result(value):
    if set(value['policies'])!=set(run.readout.ARMS) or value['scope']!='fitting_only':
        raise ValueError('Complete fitting-only model roster required')
    fields=('known_rows','unknown_rows','selected_count','selected_unknown',
        'selected_known_benefit_mass','selected_known_harm_mass','selected_known_reference_mass',
        'selected_known_easy_reference_mass','selected_known_easy_harm_mass',
        'selected_known_easy_benefit_mass','full_known_easy_reference_mass','selected_unknown_envelope_mass')
    checks=0
    for arm,p in value['policies'].items():
        check_policy(p);checks+=1
        for rec in value['recordings'].values():check_policy(rec[arm]);checks+=1
        for field in fields:equal_number(p[field],sum(rec[arm][field] for rec in value['recordings'].values()))
    for pair in value['pairs'].values():
        for side in ('matched_left','matched_right'):check_policy(pair[side]);checks+=1
        if (pair['matched_left']['selected_count']!=pair['matched_count'] or
                pair['matched_right']['selected_count']!=pair['matched_count'] or pair['matched_role']!='utility_only'):
            raise ValueError('Matched utility contract changed')
        if pair['matched_empty']!=(pair['matched_count']==0):raise ValueError('Incorrect empty match')
        if pair['matched_empty'] and pair['matched_right']['easy_selected_risk_upper'] is not None:
            raise ValueError('Empty matched risk cannot be safe zero')
        for mode in ('full','matched'):
            p=pair[mode];den=value['full_known_reference_mass']
            equal_number(p['lower_mass'],p['known_difference_mass']-p['unknown_exchanged_envelope_mass'])
            equal_number(p['upper_mass'],p['known_difference_mass']+p['unknown_exchanged_envelope_mass'])
            for bound in ('lower','upper'):
                equal_number(p[bound+'_pp'],100*p[bound+'_mass']/den if den>0 else None)
    return checks


def collect():
    reg,_,_=run.admission();models=run.models()
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text())
    end=json.loads((run.PUBLIC/'complete.json').read_text())
    if (end['registration_sha256']!=run.sha(run.REG) or
            end['training_freeze_sha256']!=run.sha(run.PUBLIC/'training_freeze.json')):
        raise ValueError('Registered models and readout mismatch')
    rows=[];checks=0
    for ref in end['groups']:
        p=run.ROOT/ref['path']
        if run.sha(p)!=ref['sha256']:raise ValueError('Frozen output changed')
        row=json.loads(p.read_text())
        if row['inner_held_scored'] or row['training_freeze_sha256']!=end['training_freeze_sha256']:
            raise ValueError('Scope or model binding changed')
        checks+=validate_result(row['result']);rows.append(row)
    summary=run.readout.summarize(rows)
    if summary!=json.loads((run.PUBLIC/'summary.json').read_text()) or end['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Summary does not reproduce')
    out=dict(policy_arithmetic_checks=checks,source_count=len({r['source'] for r in rows}),
        unique_source_recordings=len({(r['source'],rec) for r in rows for rec in r['result']['recordings']}),
        seeds=reg['seeds'],normalized_cost_pp={},slices={},record_budget_transitions={},
        reporter_sha256=run.sha(__file__),scope='fitting_only',new_held_scored=False)
    upper=json.loads((run.previous.diag.PUBLIC/'summary.json').read_text())['positive_benefit_upper_pp']
    for arm in run.readout.ARMS:
        mass={k:statistics.fmean(100*r['result']['policies'][arm][k]/r['result']['full_known_reference_mass'] for r in rows)
              for k in ('selected_known_benefit_mass','selected_known_harm_mass','selected_unknown_envelope_mass')}
        mass['retained_fraction_of_eligible_positive_benefit']=mass['selected_known_benefit_mass']/upper
        out['normalized_cost_pp'][arm]=mass
    for kind in ('source','context','seed'):
        out['slices'][kind]={}
        for name in sorted({r[kind] for r in rows}):
            chosen=[r for r in rows if r[kind]==name];values={}
            for arm in run.readout.ARMS:
                ps=[r['result']['policies'][arm] for r in chosen]
                rs=[v[arm] for r in chosen for v in r['result']['recordings'].values()]
                values[arm]=dict(budget_views=sum(run.readout.budget_status(p)['supported'] for p in ps),
                    required_views=len(ps),budget_record_views=sum(run.readout.budget_status(p)['supported'] for p in rs),
                    required_record_views=len(rs),known_all_record_violations=sum(
                        p['known_positive_risk'] is not None and p['known_positive_risk']>.02+1e-12 for p in rs))
            out['slices'][kind][str(name)]=values
    transitions={f'{a}_to_{b}':0 for a in (0,1) for b in (0,1)}
    for row in rows:
        for arms in row['result']['recordings'].values():
            a=int(run.readout.budget_status(arms['cost_sensitive'])['supported'])
            b=int(run.readout.budget_status(arms['record_balanced'])['supported'])
            transitions[f'{a}_to_{b}']+=1
    out['record_budget_transitions']=transitions
    out['constraint_factor_range']=[min(r['constraint_factor_min'] for r in models.values()),
                                    max(r['constraint_factor_max'] for r in models.values())]
    return freeze,end,summary,out


def markdown(freeze,end,summary,detail):
    fmt=lambda x:'not_estimable' if x is None else f'{x:+.6f}'
    lines=['# Record-Balanced Constraint Readout','','## Scope','',
        f"Fresh_run: {freeze['count']} neural policy fits, {freeze['optimizer_steps']:,} updates, {freeze['seconds']:.3f}s, peak RSS {freeze['peak_RSS_bytes']:,}B.",
        'Cached_verified: fixed forecasts, causal feature packets, moments and frozen cost-sensitive/origin controls.',
        'Not_run: new inner-held or independent evaluation, deployment, or new trajectory forecaster.',
        f"All 72 fitting views, {detail['unique_source_recordings']} unique source-qualified recordings and three head seeds remain.",
        'The 444 record/context/head views are repeated observations, not 444 independent recordings.',
        'No recording is dropped. The locality082/51 fixed-candidate support obstruction remains.',
        'Utility and2%budget definitions are unchanged. Equal recording weights alter training penalties only.','',
        '## Full-Policy Budgets','','|Policy|Aggregate budgets|Record budgets|Known all/easy record violations|Undefined easy|Net easy degradation violations|',
        '|---|---:|---:|---:|---:|---:|']
    for arm,d in summary['safety'].items():
        lines.append(f"|{arm}|{d['budget_views']}/{d['required_views']}|{d['budget_record_views']}/{d['required_record_views']}|{d['known_all_record_violations']}/{d['known_easy_record_violations']}|{d['undefined_easy_record_views']}|{d['easy_degradation_record_violations']}|")
    lines+=['','## Utility','','Bounds cover unknown-label completions on fitting rows, not confidence intervals or FDE gain.',
        'Masses are normalized per source/context/head view by full known reference cost, then macro-averaged.',
        'Retention divides macro-averaged known benefit by the identically normalized eligible-positive upper bound.','',
        '|Policy|Known benefit pp|Known harm pp|Unknown envelope pp|Eligible positive benefit retained|',
        '|---|---:|---:|---:|---:|']
    for arm,d in detail['normalized_cost_pp'].items():
        lines.append(f"|{arm}|{d['selected_known_benefit_mass']:.6f}|{d['selected_known_harm_mass']:.6f}|{d['selected_unknown_envelope_mass']:.6f}|{100*d['retained_fraction_of_eligible_positive_benefit']:.3f}%|")
    lines+=['','|Contrast|Full lower/upper pp|Matched lower/upper pp|Empty matched|','|---|---:|---:|---:|']
    for name,d in summary['pairs'].items():
        lines.append(f"|{name}|{fmt(d['full_lower_pp'])}/{fmt(d['full_upper_pp'])}|{fmt(d['matched_lower_pp'])}/{fmt(d['matched_upper_pp'])}|{d['empty_matched_views']}|")
    lines+=['','Count matching assesses utility only. Empty matched risk remains not_estimable; it cannot pass safety.','',
        '## Recording Slices','','|Slice|Origin|Cost-sensitive|Record-balanced|','|---|---:|---:|---:|']
    for kind in ('source','seed','context'):
        for name,arms in detail['slices'][kind].items():
            lines.append('|'+kind+':'+name+'|'+'|'.join(f"{arms[a]['budget_record_views']}/{arms[a]['required_record_views']}" for a in run.readout.ARMS)+'|')
    lines+=['','## Verification','',
        f"Independent arithmetic reconciles {detail['policy_arithmetic_checks']} policy summaries. Readout {end['seconds']:.3f}s.",
        'Controls reproduce their frozen whole-view and recording scores. No data role, threshold, or model choice follows from these results.',
        'Engineering reproducibility is not independent generalization. No Stage5C, SMC, metric/seconds, true3D, foundation or submission-ready claim.','']
    return '\n'.join(lines)


def main(check=False):
    freeze,end,summary,detail=collect()
    outputs={'readout.json':json.dumps(detail,indent=2,allow_nan=False)+'\n',
             'readout.md':markdown(freeze,end,summary,detail)}
    for name,text in outputs.items():
        p=run.PUBLIC/name
        if check:
            if p.read_text()!=text:raise ValueError('Report replay differs')
        elif p.exists():raise ValueError('Completed report immutable; use --check')
        else:p.write_text(text)
    print(json.dumps(dict(check=check,policy_checks=detail['policy_arithmetic_checks'],scope='fitting_only')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');main(parser.parse_args().check)
