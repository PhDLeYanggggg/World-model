"""Reconcile fitting arithmetic and report the fixed primal-loss experiment."""
import argparse
import json
import statistics

from scripts import run_m3w_cost_sensitive_policy as run
from scripts.report_m3w_shared_train_risk import check_policy, equal_number


def validate_result(result):
    if set(result['policies'])!=set(run.readout.ARMS) or result['scope']!='fitting_only':
        raise ValueError('Fitting-only policy roster required')
    checks = 0
    fields = ('known_rows','unknown_rows','selected_count','selected_unknown','selected_unknown_envelope_mass',
        'selected_known_benefit_mass','selected_known_harm_mass','selected_known_reference_mass',
        'selected_known_easy_reference_mass','selected_known_easy_harm_mass',
        'selected_known_easy_benefit_mass','full_known_easy_reference_mass')
    for arm,p in result['policies'].items():
        check_policy(p);checks += 1
        for record in result['recordings'].values():check_policy(record[arm]);checks += 1
        for field in fields:equal_number(p[field],sum(r[arm][field] for r in result['recordings'].values()))
    for pair in result['pairs'].values():
        for side in ('matched_left','matched_right'):check_policy(pair[side]);checks += 1
        if (pair['matched_left']['selected_count']!=pair['matched_count'] or
            pair['matched_right']['selected_count']!=pair['matched_count'] or pair['matched_role']!='utility_only'):
            raise ValueError('Matched comparison contract changed')
        if pair['matched_empty']!=(pair['matched_count']==0):raise ValueError('Empty status inconsistent')
        if pair['matched_empty'] and pair['matched_right']['easy_selected_risk_upper'] is not None:
            raise ValueError('Empty risk cannot be safe zero')
        for mode in ('full','matched'):
            p=pair[mode]
            equal_number(p['lower_mass'],p['known_difference_mass']-p['unknown_exchanged_envelope_mass'])
            equal_number(p['upper_mass'],p['known_difference_mass']+p['unknown_exchanged_envelope_mass'])
            for bound in ('lower','upper'):
                den=result['full_known_reference_mass']
                equal_number(p[bound+'_pp'],100*p[bound+'_mass']/den if den>0 else None)
    return checks


def collect():
    reg,_,_=run.admission();models=run.models()
    finish=json.loads((run.PUBLIC/'complete.json').read_text())
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text())
    if (finish['registration_sha256']!=run.sha(run.REG) or
            finish['training_freeze_sha256']!=run.sha(run.PUBLIC/'training_freeze.json')):
        raise ValueError('Readout/final binding mismatch')
    rows=[];checks=0
    for ref in finish['groups']:
        path=run.ROOT/ref['path']
        if run.sha(path)!=ref['sha256']:raise ValueError('Frozen result changed')
        row=json.loads(path.read_text())
        if row['inner_held_scored'] or row['training_freeze_sha256']!=finish['training_freeze_sha256']:
            raise ValueError('Scope or model binding changed')
        checks+=validate_result(row['result']);rows.append(row)
    summary=run.readout.summarize(rows)
    if summary!=json.loads((run.PUBLIC/'summary.json').read_text()) or finish['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Summary mismatch')
    detail=dict(policy_arithmetic_checks=checks,contexts=len(reg['contexts']),seeds=reg['seeds'],
        unique_source_recordings=len({(r['source'],rec) for r in rows for rec in r['result']['recordings']}),
        normalized_cost_pp={},slices={},gradient_diagnostics={},reporter_sha256=run.sha(__file__))
    for key in ('source','context','seed'):
        slices={}
        for value in sorted({r[key] for r in rows}):
            selected=[r for r in rows if r[key]==value]
            slices[str(value)]={arm:dict(
                budget_views=sum(run.readout.budget_status(r['result']['policies'][arm])['supported'] for r in selected),
                required_views=len(selected),
                known_benefit_pp=statistics.fmean(100*r['result']['policies'][arm]['selected_known_benefit_mass']
                    /r['result']['full_known_reference_mass'] for r in selected)) for arm in run.readout.ARMS}
        detail['slices'][key]=slices
    oracle=json.loads((run.diag.PUBLIC/'summary.json').read_text())['positive_benefit_upper_pp']
    for arm in run.readout.ARMS:
        mass={k:statistics.fmean(100*r['result']['policies'][arm][k]/r['result']['full_known_reference_mass'] for r in rows)
              for k in ('selected_known_benefit_mass','selected_known_harm_mass','selected_unknown_envelope_mass')}
        mass['retained_fraction_of_eligible_positive_benefit']=mass['selected_known_benefit_mass']/oracle
        detail['normalized_cost_pp'][arm]=mass
    for arm in ('exact_gradient','cost_sensitive'):
        ds=[r['gradient_diagnostic'][arm] for r in models.values()]
        detail['gradient_diagnostics'][arm]=dict(
            total_rows=sum(d['rows'] for d in ds),wrong_rows=sum(d['wrong'] for d in ds),
            wrong_saturated_rows=sum(d['wrong_saturated_rows'] for d in ds),
            mean_wrong_cost_mass=statistics.fmean(d['wrong_cost_mass'] for d in ds),
            mean_wrong_STE_gradient_mass=statistics.fmean(d['wrong_STE_gradient_mass'] for d in ds),
            mean_wrong_cost_gradient_mass=statistics.fmean(d['wrong_cost_gradient_mass'] for d in ds),
            mean_wrong_saturated_cost_mass=statistics.fmean(d['wrong_saturated_cost_mass'] for d in ds))
    return freeze,finish,summary,detail


def markdown(freeze,finish,summary,detail):
    f=lambda v:'not_estimable' if v is None else f'{v:+.6f}'
    out=['# Cost-Sensitive Policy: Fitting Readout','','## Scope','',
        f"Fresh_run: {freeze['count']} neural policy fits, {freeze['optimizer_steps']:,} updates, {freeze['seconds']:.3f}s, peak RSS {freeze['peak_RSS_bytes']:,}B.",
        'Cached_verified: fixed forecasts, source data, moments and exact-gradient/origin controls.',
        'Not_run: new inner-held evaluation, independent calibration/test, deployment or trajectory forecaster training.',
        f"All 72 fitting views and {detail['unique_source_recordings']} source-qualified recordings remain. Three head seeds are not forecasting seeds.",
        'The same locality082/51 action/support obstruction remains; no row or recording was dropped.',
        'All/easy positive-harm and easy-degradation budgets remain 2%. Empty risk is not_estimable.',
        'Utility completion bounds below are TRAIN diagnostics, not FDE improvements or confidence intervals.','',
        '## Exact Fitting Budgets','','|Policy|View budgets|Record budgets|Known all/easy violations|Easy degradation violations|Actions / unknown|',
        '|---|---:|---:|---:|---:|---:|']
    for arm,p in summary['safety'].items():
        out.append(f"|{arm}|{p['budget_views']}/{p['required_views']}|{p['budget_record_views']}/{p['required_record_views']}|{p['known_all_violations']}/{p['known_easy_violations']}|{p['easy_degradation_violations']}|{p['selected']}/{p['unknown']}|")
    out+=['','## Utility and Retention','','|Contrast|Full lower/upper pp|Matched lower/upper pp|Empty matched|','|---|---:|---:|---:|']
    for name,p in summary['pairs'].items():
        out.append(f"|{name}|{f(p['full_lower_pp'])}/{f(p['full_upper_pp'])}|{f(p['matched_lower_pp'])}/{f(p['matched_upper_pp'])}|{p['empty_matched_views']}|")
    out+=['','|Policy|Known benefit pp|Known harm pp|Unknown envelope pp|Eligible positive benefit retained|','|---|---:|---:|---:|---:|']
    for arm,p in detail['normalized_cost_pp'].items():
        out.append(f"|{arm}|{p['selected_known_benefit_mass']:.6f}|{p['selected_known_harm_mass']:.6f}|{p['selected_unknown_envelope_mass']:.6f}|{100*p['retained_fraction_of_eligible_positive_benefit']:.3f}%|")
    out+=['','## Slices','','Each cell is passing/required fitting views. No source/seed selection.','',
          '|Slice|Origin|Exact gradient|Cost-sensitive|','|---|---:|---:|---:|']
    for kind in ('source','seed','context'):
        for name,p in detail['slices'][kind].items():
            out.append('|'+kind+':'+name+'|'+'|'.join(f"{p[a]['budget_views']}/{p[a]['required_views']}" for a in run.readout.ARMS)+'|')
    out+=['','## Mechanism Check','',
          'These are weighted absolute logit derivatives at each model\'s own final TRAIN duals, not parameter gradients or a causal attribution of failure.',
          'Saturation threshold p(1-p)<1e-4 is diagnostic only, not an inference threshold.','',
          '|State|Wrong choices|Wrong saturated choices|Mean wrong cost|STE derivative|Cost-logistic derivative|','|---|---:|---:|---:|---:|---:|']
    for name,d in detail['gradient_diagnostics'].items():
        out.append(f"|{name}|{d['wrong_rows']}|{d['wrong_saturated_rows']}|{d['mean_wrong_cost_mass']:.6g}|{d['mean_wrong_STE_gradient_mass']:.6g}|{d['mean_wrong_cost_gradient_mass']:.6g}|")
    out+=['','## Verification and Claims','',
        f"Separate arithmetic checks reproduce {detail['policy_arithmetic_checks']} policy summaries. The readout takes {finish['seconds']:.3f}s.",
        'The oracle has been exhaustively checked on small cases; actual controls replay at complete-view and record level.',
        'Even improved fitting would not repair prior held-data failure or establish scene-joint decision generalization.',
        'No held-data CI is computed for an unrun held comparison. No deployment, Stage5C, SMC, metric/seconds, true3D, foundation or paper-readiness claim.','']
    return '\n'.join(out)


def main(check=False):
    freeze,finish,summary,detail=collect()
    outputs={run.PUBLIC/'readout.json':json.dumps(detail,indent=2,allow_nan=False)+'\n',
             run.PUBLIC/'readout.md':markdown(freeze,finish,summary,detail)}
    for path,value in outputs.items():
        if check:
            if path.read_text()!=value:raise ValueError('Report replay differs')
        elif path.exists():raise ValueError('Completed report immutable; use --check')
        else:path.write_text(value)
    print(json.dumps(dict(check=check,policy_checks=detail['policy_arithmetic_checks'],scope='fitting_only')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)
