"""Separate receipt arithmetic for local cost fitting, not new inference."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts import run_m3w_local_cost_analogue as run
from scripts.verify_m3w_selected_analogue import require, compare


def verify(rows, summary, expected):
    keys = [(r['context'],r['source'],r['seed']) for r in rows]
    require(len(keys)==len(set(keys))==72 and set(keys)==set(expected),'Complete72 views required')
    targets = banks = 0
    for r in rows:
        require(set(r['scores'])==set(run.api.ARMS),'All six frozen arms required')
        origin = r['scores']['origin']; diag = r['diagnostics']
        require(origin['rows']==origin['known']+origin['unknown'],'Query accounting failed')
        for arm,s in r['scores'].items():
            require(all(s[k]==origin[k] for k in ('rows','known','unknown')),'Cohort changed')
            require(s['supported']+s['unsupported']==s['rows'],'Support accounting failed')
            require(s['supported_known']+s['supported_unknown']==s['supported'],'Known support differs')
            for t in ('all','easy'):
                d = diag['arms'][arm][t]
                require(d['events']==s[t]['events'] and 0<=d['events']<=s['supported_known'],'Event counts changed')
                compare(d['predicted_harm_on_events']+d['predicted_harm_on_non_events'],s[t]['predicted_harm'])
                require(all(math.isfinite(d[k]) and d[k]>=0 for k in
                    ('event_harm_SSE','non_event_harm_SSE','predicted_harm_on_events','predicted_harm_on_non_events')),'Invalid harm accounting')
                m = s[t]['full_cohort_MSE']
                require(m is None or (math.isfinite(m) and m>=0),'Invalid MSE')
                require(s['unsupported']==0 or m is None,'Unsupported primary cannot be finite')
                targets += 1
        require(diag['supported_queries']==r['scores']['local_signed32']['supported'],'Diagnostic support changed')
        require(diag['known_queries']==r['scores']['local_signed32']['supported_known'],'Diagnostic knownness changed')
        for name,low,high in (('effective_donor_quantiles',1,32),('max_weight_quantiles',1/32,1)):
            q=diag[name]
            if q is not None:
                require(len(q)==5 and q==sorted(q) and all(low-1e-10<=v<=high+1e-10 for v in q),'Weight leverage out of bounds')
        for opt,old in zip(diag['donor_loss']['optimum'],diag['donor_loss']['uniform']):
            require(math.isfinite(opt) and 0<=opt<=old+1e-8,'Quadratic optimum donor loss failed')
        for ref in r['references']:
            source,local=json.loads(ref['recording'])
            require(source==r['source']==ref['query_source'] and local==ref['local_recording'],'Source namespace changed')
            require(ref['same_recording_donors']==ref['unselected_donors']==0,'Forbidden donor')
            require(sum(ref['known_donors_by_source'].values())==ref['known_donors'],'Source counts differ')
            require(ref['supported']==(ref['known_donors']>=32),'Support threshold changed')
            banks += 1
        require(sum(v['query_rows'] for v in r['references'])==origin['rows'],'Reference roster loses queries')
    means = {a:{t:statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
        if all(r['scores'][a][t]['full_cohort_MSE'] is not None for r in rows) else None
        for t in ('all','easy')} for a in run.api.ARMS}
    wins=0
    for context,seed in sorted({(r['context'],r['seed']) for r in rows}):
        group=[r for r in rows if (r['context'],r['seed'])==(context,seed)]
        require(len(group)==4,'Four sources per pair required')
        vals={a:[r['scores'][a]['easy']['full_cohort_MSE'] for r in group]
              for a in ('origin','selected_nearest32','local_signed32')}
        if all(None not in v for v in vals.values()):
            wins+=statistics.fmean(vals['local_signed32'])<min(statistics.fmean(vals['origin']),statistics.fmean(vals['selected_nearest32']))-1e-12
    n,o,f=(means[a] for a in ('local_signed32','origin','selected_nearest32'))
    conditions=dict(all_views_supported=all(r['scores']['local_signed32']['unsupported']==0 for r in rows),
        easy_MSE_beats_origin=n['easy'] is not None and n['easy']<o['easy']-1e-12,
        easy_MSE_beats_selected_nearest=n['easy'] is not None and n['easy']<f['easy']-1e-12,
        all_MSE_not_worse_than_selected_nearest=n['all'] is not None and n['all']<=f['all']+1e-12,
        at_least12_of18_pair_wins=wins>=12)
    rebuilt=dict(views=72,equal_view_MSE=means,pair_wins=int(wins),magnitude_screen=conditions,
        magnitude_screen_pass=all(conditions.values()),
        occurrences={k:sum(r['scores']['local_signed32'][k] for r in rows) for k in
                     ('rows','known','unknown','supported','unsupported')},
        fitting_diagnostic_only=True,whole_pipeline_OOF=False,held_inference=False,
        independent_roles_read=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    compare(rebuilt,summary)
    return dict(arm_target_checks=targets,reference_checks=banks,donor_loss_checks=144,
                magnitude_conditions_checked=5,magnitude_screen_pass=all(conditions.values()))


def execute(check):
    reg,_,_=run.admission();done=json.loads((run.PUBLIC/'complete.json').read_text())
    require(done['registration_sha256']==run.sha(run.REG) and done['summary_sha256']==run.sha(run.PUBLIC/'summary.json'),'Freeze changed')
    prev=json.loads((run.parent.PUBLIC/'complete.json').read_text())
    refs={Path(r['path']).name:r for r in prev['groups']};rows=[]
    for entry in done['groups']:
        p=run.ROOT/entry['path']
        require(not p.is_symlink() and p.resolve().is_relative_to((run.PUBLIC/'groups').resolve()) and run.sha(p)==entry['sha256'],'Group binding changed')
        r=json.loads(p.read_text()); require(r['registration_sha256']==run.sha(run.REG),'Registration changed')
        prior=run.parent.PUBLIC/'groups'/p.name
        require(run.sha(prior)==refs[p.name]['sha256']==r['parent_group_sha256'],'Parent result changed')
        old=json.loads(prior.read_text())
        for arm in ('origin','selected_nearest32'):
            compare(r['scores'][arm],old['scores'][arm])
            require(r['prediction_sha256'][arm]==old['prediction_sha256'][arm],'Frozen prediction changed')
        require(len(r['references'])==len(old['references']),'Reference roster changed')
        for a,b in zip(r['references'],old['references']):
            fields=('query_sha256','donor_sha256','known_donors','causal_donors','supported','known_donors_by_source')
            if a['supported']:fields+=('mean_sha256','std_sha256','neighbor_sha256')
            require(all(a[k]==b[k] for k in fields),'Donors/neighbors/normalizer changed')
        rows.append(r)
    result=verify(rows,json.loads((run.PUBLIC/'summary.json').read_text()),run.pooled.previous.previous.parent.expected(reg))
    result.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),verifier_sha256=run.sha(Path(__file__)),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'),prior_control_comparisons=144,
        result_source='cached_verified_separate_receipt_arithmetic',new_inference=False,independent_replication=False)
    p=run.PUBLIC/'verification.json'
    if check:require(result==json.loads(p.read_text()),'Verification replay differs')
    else:run.once(p,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
