"""Publish all registered head replacements and matched causal controls."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_cv_reference import value,ci
from scripts.complete_m3w_european_producer_transport import contrast_counts

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_geometric_cost_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(a):
    fields=('zero_CV','decision_sha256','safety_observed_pass','switch_rate')
    views={k:dict(**{f:r[f] for f in fields},ADE_vs_CV={s:compact_metric(m) for s,m in r['ADE_vs_CV'].items()},
        FDE_vs_CV=compact_metric(r['FDE_vs_CV']),raw_ADE_vs_CV=compact_metric(r['raw_ADE_vs_CV']),
        opportunity=r['ledger']['summary'],gain_capture_fraction=r['ledger']['gain_capture_fraction'])
        for k,r in a['views'].items()}
    comparisons={field:{k:{s:compact_metric(m) for s,m in r.items()} for k,r in a[field].items()}
        for field in ('replacements','neural_vs_damping')}
    return dict(result_source=a['result_source'],views=views,**comparisons,
        new_heads=54,new_updates=108000,forecast_training=False,threshold_refit=False,
        reserved_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    v=json.loads((PUBLIC/'verification.json').read_text())
    replay=json.loads((PUBLIC/'head_replay.json').read_text())
    if not v['all_passed'] or v['analysis_sha256']!=sha(PUBLIC/'analysis.json') or v['identity']!=a['identity']:
        raise ValueError('Exact complete metric replay required')
    if replay['identity']!=a['identity'] or len(replay['checks'])!=54 or any(
            not r['exact'] or not r['sampler_exact'] or r['rows']!=4096 for r in replay['checks']):
        raise ValueError('Every trained head must replay and match its old sampler')
    for r in replay['checks']:
        if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:
            raise ValueError('Changed fitted checkpoint')
    result=summary(a);result['analysis_sha256']=v['analysis_sha256']
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    lines=['# All Geometric-Cost Head Views','',
        'All144 registered views:three folds,three seeds,two candidates,two events,four head arms. No outcome selects a deployment.',
        'Each view has eight excluded development localities;3,000 locality-bootstrap draws provide conditional intervals,not independent confirmation.',
        'Old,utility_only,risk_only,both refer to which score heads use the causal geometric envelope. Forecasts remain unchanged.','',
        '| Candidate/fold/seed/event/arm | ADE gain vs CV (%) | Conditional CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Observed safety |',
        '|---|---:|---|---:|---:|---:|---|---:|---|']
    for key,r in a['views'].items():
        e=r['ADE_vs_CV']['easy']['worst_scene_gain_percent']
        worst='undefined' if e is None else f'{-e:.6f}'
        lines.append(f"| {key} | {value(r['ADE_vs_CV']['all'])} | {ci(r['ADE_vs_CV']['all'])} | {value(r['FDE_vs_CV'])} | {value(r['ADE_vs_CV']['hard'])} | {worst} | {r['zero_CV']['harmed_rows']}/{r['zero_CV']['rows']} | {100*r['switch_rate']:.6f} | {r['safety_observed_pass']} |")
    for field,title in [('replacements','Replacement Versus Its Old Head Pair'),('neural_vs_damping','Neural Versus Matched Protected Damping')]:
        lines+=['', '## '+title,'','Positive gain favors the replacement/neural policy. Shared localities and many contrasts preclude winner-selected confirmation.','',
            '| Comparison | Subset | ADE gain (%) | Conditional CI |','|---|---|---:|---|']
        for key,r in a[field].items():
            for subset,m in r.items():
                lines.append(f'| {key} | {subset} | {value(m)} | {ci(m)} |')
    lines+=['','Image-pixel released detector tracks,obs8/pred12 rawstride12,not t50,seconds,metric,human-gold,physical safety,true3D or foundation.',
        'No reserved readout,deployment promotion,Stage5C execution orSMC.','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    groups={}
    for candidate in ('neural','damping097'):
        for event in ('all','easy'):
            for arm in ('old','utility_only','risk_only','both'):
                rows=[r for k,r in a['views'].items() if k.startswith(candidate+'_') and k.endswith('_'+event+'_'+arm)]
                if len(rows)!=9:raise ValueError('All fold/seed cells required')
                gain=[r['ADE_vs_CV']['all']['equal_scene_gain_percent'] for r in rows]
                groups[candidate+'_'+event+'_'+arm]=dict(views=9,safety_pass=sum(r['safety_observed_pass'] for r in rows),
                    gain_range_percent=[min(gain),max(gain)],zero_switch_views=sum(r['switch_rate']==0 for r in rows))
    counts={field:{arm:{s:contrast_counts([r[s] for k,r in a[field].items() if k.endswith('_'+arm)])
        for s in ('all','hard','easy')} for arm in ('old','utility_only','risk_only','both')
        if any(k.endswith('_'+arm) for k in a[field])} for field in ('replacements','neural_vs_damping')}
    (PUBLIC/'group_metrics.json').write_text(json.dumps(dict(groups=groups,contrasts=counts),indent=2)+'\n')
    with (PUBLIC/'score_reliability.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['view','locality','slice','rows','mean_cv','gain_pred','gain_actual','harm_pred','harm_actual',
            'gain_mae','harm_mae','risk_mass_pred','risk_mass_actual','risk_harm_pred','risk_harm_actual',
            'risk_mass_mae','risk_harm_mae'])
        for key,r in a['views'].items():
            for site,site_rows in r['score_errors'].items():
                for subset,row in site_rows.items():
                    data=[key,site,subset,row['rows'],row['mean_cv']]
                    if row['rows']:
                        u,m=row['utility'],row['risk']
                        data += [u['predicted_mean'][0],u['actual_mean'][0],u['predicted_mean'][1],u['actual_mean'][1],
                            *u['mae'],m['predicted_mean'][0],m['actual_mean'][0],m['predicted_mean'][1],m['actual_mean'][1],*m['mae']]
                    else:data += [None]*12
                    writer.writerow(data)
    lines=['# Opportunity Accounting','',
        'Hindsight error decomposition only. Predictions/gates never consume these labels. All contributions share the locality CV denominator.','',
        '| View | Oracle benefit/CV (%) | Captured/CV (%) | Paid harm/CV (%) | Utility-rejected benefit/CV (%) | Risk-rejected benefit/CV (%) | Gain capture (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,r in a['views'].items():
        def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
        stats=[fmt(r['ledger']['summary'][n]['equal_locality']) for n in
            ('oracle_gain','captured_gain','switched_harm','missed_nonpositive_utility','missed_risk_veto')]
        capture=r['ledger']['gain_capture_fraction']
        lines.append(f"| {key} | {' | '.join(stats)} | {fmt(None if capture is None else 100*capture)} |")
    (PUBLIC/'opportunity_attribution.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(views=144,replacements=108,heads_replayed=54,new_updates=108000)))


if __name__=='__main__':
    main()
