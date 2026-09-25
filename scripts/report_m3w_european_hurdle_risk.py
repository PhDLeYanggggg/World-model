"""Publish complete matched hurdle comparisons and verified completion receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_m3w_european_protected_motion import compact_metric
from scripts.report_m3w_european_cv_reference import value,ci
from scripts.complete_m3w_european_producer_transport import contrast_counts

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_hurdle_risk_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_hurdle_risk_v1'
FIELDS=('replacements','neural_vs_damping','objective_pairs','envelope_pairs')
ARMS=('old','envelope','product_mse','hurdle')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify():
    a=json.loads((PUBLIC/'analysis.json').read_text());v=json.loads((PUBLIC/'verification.json').read_text())
    r=json.loads((PUBLIC/'head_replay.json').read_text())
    if not v['all_passed'] or v['analysis_sha256']!=sha(PUBLIC/'analysis.json') or v['identity']!=a['identity']:
        raise ValueError('Full exact metric replay required')
    if len(a['views'])!=144 or a['original_views_reproduced']!=72 or len(r['checks'])!=72 or r['identity']!=a['identity']:
        raise ValueError('Complete registered matrix and checkpoint replays required')
    for c in r['checks']:
        if not c['exact'] or not c['sampler_exact'] or c['rows']!=4096 or c['parameters']!=22979:
            raise ValueError('Head replay or architecture mismatch')
        if sha(ROOT/c['checkpoint']['path'])!=c['checkpoint']['sha256']:raise ValueError('Checkpoint changed')
    return a,r


def reports(a,r):
    out=dict(result_source=a['result_source'],analysis_sha256=sha(PUBLIC/'analysis.json'),new_heads=72,new_updates=144000,
        views={},deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    lines=['# Hurdle Risk: All Registered Views','',
        'Fresh72head fits with cached-verified forecasts and frozen utility. No arm selected from outcomes.',
        'Three folds/three seeds; eight producer-excluded development localities per view;3,000 conditional locality resamples.','',
        '| View | ADE gain vs CV (%) | Conditional95%CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Observed safety |',
        '|---|---:|---|---:|---:|---:|---|---:|---|']
    for k,v in a['views'].items():
        out['views'][k]={f:v[f] for f in ('zero_CV','decision_sha256','safety_observed_pass','switch_rate')}
        out['views'][k].update(ADE_vs_CV={s:compact_metric(m) for s,m in v['ADE_vs_CV'].items()},
            FDE_vs_CV=compact_metric(v['FDE_vs_CV']),opportunity=v['ledger']['summary'])
        worst=v['ADE_vs_CV']['easy']['worst_scene_gain_percent'];worst='undefined' if worst is None else f'{-worst:.6f}'
        lines.append(f"| {k} | {value(v['ADE_vs_CV']['all'])} | {ci(v['ADE_vs_CV']['all'])} | {value(v['FDE_vs_CV'])} | {value(v['ADE_vs_CV']['hard'])} | {worst} | {v['zero_CV']['harmed_rows']}/{v['zero_CV']['rows']} | {100*v['switch_rate']:.6f} | {v['safety_observed_pass']} |")
    for field in FIELDS:
        out[field]={k:{s:compact_metric(m) for s,m in v.items()} for k,v in a[field].items()}
        lines+=['','## '+field,'','Positive favors the named replacement or neural candidate; objective_pairs compare hurdle to product-MSE.','',
            '| Comparison | Subset | ADE gain (%) | Conditional95%CI |','|---|---|---:|---|']
        for k,v in a[field].items():
            for s,m in v.items():lines.append(f'| {k} | {s} | {value(m)} | {ci(m)} |')
    lines+=['','Detector-track image pixels,obs8/pred12 rawstride12. Not seconds,metric,human-gold,physical safety,independent confirmation,true3D or foundation.',
        'No Stage5C/SMC execution,threshold refit,calibration refit or deployment promotion.','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
    groups={}
    for candidate in ('neural','damping097'):
        groups[candidate]={}
        for arm in ARMS:
            rows=[v for k,v in a['views'].items() if k.startswith(candidate+'_') and k.endswith('_'+arm)]
            if len(rows)!=18:raise ValueError('All fold/seed/event cells required')
            groups[candidate][arm]=dict(views=18,safety_pass=sum(v['safety_observed_pass'] for v in rows),
                max_easy_degradation=max(-v['ADE_vs_CV']['easy']['worst_scene_gain_percent'] for v in rows),
                switch_range=[min(v['switch_rate'] for v in rows),max(v['switch_rate'] for v in rows)])
    contrasts={arm:{s:contrast_counts([v[s] for k,v in a['neural_vs_damping'].items() if k.endswith('_'+arm)])
        for s in ('all','easy','hard')} for arm in ARMS}
    for arm,row in contrasts.items():
        pos=[k for k,v in a['neural_vs_damping'].items() if k.endswith('_'+arm) and v['all']['scene_bootstrap_ci95'][0]>0]
        row['safe_positive_all_CI']=[k for k in pos if a['views']['neural_'+k]['safety_observed_pass']]
    obj={candidate:{s:contrast_counts([v[s] for k,v in a['objective_pairs'].items() if k.startswith(candidate+'_')])
        for s in ('all','easy','hard')} for candidate in ('neural','damping097')}
    (PUBLIC/'group_metrics.json').write_text(json.dumps(dict(groups=groups,neural_damping=contrasts,objective_pairs=obj),indent=2)+'\n')
    with (PUBLIC/'factor_reliability.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');fields=('rows','positive_rows','probability_mean','positive_rate','brier','ece','positive_fraction_mse')
        w.writerow(['view','locality','slice',*fields])
        for k,v in a['views'].items():
            for site,slices in v.get('factor_reliability',{}).items():
                for s,row in slices.items():w.writerow([k,site,s,*[row[x] for x in fields]])
    with (PUBLIC/'score_reliability.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['view','locality','slice','rows','reference_pred','harm_pred','reference_actual','harm_actual'])
        for k,v in a['views'].items():
            for site,slices in v['score_errors'].items():
                for s,row in slices.items():
                    vals=([*row['risk']['predicted_mean'],*row['risk']['actual_mean']] if row['rows'] else [None]*4)
                    w.writerow([k,site,s,row['rows'],*vals])
    with (PUBLIC/'training_losses.csv').open('w',newline='') as f:
        fields=('step','loss','moment_mse','bce','conditional_mse','positive_rows','gradient_norm')
        w=csv.writer(f,lineterminator='\n');w.writerow(['head',*fields])
        for check in r['checks']:
            cp=ROOT/check['checkpoint']['path'];report=json.loads((cp.parent/'complete.json').read_text())
            for row in report['fit']['trace']:w.writerow([check['head'],*[row[x] for x in fields]])
    prior_comparisons={}
    for k,v in a['views'].items():
        if not k.endswith('_hurdle'):continue
        cp=PRIVATE/'heads'/k/'complete.json'
        prior=json.loads(cp.read_text())['fit']['prior']['probability']
        sites={}
        for site,slices in v['factor_reliability'].items():
            row=slices['population'];rate=row['positive_rate']
            constant_brier=prior*prior-2*prior*rate+rate
            sites[site]=dict(rows=row['rows'],training_prior=prior,actual_rate=rate,
                constant_brier=constant_brier,model_brier=row['brier'],brier_gain=constant_brier-row['brier'])
        values=np.array([x['brier_gain'] for x in sites.values()])
        rng=np.random.default_rng(39271)
        draws=values[rng.integers(0,len(values),(3000,len(values)))].mean(1)
        prior_comparisons[k]=dict(by_locality=sites,equal_locality_brier_gain=float(values.mean()),
            conditional_locality_ci95=np.quantile(draws,[.025,.975]).tolist())
    (PUBLIC/'probability_prior_comparison.json').write_text(json.dumps(dict(
        status='posthoc_fixed_training_prior_diagnostic_not_arm_selection',
        comparisons=prior_comparisons,meaning='positive_absolute_Brier_difference_favors_hurdle',
        scope='opened_development_conditional_CI_no_multiplicity_correction'),indent=2)+'\n')


def complete(a,r):
    def bindings(x):
        if isinstance(x,dict):
            for key,v in x.items():
                if key=='bindings':
                    for f,h in v.items():
                        if sha(ROOT/f)!=h:raise ValueError('Frozen identity changed')
                else:bindings(v)
    bindings(a['identity'])
    events=[json.loads(s) for s in (PRIVATE/'events.jsonl').read_text().splitlines()]
    last={x['pid']:x for x in events}
    for pid,e in last.items():
        live=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
        if e['state']!='phase_complete' or 'run_m3w_european_hurdle_risk.py' in live.stdout:raise ValueError('Required phase not terminal')
    fits=[json.loads((ROOT/c['checkpoint']['path']).with_name('complete.json').read_text())['fit'] for c in r['checks']]
    if any(f['step']!=2000 or not f['complete'] or f['unknown_rows_sampled']!=0 for f in fits):raise ValueError('Training incomplete')
    prior=ROOT/'outputs/publication_readiness_2026_09/european_geometric_cost_v1/completion_checks.json'
    tests=json.loads(prior.read_text())['scoped_test_files']+['tests/test_m3w_hurdle_risk.py','tests/test_m3w_hurdle_reporting.py']
    run=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    (PUBLIC/'scoped_tests.txt').write_text(run.stdout+run.stderr)
    if run.returncode or not re.search(r'\b218 passed\b',run.stdout):raise ValueError('Scoped suite failed; inspect report')
    output=dict(result_source='fresh_run_training_replay_completion_checks',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        analysis_sha256=sha(PUBLIC/'analysis.json'),summary_sha256=sha(PUBLIC/'summary_metrics.json'),reporter_sha256=sha(Path(__file__)),
        training_trace_sha256=sha(PUBLIC/'training_losses.csv'),scoped_tests_sha256=sha(PUBLIC/'scoped_tests.txt'),
        new_heads=72,new_updates=sum(f['step'] for f in fits),parameters_each=22979,
        summed_head_fit_seconds=sum(f['seconds'] for f in fits),unknown_label_training_draws=0,
        checkpoint_replays=72,sampler_matches=72,replay_rows_each=4096,replay_sampling='first_held_index_rows_not_random',
        full_metric_views=144,original_controls_reproduced=72,scoped_tests_passed=218,scoped_test_files=tests,
        full_legacy_suite_run=False,required_terminal_pids=sorted(last),all_required_processes_finished=True,
        threshold_refit=False,calibration_refit=False,forecasts_unchanged=True,reserved_roles_opened=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    (PUBLIC/'completion_checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(heads=72,updates=144000,tests=218,processes_finished=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--complete',action='store_true');args=p.parse_args()
    a,r=verify();reports(a,r)
    if args.complete:complete(a,r)
    print(json.dumps(dict(views=len(a['views']),training_heads=72)))


if __name__=='__main__':main()
