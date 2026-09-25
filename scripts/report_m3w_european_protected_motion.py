"""Complete, nonselective aggregate reporting for the protected-motion study."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.report_m3w_european_cv_reference import require_verification, equal_errors, value, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_protected_motion_v1'
OLD = ROOT/'outputs/publication_readiness_2026_09/european_conditional_risk_v1'
ARMS = ('independent', 'scene_uniform', 'joint', 'unary_exact', 'joint_exact')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compact_metric(m, *, scenes=False):
    keys = ('indexed_rows', 'supported_rows', 'unknown_rows', 'equal_scene_gain_percent',
            'worst_scene_gain_percent', 'scene_bootstrap_ci95')
    out = {k: m[k] for k in keys}
    if scenes:
        out['by_scene'] = m['by_scene']
    return out


def compact_description(m, *, scenes=False):
    return {k: compact_metric(v, scenes=scenes) if isinstance(v, dict) and 'by_scene' in v else v
            for k, v in m.items()}


def summary_metrics(r):
    policies = {n: dict(full=compact_description(v['full'], scenes=True),
        joint_population={k: compact_description(m) for k, m in v['joint_population'].items()},
        comparisons={k: compact_metric(m) for k, m in v['comparisons'].items()}, queries=v['queries'])
        for n, v in r['policies'].items()}
    return dict(result_source=r['result_source'], source_rows=r['source_rows'], joint_rows=r['joint_rows'],
        bootstrap_resamples=3000, bootstrap_unit='source_locality',
        uncertainty='conditional_source_development_not_independent_confirmation',
        new_heads=r['new_heads'], cached_heads=r['cached_heads'], new_neural_updates=r['new_neural_updates'],
        policies=policies,
        references={seed: {n: {k: compact_description(m) for k, m in v.items()}
                    for n, v in refs.items()} for seed, refs in r['references'].items()},
        neural_vs_damping={n: {k: compact_metric(m) for k, m in v.items()}
                            for n, v in r['neural_vs_damping'].items()},
        old_neural_decision_changes=r['old_neural_decision_changes'],
        candidate_counts={c: sum(n.split('_', 2)[1] == c for n in policies)
                          for c in ('neural', 'damping097')},
        deployment_changed=False, calibrated_safety=False, stage5c_executed=False, smc_enabled=False)


def accounting(result):
    require_verification(PUBLIC, result)
    replay = json.loads((PUBLIC/'checkpoint_replay.json').read_text())
    if len(replay['sampler_checks']) != 9 or not all(r['exact'] and r['compared_neural_heads']==6
            and r['total_draws']==512000 for r in replay['sampler_checks']):
        raise ValueError('Matching source draws across all six neural heads required')
    old = json.loads((OLD/'analysis.json').read_text())
    if sha(OLD/'analysis.json')!=result['identity']['previous_analysis_sha256']:
        raise ValueError('Previous metric identity mismatch')
    reference_checks = []
    for seed, refs in result['references'].items():
        for newname, oldname in [('CV','CV'),('damping097','fixed_damping097'),
                                 ('neural','neural'),('training_selected_baseline','training_selected_baseline')]:
            for field in ('ADE_vs_CV','FDE_vs_CV','positive_easy_ADE_vs_CV'):
                equal_errors(refs[newname]['full'][field], old['references'][seed][oldname]['full'][field])
            reference_checks.append(seed+'_'+newname)
    for name, policy in result['policies'].items():
        seed, candidate, tail = name.split('_',2)
        if candidate=='neural':
            equal_errors(policy['full']['ADE_vs_CV'],old['policies'][seed+'_'+tail]['full']['ADE_vs_CV'])
    packed = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
    receipt = json.loads((packed/'receipt.json').read_text())
    expected = result['identity']['previous_identity']['previous_identity']['previous_identity']['parent_identity']
    if receipt['identity']!=expected:raise ValueError('Packed source identity mismatch')
    arrays = {}
    for key in ('recordings','frames'):
        p = packed/(key+'.npy')
        if sha(p)!=receipt['arrays'][key]:raise ValueError('Query coordinates changed')
        arrays[key]=np.load(p,mmap_mode='r',allow_pickle=False)
    checks=[]
    for a in result['controls']:
        p=ROOT/a['path']
        if sha(p)!=a['sha256']:raise ValueError('Control receipt changed')
        r=json.loads(p.read_text());p=ROOT/r['path']
        if sha(p)!=r['sha256'] or r['used_future_inputs']:raise ValueError('Invalid causal decision receipt')
        with np.load(p,allow_pickle=False) as z:
            keys,inverse=np.unique(np.column_stack((arrays['recordings'][z['ids']],arrays['frames'][z['ids']])),axis=0,return_inverse=True)
            if len(keys)!=len(r['queries']):raise ValueError('Wrong query population')
            matched=0
            for i,q in enumerate(r['queries']):
                use=inverse==i
                if keys[i].tolist()!=[q['recording'],q['frame']] or int(use.sum())!=q['agents']:
                    raise ValueError('Wrong recording/frame/agent block')
                if q['matched']:
                    counts=[int(z[k][use].sum()) for k in ('independent','unary_exact','joint_exact')]
                    if len(set(counts))!=1 or counts[0]!=q['reference_count']:
                        raise ValueError('Actual intervention counts not matched')
                    if not z['matched'][use].all() or not np.all(z['matched_nonzero'][use]==(counts[0]>0)):
                        raise ValueError('Matched population mask disagrees')
                    matched+=1
            if not r['source_support_available'] and any(z[k].any() for k in ('pointwise',*ARMS)):
                raise ValueError('Unsupported source fold failed to abstain')
            checks.append(dict(path=a['path'],queries=len(keys),matched_count_checks=matched))
    if len(checks)!=144:raise ValueError('All candidate control receipts required')
    return dict(result_source='fresh_accounting_on_cached_verified_artifacts',analysis_sha256=sha(PUBLIC/'analysis.json'),
        reporter_sha256=sha(Path(__file__)),unchanged_reference_checks=reference_checks,
        unchanged_neural_pointwise_views=24,sampler_checks=replay['sampler_checks'],
        decision_checks=checks,all_passed=True,new_training=False)


def main():
    r=json.loads((PUBLIC/'analysis.json').read_text())
    audit=accounting(r)
    (PUBLIC/'accounting_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    summary = summary_metrics(r)
    summary['analysis_sha256'] = audit['analysis_sha256']
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(summary, separators=(',', ':'), allow_nan=False)+'\n')
    lines=['# Risk-Protected Motion Results','',
        '45 fresh damping heads; 45 cached_verified neural-candidate heads; both decision banks freshly executed.',
        'All 48 registered views are retained. No winner, threshold or deployment selected from this readout.',
        '', '## Full Pointwise Population','',
        '| Policy (seed/candidate/event/head/guard) | ADE vs CV (%) | Conditional locality CI | FDE vs CV (%) | Hard gain (%) | Easy degradation (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for name,p in r['policies'].items():
        m=p['full'];e=m['positive_easy_ADE_vs_CV']
        lines.append(f"| {name} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {value(m['FDE_vs_CV'])} | {value(m['hard_ADE_vs_CV'])} | {-e['equal_scene_gain_percent']:.6f} | {-e['worst_scene_gain_percent']:.6f} | {m['zero_CV']['harmed_rows']}/{m['zero_CV']['rows']} | {100*m['switch_rate']:.6f} |")
    lines+=['','## Direct Neural-versus-Damping Contrasts','',
        'Positive means neural wins. Same risk budget, not necessarily the same realized number of interventions between candidates.',
        '| Seed/event/head/guard | Population/rule | Neural gain over protected damping (%) | Conditional CI |','|---|---|---:|---|']
    for name,contrasts in r['neural_vs_damping'].items():
        for rule,m in contrasts.items():lines.append(f'| {name} | {rule} | {value(m)} | {ci(m)} |')
    lines+=['','## Joint Pilot','',
        '1,152 queries / 6,116 targets, versus 318,969 full pointwise targets. There are no zero-CV cases in the joint pilot.',
        '| Policy | Rule | ADE vs CV (%) | CI | Worst easy degradation (%) | Switch (%) |','|---|---|---:|---|---:|---:|']
    for name,p in r['policies'].items():
        for rule,m in p['joint_population'].items():
            lines.append(f"| {name} | {rule} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {-m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']:.6f} | {100*m['switch_rate']:.6f} |")
    lines+=['','## Within-Candidate Matched Counts','',
        '| Policy | Contrast | Gain (%) | CI |','|---|---|---:|---|']
    for name,p in r['policies'].items():
        for contrast,m in p['comparisons'].items():lines.append(f'| {name} | {contrast} | {value(m)} | {ci(m)} |')
    lines+=['','Undefined contrasts are kept, not converted to zero or recalculated after removing unsupported localities.',
        'Bootstrap: 3,000 locality resamples; conditional on shared source data/fitted models. Not independent confirmation.',
        'Easy <=2% and zero-CV added harm 0 remain unchanged. Empty support is not safety validation.',
        'Image-pixel raw-stride12 obs8/pred12; not t50, metric, seconds, physical safety, foundation or deployment evidence. Stage5C/SMC off.','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    lines=['# New Damping-Head Training Losses','',
        'Only damping heads are newly fitted. Cached neural-candidate head losses are in the preceding experiment.',
        'Training minibatch objectives differ between utility/all/easy tasks; compare downstream errors, not cross-task raw loss.',
        '', '| Head | Updates | First loss | Last logged loss | Fit seconds | Unique fitting draws |',
        '|---|---:|---:|---:|---:|---:|']
    for name,x in r['training'].items():
        if x['result_source']!='fresh_run':continue
        f=x['fit']
        if 'trace' in f:lines.append(f"| {name} | {f['step']} | {f['trace'][0]['loss']:.8f} | {f['trace'][-1]['loss']:.8f} | {f['seconds']:.4f} | {f['unique_training_rows']} |")
        else:lines.append(f'| {name} | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |')
    lines+=['','54,000 new neural updates; pilot included. Exact source draw sequence, support, weights and CV cost scale match across six neural heads per seed/fold.',
        'Features/targets and their fitted moments differ necessarily with the candidate. This is not new trajectory-model training.','']
    (PUBLIC/'training_losses.md').write_text('\n'.join(lines))
    print(json.dumps(dict(reference_checks=len(audit['unchanged_reference_checks']),
        sampler_groups=len(audit['sampler_checks']),decision_receipts=len(audit['decision_checks']),all_passed=True)))


if __name__=='__main__':main()
