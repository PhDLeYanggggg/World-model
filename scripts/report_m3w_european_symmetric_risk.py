"""Nonselective symmetric-risk reporting with causal decision reconstruction."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.report_m3w_european_cv_reference import require_verification, sha, value, ci, equal_errors
from scripts.report_m3w_european_protected_motion import compact_metric, compact_description
from src.world_model.m3w_european_conditional_risk import pointwise_rule

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_risk_v1'
ARMS = ('independent','scene_uniform','joint','unary_exact','joint_exact')


def load_artifact(a):
    path = ROOT/a['path']
    if sha(path)!=a['sha256']:
        raise ValueError('Artifact changed: '+a['path'])
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def accounting(r):
    require_verification(PUBLIC,r)
    replay = json.loads((PUBLIC/'checkpoint_replay.json').read_text())
    if len(replay['checks'])!=36 or not all(c['paired_sampler_exact'] and c['total_draws']==512000
            and c['unknown_draws']==0 for c in replay['checks']):
        raise ValueError('Matched complete sampling evidence required')
    packed = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
    packed_receipt = json.loads((packed/'receipt.json').read_text())
    parent = r['identity']['previous_identity']['previous_identity']['previous_identity']['previous_identity']['previous_identity']['parent_identity']
    if packed_receipt['identity']!=parent:
        raise ValueError('Source population identity mismatch')
    arrays = {}
    for name in ('recordings','frames','history'):
        path = packed/(name+'.npy')
        if sha(path)!=packed_receipt['arrays'][name]:
            raise ValueError('Accounting data changed')
        arrays[name] = np.load(path,mmap_mode='r',allow_pickle=False)
    oldpath = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_utility_v1/analysis.json'
    if sha(oldpath)!=r['identity']['previous_analysis_sha256']:
        raise ValueError('Prior control analysis changed')
    old = json.loads(oldpath.read_text())
    for key,artifacts in r['frozen_controls'].items():
        if artifacts!=old['frozen_controls'][key]:
            raise ValueError('Frozen control artifact mismatch')
        for a in artifacts.values():
            if sha(ROOT/a['path'])!=a['sha256']:
                raise ValueError('Frozen risk/control hash changed')
    for key,row in r['frozen_utility'].items():
        if row['artifacts']!=old['training'][key]['artifacts'] or row['lineage']!=old['training'][key]['lineage']:
            raise ValueError('Frozen symmetric utility changed')
        for a in row['artifacts'].values():
            if sha(ROOT/a['path'])!=a['sha256']:
                raise ValueError('Frozen utility hash changed')
    ridge_views=[k for k in r['decision_changes'] if '_ridge_' in k]
    if len(ridge_views)!=24 or any(v for k in ridge_views for counts in r['decision_changes'][k].values() for v in counts.values()):
        raise ValueError('Frozen ridge decisions must remain byte-exact')
    for key in ridge_views:
        for field in ('ADE_vs_CV','FDE_vs_CV','positive_easy_ADE_vs_CV'):
            equal_errors(r['policies'][key]['full'][field],old['policies'][key]['full'][field])
        for rule in ARMS:
            equal_errors(r['policies'][key]['joint_population'][rule]['ADE_vs_CV'],
                old['policies'][key]['joint_population'][rule]['ADE_vs_CV'])
    checks = []
    for a in r['controls']:
        path = ROOT/a['path']
        if sha(path)!=a['sha256']:
            raise ValueError('Decision receipt changed')
        receipt = json.loads(path.read_text())
        z = load_artifact(dict(path=receipt['path'],sha256=receipt['sha256']))
        if receipt['used_future_inputs']:
            raise ValueError('Future inputs are forbidden')
        candidate,fold,seed,event,tail = receipt['identity']['name'].split('_',4)
        guard = receipt['identity']['guard']
        arm = tail[:-(len(guard)+1)]
        key = f'{candidate}_{fold}_{seed}'
        utility = load_artifact(r['frozen_utility'][key]['artifacts']['scores'])
        risk_artifacts = r['frozen_controls'][f'{key}_{event}_ridge'] if arm=='ridge' else r['training'][f'{key}_{event}']['artifacts']
        risk = load_artifact(risk_artifacts['scores'])
        ids = z['pointwise_ids']
        np.testing.assert_array_equal(ids,utility['ids'])
        np.testing.assert_array_equal(ids,risk['ids'])
        scale = r['frozen_utility'][key]['lineage']['cost_scale']
        u = (utility['costs'][:,0]-utility['costs'][:,1])/scale
        moving = np.linalg.norm(np.diff(arrays['history'][ids],axis=1),axis=2).sum(1)>0
        available = guard=='no_guard' or receipt['identity']['source_support']['gate_available']
        np.testing.assert_array_equal(z['pointwise'],pointwise_rule(u,risk['moments'],moving,
            budget=.02,support_available=available))
        keys,inverse = np.unique(np.column_stack((arrays['recordings'][z['ids']],arrays['frames'][z['ids']])),axis=0,return_inverse=True)
        if len(keys)!=len(receipt['queries']):
            raise ValueError('Joint population mismatch')
        location = {int(row):i for i,row in enumerate(ids)}
        local = np.array([location[int(row)] for row in z['ids']])
        matched = 0
        for i,q in enumerate(receipt['queries']):
            use = inverse==i
            if keys[i].tolist()!=[q['recording'],q['frame']] or int(use.sum())!=q['agents']:
                raise ValueError('Joint query key mismatch')
            mass = risk['moments'][local[use],0].sum()
            for rule in ARMS:
                bits = z[rule][use]
                harm = np.dot(bits,risk['moments'][local[use],1])
                if bits.any() and (not available or mass<=0 or harm>.02*mass+1e-8):
                    raise ValueError('Actual joint decisions violate frozen predicted-risk bound')
            if q['matched']:
                counts = [int(z[k][use].sum()) for k in ('independent','unary_exact','joint_exact')]
                if len(set(counts))!=1 or counts[0]!=q['reference_count']:
                    raise ValueError('Actual counts are not matched')
                if not z['matched'][use].all() or not np.all(z['matched_nonzero'][use]==(counts[0]>0)):
                    raise ValueError('Actual matched-count support differs')
                matched+=1
        if not available and any(z[k].any() for k in ('pointwise',*ARMS)):
            raise ValueError('Unsupported fold did not abstain')
        checks.append(dict(path=a['path'],pointwise_exact=True,pointwise_rows=len(ids),
            queries=len(keys),matched_count_checks=matched,predicted_risk_bound_checked=True))
    if len(checks)!=144 or len(r['frozen_controls'])!=72 or len(r['frozen_utility'])!=18:
        raise ValueError('Incomplete fixed matrix accounting')
    return dict(result_source='fresh_accounting_on_cached_verified_models',analysis_sha256=sha(PUBLIC/'analysis.json'),
        reporter_sha256=sha(Path(__file__)),decision_checks=checks,verified_old_risk_heads=72,
        frozen_utility_controls=18,matched_new_checkpoints=36,unchanged_ridge_policy_views=24,all_passed=True)


def summary(r):
    return dict(result_source=r['result_source'],analysis_sha256=sha(PUBLIC/'analysis.json'),
        source_rows=r['source_rows'],joint_rows=r['joint_rows'],new_heads=36,new_neural_updates=72000,
        policies={n:dict(full=compact_description(p['full'],scenes=True),
            joint_population={k:compact_description(m) for k,m in p['joint_population'].items()},
            comparisons={k:compact_metric(m) for k,m in p['comparisons'].items()},queries=p['queries']) for n,p in r['policies'].items()},
        neural_vs_damping={n:{k:compact_metric(m) for k,m in x.items()} for n,x in r['neural_vs_damping'].items()},
        symmetric_vs_asymmetric={n:{k:compact_metric(m) for k,m in x.items()} for n,x in r['symmetric_vs_asymmetric'].items()},
        risk_reliability_file='risk_reliability.json',decision_changes=r['decision_changes'],
        bootstrap_resamples=3000,bootstrap_unit='source_locality',
        uncertainty='conditional_development_not_independent_confirmation',
        deployment_changed=False,calibrated_safety=False,stage5c_executed=False,smc_enabled=False)


def main():
    r = json.loads((PUBLIC/'analysis.json').read_text())
    audit = accounting(r)
    (PUBLIC/'accounting_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(summary(r),separators=(',',':'),allow_nan=False)+'\n')
    (PUBLIC/'risk_reliability.json').write_text(json.dumps(dict(analysis_sha256=sha(PUBLIC/'analysis.json'),
        result_source='fresh_source_reliability_diagnostic_not_calibration',localities=r['risk_reliability']),separators=(',',':'),allow_nan=False)+'\n')
    lines = ['# Symmetric Risk Results','',
        '36 newly trained Torch risk heads / 72,000 updates. All 18 utility heads and all trajectories frozen.',
        'Ridge risk and decision banks are cached_verified; metrics are recomputed, not called fresh optimization.',
        'All 48 registered views retained; no post-readout selection. Source development only, not reserved evaluation.',
        '', '## Full Population','',
        '| Seed/candidate/event/risk/guard | ADE gain vs CV (%) | Conditional CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harmed | Switch (%) |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for name,p in r['policies'].items():
        m=p['full']
        lines.append(f"| {name} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {value(m['FDE_vs_CV'])} | {value(m['hard_ADE_vs_CV'])} | {-m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']:.6f} | {m['zero_CV']['harmed_rows']}/{m['zero_CV']['rows']} | {100*m['switch_rate']:.6f} |")
    for title,field in [('New Risk vs Frozen Old Risk','symmetric_vs_asymmetric'),('Neural vs Protected Damping','neural_vs_damping')]:
        lines += ['', '## '+title,'','| View | Population/rule | Paired gain (%) | Conditional CI |','|---|---|---:|---|']
        for name,contrasts in r[field].items():
            for rule,m in contrasts.items():
                lines.append(f'| {name} | {rule} | {value(m)} | {ci(m)} |')
    lines += ['','## Joint Pilot','','| View | Rule | ADE gain (%) | CI | Worst easy degradation (%) | Switch (%) |','|---|---|---:|---|---:|---:|']
    for name,p in r['policies'].items():
        for rule,m in p['joint_population'].items():
            lines.append(f"| {name} | {rule} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {-m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']:.6f} | {100*m['switch_rate']:.6f} |")
    lines += ['','## Within-Candidate Joint Contrasts','','| View | Contrast | Gain (%) | CI |','|---|---|---:|---|']
    for name,p in r['policies'].items():
        for rule,m in p['comparisons'].items():
            lines.append(f'| {name} | {rule} | {value(m)} | {ci(m)} |')
    lines += ['','Undefined localities remain undefined; none are dropped to manufacture a CI.',
        'Matched counts apply within each candidate. The two candidates share predicted-risk budgets, not necessarily intervention counts.',
        '318,969 full targets; 6,116 joint targets / 1,152 queries. Joint contains zero zero-CV examples.',
        '3,000 locality resamples, conditional on shared development data/models; no independent safety certification.',
        'Image-pixel obs8/pred12 raw stride12. No t50, seconds, metric, true3D, foundation or deployment claim. Stage5C/SMC off.','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    lines = ['# Risk Training Losses','','Only the neural risk objective changed. These are minibatch fitting losses, not held-out accuracy.',
        'Raw MSE and underharm4 loss values are not directly comparable objectives.',
        '', '| Head | Updates | First loss | Last logged loss | Fit seconds | Known unique rows |','|---|---:|---:|---:|---:|---:|']
    for name,t in r['training'].items():
        f=t['fit']
        lines.append(f"| {name} | {f['step']} | {f['trace'][0]['loss']:.8f} | {f['trace'][-1]['loss']:.8f} | {f['seconds']:.3f} | {f['unique_training_rows']} |")
    lines += ['','The 100-update pilot is included in 72,000 updates. Every new head has exactly the old head\'s training draws, known-label support, preprocessing and initialization constants.',
        'All fits are completed before outer-source readout. No early stopping or checkpoint selection from that readout.','']
    (PUBLIC/'training_losses.md').write_text('\n'.join(lines))
    lines=['# Conditional Risk Reliability','',
        'Per-locality ADE-supported rows. New and old selected sets can differ. Unknown-label rows are reported separately.',
        'Predicted and realized ratios divide mean positive event harm by mean event CV-error mass.',
        'Zero event mass remains undefined; these are post-readout diagnostics, not fitted calibration or safety guarantees.',
        '', '| Candidate/fold/seed/event/head/guard/locality | Version | All supported | Selected supported | Selected unknown | All harm bias | Selected predicted ratio | Selected realized ratio |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def number(x):
        return 'undefined' if x is None else f'{x:.6f}'
    for name,pair in r['risk_reliability'].items():
        for version,x in pair.items():
            bias=x['all']['bias']
            lines.append(f"| {name} | {version} | {x['all']['rows']} | {x['selected']['rows']} | {x['selected_unknown_rows']} | {number(bias[1] if bias is not None else None)} | {number(x['selected']['predicted_ratio'])} | {number(x['selected']['realized_ratio'])} |")
    (PUBLIC/'risk_reliability_table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(all_passed=True,heads=36,policy_views=48,decision_receipts=144,
        summary_bytes=(PUBLIC/'summary_metrics.json').stat().st_size)))


if __name__=='__main__':
    main()
