"""Render complete channel-repair results, including exposed fitting diagnostics."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_channel_specialist as run

source, method = run.source, run.method


def summary(analysis):
    contexts=[]
    for context in analysis['contexts']:
        means=context['summary']['source_macro']; rows={}
        for arm in ('shared64',*method.ARMS):
            metrics=means[arm]; old=means['shared64']
            rows[arm]=dict(metrics,
                MSE_reduction_percent=100*(1-metrics['normalized_MSE']/old['normalized_MSE']),
                cost_reduction_percent=100*(1-metrics['total_joint_cost']/old['total_joint_cost']),
                admissions=sum(f['arms'][arm]['metrics']['accepted_known_proposals'] for f in context['folds']),
                harmful_admissions=sum(f['arms'][arm]['metrics']['false_safe_count'] for f in context['folds']),
                nodes=sum(f['arms'][arm]['nodes'] for f in context['folds']))
        contexts.append(dict(job=context['job'],arms=rows,passes=context['summary']['passes_information_screen']))
    return dict(contexts=contexts,passing_contexts=analysis['passing_contexts'],
        safety_certificate=False,deployment_changed=False,submission_ready=False)


def fitting_diagnostics(analysis):
    reference=source.read(run.audit.PUBLIC/'analysis.json'); records=[]
    for context, registered in zip(analysis['contexts'],reference['contexts']):
        run.rel.base.exact(context['job'],registered['job']); data=run.inputs(registered)
        for fold in context['folds']:
            fit=data['sites']!=fold['source']; row=dict(job=context['job'],source=fold['source'],arms={})
            for arm in ('shared64',*method.ARMS):
                state=run.rel.base.read_checkpoint(source.checked(fold['arms'][arm]['checkpoint']))['state']
                predict=method.parent.predict if arm=='shared64' else method.predict
                p=predict(state,data['x'][fit])
                row['arms'][arm]=dict(
                    fitting=method.parent.previous.metrics(p,data['target'][fit],data['weights'][fit],state['target_scale']),
                    source_held=fold['arms'][arm]['metrics'])
            records.append(row)
    return dict(records=records,result_source='fresh_run_posthoc_diagnostic_cached_verified_models',
        fitting_is_not_generalization=True,model_selection=False,new_V_or_H=False)


def main():
    a=source.read(run.PUBLIC/'analysis.json'); v=source.read(run.PUBLIC/'verification.json')
    run.rel.base.exact(v['analysis_sha256'],source.sha(run.PUBLIC/'analysis.json'))
    s=summary(a); source.once(run.PUBLIC/'result_summary.json',s)
    gap=fitting_diagnostics(a); source.once(run.PUBLIC/'fitting_gap.json',source.stable(gap))
    lines=['# Channel-Specific Risk Heads: No Safe Decision Repair','',
        'I separated the all-risk and easy-risk regression heads while keeping the',
        'original targets, causal features, weights and decision threshold. The primary',
        'uses two 32-tree forests against one shared 64-tree forest. This changes the',
        'model, not the original 2% evaluation contract.','',
        '| Context | Model | MSE reduction % | Decision-cost reduction % | Accepted | Harmful |',
        '|---|---|---:|---:|---:|---:|']
    for c in s['contexts']:
        for arm,r in c['arms'].items():
            lines.append(f"| {c['job'].split('_')[-1]} | {arm} | {r['MSE_reduction_percent']:+.6f} | {r['cost_reduction_percent']:+.6f} | {r['admissions']} | {r['harmful_admissions']} |")
    lines+=['','Positive reductions are better. The primary information screen passes',
        f"**{s['passing_contexts']}/3** contexts; two were required. No deployment promotion.",
        'These are source-macro known-proposal decision costs, not trajectory ADE',
        'improvement, independent test results or a full-policy safety certificate.','',
        '## What This Rejects','',
        'Removing shared partitions is not sufficient to repair safe admission.',
        'Context1 switches more often but harmful admissions increase from 1 to 19.',
        'All 113 primary admissions come from source076; their descriptive normalized',
        'known easy-harm/reference ratio is 5.820542%. The other sources admit none,',
        'which leaves empty support NA rather than safe. Context2 still rejects all',
        '38,579 known proposals and has worse MSE. The secondary shared32 control',
        'cannot retrospectively replace the primary and does not satisfy the screen.','',
        'The earlier ego-frame improvement occurred almost entirely in unchanged',
        'rejects. That boundary finding is preserved in the separate audit. Lower',
        'regression loss and more admissions are each insufficient outcomes.','',
        '## Scope And Checks','',
        '- fresh_run: 16 new model objects containing 24 forests, full refits and boundary attribution.',
        '- cached_verified: eight shared64 controls, original data/schema/ID bindings and producer lineage.',
        '- not_run: new neural dynamics, full-policy risk, nested-C/H or independent confirmation.',
        '- All 24 comparison cells refit exactly; shared32 equals the shared64 first-tree prefix.',
        '- Native arm64, CPU4, inter-op1, workers0; sklearn forest n_jobs4.',
        '- The reused runtime banner still says forest_threads1; actual registered forest setting is4.',
        '- Partial-label supervision and dependent windows remain; eight sources are not eight independent seeds.',
        '- Tree counts are matched, not nodes/parameters; every node count is retained in the JSON.',
        '- Original 2% all/easy harm, easy-net and support rules remain. No identity exemption.',
        '- Original24-group primary remains NA. Reserved Moving20, DUT, DroneCrowd and H stay closed.',
        '- Native obs8/pred12 silver image coordinates, not metric, seconds, true3D or foundation.',
        '- Stage5C/SMC disabled. No submission-ready claim.','',
        '## Fitting Versus Source-Held Diagnostics','',
        'The following posthoc comparison reuses completed models. It is not model',
        'selection or a new split. Training performance is optimistic by construction.','',
        '| Excluded source | Shared64 fit cost | Shared64 held cost | Separate64 fit cost | Separate64 held cost |',
        '|---|---:|---:|---:|---:|']
    for r in gap['records']:
        x,y=r['arms']['shared64'],r['arms']['separate64']
        lines.append(f"| {r['source']} | {x['fitting']['total_joint_cost']:.8g} | {x['source_held']['total_joint_cost']:.8g} | {y['fitting']['total_joint_cost']:.8g} | {y['source_held']['total_joint_cost']:.8g} |")
    lines+=['','[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
