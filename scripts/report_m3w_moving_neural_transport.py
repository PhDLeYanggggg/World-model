"""Report all registered neural risk arms and actual training losses."""
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_neural_transport as run


def reduction(value,reference):
    return 100*(1-value/reference) if reference>0 else None


def summary(analysis):
    out=[]
    for c in analysis['contexts']:
        means=c['summary']['source_macro']; rows={}
        for arm in run.ARMS:
            values=means[arm]; ref=means['shared64']
            rows[arm]=dict(values,MSE_reduction_percent=reduction(values['normalized_MSE'],ref['normalized_MSE']),
                cost_reduction_percent=reduction(values['total_joint_cost'],ref['total_joint_cost']),
                admitted=sum(f['metrics'][arm]['accepted_known_proposals'] for f in c['folds']),
                harmful=sum(f['metrics'][arm]['false_safe_count'] for f in c['folds']))
        out.append(dict(job=c['job'],arms=rows,passes=c['summary']['passes_information_screen'],
            transport_cost_reduction_vs_plain_percent=reduction(means['neural_transport']['total_joint_cost'],means['neural_plain']['total_joint_cost'])))
    return dict(contexts=out,passing_contexts=analysis['passing_contexts'],
        primary_updates=sum(f['steps'] for c in analysis['contexts'] for f in c['folds']),
        evaluated_known_rows=sum(f['rows'] for c in analysis['contexts'] for f in c['folds']),
        minimum_fitting_epochs=min(f['minimum_fitting_visits'] for c in analysis['contexts'] for f in c['folds']),
        new_neural_risk_training=True,new_neural_trajectory_training=False,
        safety_certificate=False,independent_confirmation=False,deployment_changed=False,submission_ready=False)


def number(v):
    return 'NA' if v is None else f'{v:+.6f}'


def main():
    s=run.source; a=s.read(run.PUBLIC/'analysis.json')
    run.rel.base.exact(s.sha(run.PUBLIC/'analysis.json'),s.read(run.PUBLIC/'verification.json')['analysis_sha256'])
    result=summary(a); s.once(run.PUBLIC/'result_summary.json',result)
    lines=['# Neural Risk Transport: Trained, Not Promoted','',
        'I trained a source-only risk regressor and used its predicted difference',
        'between a query and its fixed references to adjust the reference outcomes.',
        'The same fitted network also supplies a plain prediction control. A ridge',
        'model separates linear adjustment from nonlinear learning. The features,',
        'reference IDs, source roles and original risk rules are unchanged.','',
        '**Transport helps relative to the plain network, but not enough to beat',
        'the existing forest or repair safe intervention.**','',
        '| Context | Model | Risk-MSE reduction % | Decision-cost reduction % | Admitted | Harmful |',
        '|---|---|---:|---:|---:|---:|']
    for c in result['contexts']:
        for name,r in c['arms'].items():
            lines.append(f"| {c['job'].split('_')[-1]} | {name} | {number(r['MSE_reduction_percent'])} | "
                f"{number(r['cost_reduction_percent'])} | {r['admitted']} | {r['harmful']} |")
    lines+=['','Positive reductions are better; shared64 is the reference. These are',
        'source-macro known-proposal decision costs, not ADE/FDE gains or the',
        'full-policy scene-level safety estimator.',
        f"The registered primary information screen passes **{result['passing_contexts']}/3** contexts.",
        'Secondary arms cannot retrospectively replace the primary. No deployment change.','',
        '## Actual Training','',
        f"Eight MLPs (9,890 parameters each), eight ridge fits and {result['primary_updates']:,} primary optimizer updates complete.",
        'Every donor-fitting row receives 32 epochs. The real pilot verifies exact',
        'one-epoch resume versus uninterrupted two-epoch training. All eight full',
        'neural and ridge fits then reproduce exactly. No NumPy fallback substitutes',
        'for the Torch networks. No new trajectory/dynamics model is trained.','',
        '| Excluded source | Fit rows | Updates | First logged fit loss | Epoch 32 fit loss | Source-held plain neural MSE |',
        '|---|---:|---:|---:|---:|---:|']
    trace=[]
    for c in a['contexts']:
        for f in c['folds']:
            first,last=f['loss_trace'][0],f['loss_trace'][-1]
            lines.append(f"| {f['source']} | {f['fit_rows']} | {f['steps']} | {first['weighted_fit_MSE']:.6f} | "
                f"{last['weighted_fit_MSE']:.6f} | {f['metrics']['neural_plain']['normalized_MSE']:.6f} |")
            trace.extend(dict(job=c['job'],source=f['source'],**r) for r in f['loss_trace'])
    lines+=['','Loss is weighted two-channel MSE after donor-only target-RMS normalization.',
        'The first log is after one optimizer update, not an untrained-model score.',
        'Epoch 32 is the fixed final checkpoint, not validation-selected. Loss still',
        'decreases late in several folds; convergence is not established. The large',
        'fit-versus-source-held gap also precludes assuming more epochs will fix transfer.',
        '[Complete loss trace](loss_trace.csv).','',
        '## Evidence Boundaries','',
        '- fresh_run: eight actual Torch risk fits, eight ridge controls, full refits and inference.',
        '- cached_verified: causal features, labels, fixed reference IDs, track32 and shared64 controls.',
        '- not_run: new neural trajectories, full-policy deployment, nested-C/H, independent confirmation.',
        '- All 48,061 known proposals remain; 1,274 unknown proposals are not fabricated safe.',
        '- References are source-excluded but their residuals are in-sample for the fitted donor model.',
        '- Residual transport is an established type of regression correction, not a novelty claim.',
        '- Original 2% all/easy harm, easy-net and support rules remain; empty support stays NA.',
        '- No identity exemption, original 24-group primary pass, reserved readout or submission readiness.',
        '- Native obs8/pred12 image-coordinate silver data, not metric/seconds/true3D/foundation.',
        '- Stage5C/SMC remain disabled.','',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)',
        '| [Related work and paper note](paper_addendum.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    with (run.PUBLIC/'loss_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(trace[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(trace)


if __name__=='__main__': main()
