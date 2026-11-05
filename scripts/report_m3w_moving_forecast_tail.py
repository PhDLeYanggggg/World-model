"""Render completed frozen-tail diagnostics without fitting or selecting a policy."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_forecast_tail as run


def pct(v):
    return 'NA' if v is None else f'{100*v:.3f}%'


def tables(a):
    lines=['| Context | Signed-risk head | Positive-tail head | Envelope | Label oracle (diagnostic) | Internal screen |',
        '|---|---:|---:|---:|---:|---|']
    detail=['# Source-Held Tail Ranking','',
        'Each row excludes one own-fitting source. No new C/H or reserved test.',
        'Top10% refers to query-balanced weight, not an actual deployed subset.',
        'NA means no positive mass; it is not zero or a passing case.','',
        '| Context/source | Known rows | Easy rows | Unknown proposals | Signed | Tail | Envelope | Oracle | Tail on fitting sources |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i,c in enumerate(a['contexts']):
        s=c['summary']; r=s['source_macro_capture_at_10percent']['easy']
        lines.append(f"| {i} | {pct(r['signed_risk'])} | {pct(r['positive_tail'])} | {pct(r['envelope'])} | "
            f"{pct(r['oracle_diagnostic'])} | {'pass' if s['passes_information_screen'] else 'fail'} |")
        for f in c['folds']:
            r=f['ranking']['easy']; attr=f['attribution']
            vals=[pct(r[k]['0.1']) for k in run.method.SCORES]
            fit=pct(f['fitting_ranking']['easy']['positive_tail']['0.1'])
            detail.append(f"| {i}/{f['held_source']} | {f['rows']} | {attr['known_easy_rows']} | "
                f"{attr['unknown_proposals']} | {' | '.join(vals)} | {fit} |")
    return '\n'.join(lines), '\n'.join(detail)+'\n'


def figure(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True,layout='constrained')
    colors={'signed_risk':'#454545','positive_tail':'#097b76','envelope':'#bf5534','oracle_diagnostic':'#8e8e8e'}
    for i,(c,ax) in enumerate(zip(a['contexts'],axes)):
        for name in run.method.SCORES:
            values=[]
            for frac in run.method.FRACTIONS:
                rows=[f['ranking']['easy'][name][str(frac)] for f in c['folds']]
                values.append(float('nan') if any(v is None for v in rows) else 100*sum(rows)/len(rows))
            ax.plot([100*v for v in run.method.FRACTIONS],values,marker='o',color=colors[name],
                linestyle='--' if name=='oracle_diagnostic' else '-',label=name.replace('_',' '))
        ax.set_title(f'Context {i}'); ax.set_xlabel('Ranking weight budget (%)')
        ax.set_xticks([1,5,10,20]); ax.set_ylim(-2,102); ax.grid(alpha=.2)
    axes[0].set_ylabel('Easy positive-excess mass captured (%)')
    axes[-1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Source-held fitting diagnostic, not a deployment or safety result',fontsize=12)
    plt.rcParams['svg.hashsalt']='m3w_forecast_tail_v1'
    fig.savefig(run.PUBLIC/'tail_ranking.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'tail_ranking.png',dpi=170); plt.close(fig)


def main():
    a=run.source.read(run.PUBLIC/'analysis.json'); v=run.source.read(run.PUBLIC/'verification.json')
    run.rel.base.exact(v['analysis_sha256'],run.source.sha(run.PUBLIC/'analysis.json'))
    table,detail=tables(a)
    receipt={p:run.source.read(run.PUBLIC/(p+'_execution.json')) for p in ('pilot','run','verify')}
    rows=sum(c['known_proposed_rows'] for c in a['contexts'])
    unknown=sum(c['unknown_proposed_rows'] for c in a['contexts'])
    text=f'''# Can Past-Only Information Identify The Harmful Forecast Tail?

Eight positive-target forest probes are freshly trained on source-separated
own-B data. Eight signed-risk controls are cached_verified by model/data hashes
and exact predictions. All16 models also reproduce by a fresh full refit.
The frozen trajectory bank is reused, not retrained. This is exposed-development
information evidence, not an independent test or a deployable model.

## Fixed Primary Ranking Diagnostic

At10% query-balanced weight, capture of positive easy-channel budget-excess
mass is shown below. Scores are computed from past history and frozen causal
forecasts; future observations supply labels only. Source-macro means are used.

{table}

Internal screen passes **{a['passing_contexts']}/3**. The screen requires50% mass
capture and >=5 percentage points over both controls, in at least2contexts.
This does not replace the original safety gates. A label oracle is a diagnostic
upper ranking, not a learned predictor. Tie allocation is fractional and
label-independent; it does not describe actual intervention counts.

There are {rows:,} known proposals and {unknown:,} unknown proposals, across
eight source folds and three dependent contexts from one producer seed17.
Unknown futures are not assigned safe labels. The underlying packing is
64admitted SOURCE recordings /195286rows, not195286independent observations.

## What Was And Was Not Tested

New target: positive parts of the original two signed risk excesses; same120
causal inputs,64trees/depth8/minleaf32, fitting-only preprocessing/RMS and
original source/query weights. No source/action IDs or future fields are added.
The nonlearned envelope score is maximum predicted candidate-floor displacement
in the existing causal-normalized coordinates. No new hypothesis about
metres, seconds, true3D, foundation learning, JEPA or physical safety follows.

The main quantity is a cost-mass ranking diagnostic, **not ADE improvement,
easy degradation, tail-event probability, calibration or original risk ratio**.
Costs use available future labels and retain the old incomplete-label caveat.
The result cannot prove that all possible inputs/models are insufficient.

Policy evaluation: not_run, because this probe defines no deployment policy.
No newC/H or reserved readout, no threshold search, no identity certification,
no deployment change; original24group primary remains NA, not passed.
Stage5C and SMC stay off. Submission readiness remains false.

## Execution

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
'''
    for phase,r in receipt.items():
        text+=f"| {phase} | {r['pid']} | {r['seconds']:.3f} | {r['peak_RSS_bytes']} |\n"
    text+='''
Native arm64 `.venv-pytorch`, CPU4/inter-op1/workers0. The inherited runtime
banner says forest_threads=1; the actual registered forest uses n_jobs=4.
Local resources suffice, so no new CREATE job or data transfer was needed.
Exact repeats establish engineering reproducibility, not independent evidence.

See [source folds](source_folds.md), [causal slices](causal_slices.md),
[failure analysis](failure_analysis.md), [protocol](protocol.md),
[Chinese reproduction instructions](operation_zh.md), and the complete
[machine-readable analysis](analysis.json).

![Tail ranking](tail_ranking.svg)
'''
    slices=['# Attribution Without A New Policy','',
        'Original query/source weights, within-source shares of positive excess mass.',
        'Cuts use all proposed causal rows from fitting sources before label filtering.',
        'Quartile ties can yield empty bins; bins are descriptive, not decision thresholds.',
        'Amplitudes mix one frozen neural proposal with its floor. Source and action',
        'differences are confounded by frozen proposal selection, not causal effects.','',
        '| Context/source | Axis | Group | Rows | Weight | All positive share | Easy positive share |',
        '|---|---|---|---:|---:|---:|---:|']
    for i,c in enumerate(a['contexts']):
        for f in c['folds']:
            groups={'amplitude':f['attribution']['by_amplitude']}
            groups.update({k:v['rows'] for k,v in f['attribution']['causal_slices'].items()})
            for axis,items in groups.items():
                for r in items:
                    shares=r['positive_mass_share']
                    slices.append(f"| {i}/{f['held_source']} | {axis} | {r['group']} | {r['rows']} | "
                        f"{pct(r['weight'])} | {pct(shares[0])} | {pct(shares[1])} |")
    run.PUBLIC.joinpath('README.md').write_text(text)
    run.PUBLIC.joinpath('source_folds.md').write_text(detail)
    run.PUBLIC.joinpath('causal_slices.md').write_text('\n'.join(slices)+'\n')
    summary=dict(analysis_sha256=v['analysis_sha256'],known_proposals=rows,unknown_proposals=unknown,
        fresh_fits=8,cached_controls=8,all_refits_exact=True,
        passing_contexts=a['passing_contexts'],tail_branch_followup_supported=a['tail_branch_followup_supported'],
        context_summary=[c['summary'] for c in a['contexts']],execution=receipt,
        original_rules_unchanged=True,identity_certification=False,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,new_V_or_H=False,stage5c_executed=False,smc_enabled=False)
    run.PUBLIC.joinpath('result_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure(a)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
