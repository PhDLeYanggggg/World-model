"""Deterministic tables for a frozen-action development diagnosis."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_m3w_selected_risk as run


def fmt(value, factor=1.):
    if value is None:return 'undefined'
    return f'{value*factor:.7g}'


def cell(report, factor=1.):
    text=fmt(report['mean'],factor)
    if report['ci95'] is not None:text+=' ['+', '.join(fmt(v,factor) for v in report['ci95'])+']'
    if report['undefined_views']:text+=f"; {report['undefined_views']} undefined views"
    return text


def main():
    d=json.loads((run.PUBLIC/'summary.json').read_text())
    completion=json.loads((run.PUBLIC/'completion.json').read_text())
    lines=['# Frozen-Action Risk Diagnosis','','## Material Passport','',
        '- fresh_run: residual, support and benefit/harm accounting on all108 frozen groups.',
        '- cached_verified:216 heads, predictors, protected floor and actions; no new training.',
        '- Twelve opened development localities; three forecasting seeds; independent roles remain closed.',
        '- Obs8/pred12, raw-frame stride12, image-local detector silver. No metric, seconds, human-gold, true3D/foundation or physical-safety claim.',
        '- No new policy, threshold search, deployment, Stage5C or SMC. Parent method screen remains failed.','',
        '## Signed-Risk Residuals','',
        'Optimism is actual minus predicted signed excess, in fitting-only cost-scale units. Positive means underprediction, not a percentage. The all/easy excess targets use the unchanged2%floor-error budget.',
        'Values first average nonempty dependent views within locality, then localities. Empty slices remain undefined and their counts are printed. No CI is computed when any expected view is undefined.',
        '', '| Arm / frozen slice | All optimism | Easy optimism | Controller proxy overlap % | Outside both fit sources % | Singleton query % |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in d['residuals'].items():
        lines.append('| '+name+' | '+cell(m['all_optimism_mean'])+' | '+cell(m['easy_optimism_mean'])+' | '+
            cell(m['controller_proxy_fraction'],100)+' | '+cell(m['outside_both_fit_sources_fraction'],100)+' | '+cell(m['singleton_query_fraction'],100)+' |')
    lines+=['','## Selection-Enrichment Contrasts','','| Arm / policy | All optimism: selected minus unselected | Easy optimism: selected minus unselected |','|---|---:|---:|']
    for key,m in d['selected_residual_contrasts'].items():
        lines.append('| '+key+' | '+cell(m['all_optimism_mean_selected_minus_unselected'])+' | '+cell(m['easy_optimism_mean_selected_minus_unselected'])+' |')
    lines+=['','Alleligible and selected sets differ in motion, support and outcomes. A residual contrast is descriptive selection enrichment, not a causal effect of the optimizer or proof of calibrated uncertainty.',
        '','## Benefit and Harm Exchanges','','Aggregate actions are compared against pointwise actions; rank counts match per query, while joint counts do not. All numbers below use the complete known floor-error denominator, not the selected-harm ratio.',
        '', '| Policy | Added benefit pp | Added harm pp | Removed benefit pp | Removed harm pp | Net gain change pp |','|---|---:|---:|---:|---:|---:|']
    for key,m in d['exchanges'].items():
        lines.append('| '+key+' | '+' | '.join(cell(m[k]) for k in ('added_benefit_over_floor_pp','added_harm_over_floor_pp',
            'removed_benefit_over_floor_pp','removed_harm_over_floor_pp','gain_change_over_floor_pp'))+' |')
    lines+=['','Net change = added benefit - added harm - removed benefit + removed harm. This identity is checked from row costs. Removing useful switches is not the same as introducing harmful switches.',
        '','## Selected-Query Evidence','','| Arm / policy | Selected queries/view | Complete-label selected queries/view | Unknown-label queries/view | Predicted safe, actual all-excess positive: repeated queries | Easy counterpart |','|---|---:|---:|---:|---:|---:|']
    for key,m in d['query_residuals'].items():
        arm,policy=key.split('/');counts=d['totals']['predicted_safe_actual_excess_queries'][arm][policy]
        lines.append('| '+key+' | '+cell(m['selected_queries'])+' | '+cell(m['complete_queries'])+' | '+cell(m['unknown_selected_queries'])+f" | {counts['all']} | {counts['easy']} |")
    lines+=['','The query counts repeat windows across dependent heads/seeds. They are not independent sample sizes. Incomplete selected-query labels are not used to claim a risk pass.',
        '','## Source and Support Limits','',
        'Nearest-descriptor support uses six causal descriptors, fitting-source-balanced standardization, and the other fitting source95thpercentile distance as a radius. Thresholds are never fitted on held rows. This is not full-model support, not a posterior variance and not a calibrated rejection rule.',
        'The two fitting sources are sparse domain support. Detector-label missingness and overlapping temporal rows remain limitations. Differences between fitting and held residuals mix in-sample fit bias, scene effects and model error; this experiment cannot identify these causes separately.',
        '','| Arm | In-sample fitting all optimism | In-sample fitting easy optimism |','|---|---:|---:|']
    for key,m in d['fitting_in_sample'].items():lines.append('| '+key+' | '+cell(m['all_optimism_mean'])+' | '+cell(m['easy_optimism_mean'])+' |')
    lines+=['','## Execution','',f"- Diagnostic elapsed: {completion['seconds']:.2f}s; peakRSS: {completion['peak_RSS_bytes']}bytes; PID{completion['pid']}.",
        '- Native arm64, CPU4/interop1/workers0; one hash-bound aggregate checkpoint per group.',
        '- Local execution is appropriate; approved CREATE queue checked read-only; no remote jobs submitted.',
        '- Summary contains3000locality bootstrap draws where complete, seed101531. Nominal development intervals, not multiplicity-adjusted confirmation.',
        '- Full legacy integration suite and cold raw reconstruction are not_run. Existing same-version parent replay evidence is reused, not relabeled new training.',
        '','## Relevant Prior Work','',
        'Smith and Winkler (2006), [The Optimizer\'s Curse](https://doi.org/10.1287/mnsc.1050.0451), show that selecting the highest noisy value estimate can introduce optimism even for conditionally unbiased estimates. I inspected the author-hosted paper\'s introduction and Proposition1 proof. Here risk-constrained joint selection and dependent errors do not establish those assumptions. This motivates a diagnostic hypothesis, not a theorem or a new contribution claim. No Bayesian correction from that paper is implemented.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    per=['# Locality Residual and Support Breakdown','','No locality removed or promoted to confirmation. These are already-opened development results.',
        '', '| Locality | Aggregate joint all optimism | Eligible all optimism | Aggregate joint outside-both % | Controller overlap % | Matched-rank gain change pp |','|---|---:|---:|---:|---:|---:|']
    sel=d['residuals']['subset_aggregate/joint_selected'];eligible=d['residuals']['subset_aggregate/eligible']
    for site in d['localities']:
        row=[sel['all_optimism_mean']['by_locality'][site],eligible['all_optimism_mean']['by_locality'][site],
             sel['outside_both_fit_sources_fraction']['by_locality'][site],sel['controller_proxy_fraction']['by_locality'][site],
             d['exchanges']['rank']['gain_change_over_floor_pp']['by_locality'][site]]
        per.append('| '+site+' | '+' | '.join(fmt(v,100 if i in (2,3) else 1) for i,v in enumerate(row))+' |')
    per+=['','An undefined locality-slice is not zero risk. Support-only rejection is not evaluated or tuned here.']
    (run.PUBLIC/'locality_breakdown.md').write_text('\n'.join(per)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    xs=list(range(12));labels=[s.removeprefix('european_') for s in d['localities']]
    for key,color,label in [('subset_pointwise','#59646d','pointwise'),('subset_aggregate','#168579','aggregate')]:
        v=d['residuals'][key+'/joint_selected']['all_optimism_mean']['by_locality']
        axes[0].plot(xs,[v[s] for s in d['localities']],marker='o',color=color,label=label)
        v=d['residuals'][key+'/joint_selected']['outside_both_fit_sources_fraction']['by_locality']
        axes[1].plot(xs,[None if v[s] is None else 100*v[s] for s in d['localities']],marker='o',color=color,label=label)
    for ax in axes:
        ax.set_xticks(xs,labels,rotation=75,fontsize=7);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    axes[0].axhline(0,color='#ac4545',linestyle='--');axes[0].set_title('Selected all-risk underprediction');axes[0].set_ylabel('Actual minus predicted / fitting cost scale')
    axes[1].set_title('Six-descriptor support diagnostic');axes[1].set_ylabel('Selected rows outside both fit sources (%)')
    fig.savefig(run.PUBLIC/'residual_support.png',dpi=150,metadata={'Software':'M3W frozen action diagnosis'});plt.close(fig)


if __name__=='__main__':main()
