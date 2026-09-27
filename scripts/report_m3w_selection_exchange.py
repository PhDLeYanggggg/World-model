"""Render an explicit negative-result decomposition without a new success gate."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selection_exchange as run
from scripts.report_m3w_fixed_floor_tail import ci


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())
    m = d['metrics']
    lines = ['# Frozen Selection Exchange Results', '', '## Material Passport', '',
        'Fresh outcome decomposition of 108 frozen groups / 216 dependent held views. All models and decisions are cached_verified; no new fitting or deployment.',
        'Twelve opened development localities, three forecasting seeds, obs8/pred12 rawstride12, image-local detector silver. Independent roles remain closed.', '',
        '## Benefit Versus Harm', '',
        '| New minus count-matched control | Contribution (%) [95% exploratory CI] |', '|---|---:|']
    for key in ('benefit_change_percent','harm_change_percent','net_ADE_gain_percent','predicted_utility_change_percent'):
        lines.append('| '+key+' | '+ci(m[key])+' |')
    lines += ['', 'Benefit change minus harm change exactly reconstructs the parent net ADE contrast. Each contribution uses the same control total error denominator; these are not selected-risk ratios.',
        'Negative harm change means less harm, but the benefit loss is larger. The frozen utility model also predicts less utility on the exchanged selections. A sign-only utility gate discards information about magnitude.', '',
        '## Source Breakdown', '',
        '| Locality | Benefit change | Harm change | Net ADE | Predicted utility change |', '|---|---:|---:|---:|---:|']
    for site in d['identity']['localities']:
        lines.append('| '+site+' | '+' | '.join(f"{m[k]['by_site'][site]:.6f}" for k in
            ('benefit_change_percent','harm_change_percent','net_ADE_gain_percent','predicted_utility_change_percent'))+' |')
    lines += ['', '## Action Coverage', '',
        '| Count per dependent held view | Locality-averaged estimate [95% CI] |', '|---|---:|']
    for key in ('new_only_rows','control_only_rows','common_rows','new_only_unknown','control_only_unknown'):
        lines.append('| '+key+' | '+ci(m[key])+' |')
    lines += ['', 'Unknown-label actions stay in the intervention count but have no inferred zero loss or safety outcome. Counts repeat producer/fit/seed views; they are not unique independent samples.', '',
        '## Interpretation', '',
        'The equal-count failure is primarily lost benefit in this additive accounting, not an aggregate increase in harm. This is not proof of a unique causal root mechanism. It supports testing utility-aware current-query allocation before another risk-head fit.',
        'The next experiment will freeze all estimators and compare independent admission, whole-query admission and constrained joint reassignment under the unchanged predicted all/easy 2% budgets. It must retain real outcome-risk checks: predicted feasibility is not safety.',
        'The candidate cohort is the retained simultaneous query rows, not a claim of complete all-visible-scene coverage. No independent role opening or held threshold search is authorized by this result.', '',
        '## Limits', '',
        'Bootstrap uses 3,000 draws over 12 locality means, after dependent-view averaging. This is post-readout development accounting, not a replacement primary or independent confirmation. No multiple-comparison-adjusted claim.',
        'No new training, deployment, metric, seconds, physical-safety, human-gold, true3D or foundation claim. Stage5C and SMC remain disabled.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4.3), layout='constrained')
    for i, (key, label) in enumerate((('benefit_change_percent','Benefit change'),
        ('harm_change_percent','Harm change'),('net_ADE_gain_percent','Net ADE gain'),
        ('predicted_utility_change_percent','Predicted utility change'))):
        v = m[key]; lo, hi = v['ci95']; p = v['point']
        ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt='o',capsize=4,color='#177b77')
    ax.set_yticks(range(4), ['Benefit change','Harm change','Net ADE gain','Predicted utility change'])
    ax.set_ylim(3.5,-.5); ax.axvline(0,color='#9d3f42',linestyle='--')
    ax.set_xlabel('Contribution relative to control total error (%)')
    ax.set_title('Less harm, but more benefit lost: frozen equal-count diagnosis')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(run.PUBLIC/'exchange_decomposition.png',dpi=150,metadata={'Software':'M3W frozen selection exchange'})
    plt.close(fig)
    print(json.dumps({k:dict(point=v.get('point'),ci95=v.get('ci95')) for k,v in m.items()}))


if __name__=='__main__':
    main()
