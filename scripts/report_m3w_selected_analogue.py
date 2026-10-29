"""Post-readout descriptive slices and local figure, with no selection changes."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import run_m3w_selected_analogue as run
from scripts import verify_m3w_selected_analogue as verifier


def aggregate(rows):
    result = {}
    for arm in verifier.ARMS:
        values = [r['scores'][arm] for r in rows]
        result[arm] = {k: sum(v[k] for v in values) for k in
            ('rows', 'known', 'unknown', 'supported', 'unsupported', 'benefit_harm_envelope_exceedances')}
        for target in ('all', 'easy'):
            part = {k: sum(v[target][k] for v in values) for k in
                ('events', 'observed_harm', 'predicted_harm', 'false_safe_rows')}
            for k in ('full_cohort_MSE', 'AUROC', 'AP'):
                numbers = [v[target][k] for v in values]
                part[k] = statistics.fmean(numbers) if None not in numbers else None
                part['undefined_'+k+'_views'] = numbers.count(None)
            result[arm][target] = part
    return result


def render(rows, complete_hash, reporter_hash):
    result = dict(all=aggregate(rows), complete_sha256=complete_hash, reporter_sha256=reporter_hash,
        posthoc_descriptive_only=True, result_source='cached_verified_fitting_receipt_slices',
        threshold_search=False, held_inference=False, independent_roles_read=False,
        new_inference=False, screen_changed=False, whole_pipeline_OOF=False)
    shared = [r for r in rows if r['scores']['nearest32']['unsupported'] == 0
              and r['scores']['nearest32']['known'] > 0]
    result['supported_complete_views'] = [(r['context'], r['source'], r['seed']) for r in shared]
    result['supported_complete_views'] = [list(key) for key in result['supported_complete_views']]
    result['supported_complete_diagnostic'] = aggregate(shared) if shared else None
    lines = ['# Fixed-Selection Analogue Slices', '',
        'Post-readout description, not model selection. Every view retains the original cohort.',
        'MSE is source-normalized, equal-view averaged. No unsupported view is removed.',
        'Counts sum repeated occurrences, not independent samples; no confirmatory CI.', '',
        f'Complete support occurs in {len(shared)}/72 views. Shared-support comparisons below the',
        'primary summary and in the figure are diagnostic only and cannot rescue its screen.', '',
        '| Group | Views | Original easy MSE | Estimation easy MSE | Mean easy MSE | Nearest32 easy MSE | Shuffled easy MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('context', 'seed', 'source'):
        result[key] = {}
        for val in sorted({r[key] for r in rows}):
            group = [r for r in rows if r[key] == val]
            ag = aggregate(group); result[key][str(val)] = dict(views=len(group), arms=ag)
            scores = [ag[a]['easy']['full_cohort_MSE'] for a in verifier.ARMS]
            lines.append(f'| {key}: {val} | {len(group)} | '+
                ' | '.join('undefined' if v is None else f'{v:.10g}' for v in scores)+' |')
    return result, '\n'.join(lines)+'\n'


def execute(check):
    complete = json.loads((run.PUBLIC/'complete.json').read_text()); rows = []
    for ref in complete['groups']:
        path = run.ROOT/ref['path']
        verifier.require(run.sha(path) == ref['sha256'], 'Frozen readout changed')
        rows.append(json.loads(path.read_text()))
    result, md = render(rows, run.sha(run.PUBLIC/'complete.json'), run.sha(Path(__file__)))
    path = run.PUBLIC/'slice_diagnostics.json'; table = run.PUBLIC/'slice_diagnostics.md'
    if check:
        verifier.require(json.loads(path.read_text()) == result and table.read_text() == md, 'Slice replay differs')
        figure = json.loads((run.PUBLIC/'figure_receipt.json').read_text())
        verifier.require(figure['sha256'] == run.sha(run.ROOT/figure['path']) and
                        figure['slice_sha256'] == run.sha(path), 'Figure binding changed')
        print('All descriptive slices and local figure hash verified; no new inference')
        return
    run.once(path, result)
    if table.exists(): verifier.require(table.read_text() == md, 'Table changed')
    else: table.write_text(md)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    diagnostic = result['supported_complete_diagnostic']
    verifier.require(diagnostic is not None, 'No complete-support diagnostic figure available')
    labels = ['Original', 'Estimation', 'Source mean', 'Nearest32', 'Shuffled32']
    colors = ['#66717b', '#b76833', '#84715c', '#007d78', '#a64f64']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    for ax, key, title, ylabel in zip(axes,
            ('full_cohort_MSE', 'predicted_harm', 'false_safe_rows'),
            ('Fixed-cohort easy-risk error', 'Predicted easy-harm mass', 'Known false-safe occurrences'),
            ('Equal-view normalized MSE', 'Sum after source normalization', 'Occurrences, not independent samples')):
        vals = [diagnostic[a]['easy'][key] for a in verifier.ARMS]
        for i, v in enumerate(vals):
            if v is not None:
                ax.bar(i, v, color=colors[i]); ax.text(i,v,f'{v:.4g}',ha='center',va='bottom',fontsize=8)
            else: ax.text(i,0,'undefined',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(5), labels, rotation=22, ha='right')
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max([v for v in vals if v is not None]+[1e-12])*1.25)
        if key == 'full_cohort_MSE': ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        if key == 'predicted_harm':
            observed = diagnostic['nearest32']['easy']['observed_harm']
            ax.axhline(observed,color='#222222',ls='--',lw=1,label=f'observed {observed:.4g}')
            ax.set_ylim(0,max(ax.get_ylim()[1],observed*1.2)); ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True); ax.grid(axis='y',alpha=.15)
    fig.suptitle('Cross-recording causal analogues: supported-view diagnostic only',fontsize=14)
    fig.supxlabel(f"{len(result['supported_complete_views'])}/72 complete-support views; not the primary screen, no policy execution or held readout",fontsize=10)
    p = run.PRIVATE/'selected_analogue.png'; fig.savefig(p,dpi=160); plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json',dict(path=str(p.relative_to(run.ROOT)),sha256=run.sha(p),
        reporter_sha256=run.sha(Path(__file__)),slice_sha256=run.sha(path),raw_data_in_figure=False))
    print(json.dumps(dict(slices=sum(len(result[k]) for k in ('context','seed','source')),figure=str(p))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true')
    execute(parser.parse_args().check)
