"""Descriptive fitting slices and a local figure; never change routing."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import run_m3w_risk_decoupling as run
from scripts import verify_m3w_risk_decoupling as verifier


def aggregate(rows):
    result = {}
    for arm in verifier.ARMS:
        result[arm] = {}
        for cohort in verifier.COHORTS:
            entries = [r['result']['cohorts'][arm][cohort] for r in rows]
            result[arm][cohort] = dict(rows=sum(e['rows'] for e in entries),
                known=sum(e['known'] for e in entries), unknown=sum(e['unknown'] for e in entries))
            for target in ('all', 'easy'):
                values = [e[target]['MSE'] for e in entries]
                part = {k: sum(e[target][k] for e in entries) for k in
                    ('observed_harm', 'predicted_harm', 'observed_excess', 'predicted_excess', 'false_safe_rows')}
                part['equal_view_MSE'] = statistics.fmean(values) if None not in values else None
                result[arm][cohort][target] = part
        policies = [r['result']['policies'][arm] for r in rows]
        result[arm]['policy'] = dict(selected=sum(p['selected_count'] for p in policies),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and
                                      p['known_easy_positive_risk'] > .02+1e-12 for p in policies),
            full_support=sum(p['finite_completion_supported'] for p in policies))
    return result


def execute(check=False):
    public = run.PUBLIC
    complete = json.loads((public/'complete.json').read_text())
    rows = []
    for ref in complete['readouts']:
        path = run.ROOT/ref['path']
        verifier.require(run.sha(path) == ref['sha256'], 'Readout changed')
        rows.append(json.loads(path.read_text()))
    result = dict(all=aggregate(rows), posthoc_descriptive_only=True, threshold_search=False,
        held_inference=False, new_inference=False, routing_gate_changed=False,
        result_source='cached_verified_fitting_receipt_slices',
        complete_sha256=run.sha(public/'complete.json'), reporter_sha256=run.sha(Path(__file__)))
    table = ['# Descriptive Fitting Slices', '',
        'Post-readout description only. No slice was used to change the registered gate.',
        'MSE is source-normalized and equal-view averaged; rows are repeated occurrences.',
        'Counts are not independent samples. These slices have no confirmatory CI.', '',
        '| Group | Views | Original easy MSE | Coupled easy MSE | Estimation easy MSE | Estimation known easy violations |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ('context', 'seed', 'source'):
        result[key] = {}
        for value in sorted({r[key] for r in rows}):
            group = [r for r in rows if r[key] == value]; ag = aggregate(group)
            result[key][str(value)] = dict(views=len(group), arms=ag)
            values = [ag[a]['origin']['easy']['equal_view_MSE'] for a in verifier.ARMS]
            numbers = [f'{v:.9g}' if v is not None else 'undefined' for v in values]
            table.append(f'| {key}: {value} | {len(group)} | '+ ' | '.join(numbers) +
                f" | {ag['estimation']['policy']['known_easy_violations']}/{len(group)} |")
    path = public/'slice_diagnostics.json'; md = public/'slice_diagnostics.md'
    rendered = '\n'.join(table)+'\n'
    if check:
        verifier.require(result == json.loads(path.read_text()) and rendered == md.read_text(), 'Slice replay changed')
        print('Fitting slices reproduce; no new inference or routing change')
        return
    run.once(path, result)
    if md.exists():
        verifier.require(md.read_text() == rendered, 'Descriptive table changed')
    else:
        md.write_text(rendered)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = list(verifier.ARMS); colors = ['#555d66', '#a54a36', '#007d78']
    labels = ['Frozen original', 'Coupled continuation', 'Estimation only']
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), layout='constrained')
    for axis, cohort, title in zip(axes[:2], ('eligible', 'origin'),
            ('All eligible fitting rows', 'Fixed original selection')):
        values = [result['all'][a][cohort]['easy']['equal_view_MSE'] for a in arms]
        axis.bar(range(3), values, color=colors)
        axis.set_xticks(range(3), ['Original', 'Coupled', 'Estimation'])
        axis.set_ylabel('Easy signed-risk MSE (source-normalized)')
        axis.set_title(title, fontsize=11)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axis.set_ylim(0, max(values)*1.22)
        for i, value in enumerate(values): axis.text(i, value, f'{value:.3g}', ha='center', va='bottom', fontsize=9)
    values = [result['all'][a]['policy']['known_easy_violations'] for a in arms]
    axes[2].bar(range(3), values, color=colors)
    axes[2].set_xticks(range(3), ['Original', 'Coupled', 'Estimation'])
    axes[2].set_ylim(0, 78); axes[2].set_ylabel('Known easy-risk violations / 72 views')
    axes[2].set_title('Each arm\'s selected policy', fontsize=11)
    for i, v in enumerate(values): axes[2].text(i, v, f'{v}/72', ha='center', va='bottom')
    fig.suptitle('Lower average fitting error does not establish safe selection', fontsize=14)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
    fig.supxlabel('Exposed TRAIN only; frozen B/R outputs; no held evaluation or deployment claim', fontsize=10)
    figure = run.PRIVATE/'risk_decoupling.png'; fig.savefig(figure, dpi=160); plt.close(fig)
    run.once(public/'figure_receipt.json', dict(path=str(figure.relative_to(run.ROOT)), sha256=run.sha(figure),
        reporter_sha256=run.sha(Path(__file__)), slice_sha256=run.sha(path), raw_data_in_figure=False))
    print(json.dumps(dict(slices=sum(len(result[k]) for k in ('context', 'seed', 'source')), figure=str(figure))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    execute(parser.parse_args().check)
