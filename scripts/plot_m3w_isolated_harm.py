"""Plot verified aggregates without opening new inference data."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts import report_m3w_isolated_harm as report


def main():
    _, _, summary, details = report.validate()
    arms = report.evaluation.ARMS
    labels = ['Local (2k)', 'Anchor (2k)', 'Old risk (2k)', 'Utility +2k', 'Joint risk +2k', 'Isolated risk +2k']
    colors = ['#6b7280', '#374151', '#a45536', '#236ea8', '#7b4b93', '#22856c']
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    y = np.arange(len(arms))
    axes[0].barh(y, [details['held'][a]['mean_view_signed_MSE'] for a in arms], color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel('Mean held-view signed-score MSE')
    axes[0].set_title('Prediction error (not ADE/FDE)')
    axes[0].invert_yaxis()
    s = summary['scopes']['held']['safety']
    axes[1].barh(y, [s[a]['known_easy_violations'] for a in arms], color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(0, 72)
    axes[1].set_xlabel('Views violating known easy risk /72')
    axes[1].set_title('Lower is better; unknown risk separate')
    axes[1].invert_yaxis()
    controls = report.evaluation.COMPARATORS
    short = dict(anchor='Anchor (2k)', utility_continue='Utility +2k', joint_risk='Joint risk +2k', local='Local (2k)', old_risk='Old risk (2k)')
    for mode, offset, color in [('full', -.12, '#236ea8'), ('matched', .12, '#22856c')]:
        for i, other in enumerate(controls):
            c = summary['scopes']['held']['contrasts']['isolated_risk_minus_'+other+'_'+mode]
            if c['CI95'] is not None:
                axes[2].errorbar(c['mean'], i+offset,
                    xerr=[[c['mean']-c['CI95'][0]], [c['CI95'][1]-c['mean']]],
                    fmt='o', color=color, capsize=3, label=mode if i == 0 else None)
            else:
                axes[2].text(0, i+offset, 'undefined', fontsize=9)
    axes[2].set_yticks(range(len(controls)), [short[c] for c in controls])
    axes[2].invert_yaxis()
    axes[2].axvline(0, color='#777777', linewidth=.8)
    axes[2].set_xlabel('Paired lower utility difference, percentage points')
    axes[2].set_title('Isolated risk minus each control')
    axes[2].legend(loc='best')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Protected harm continuation: fixed TRAIN-internal recording transfer', fontsize=14)
    fig.text(.02, .035, '54 new fits; 144 fit/held views; 3,000 locality bootstrap draws. Nominal exposed-TRAIN intervals, not independent confirmation.\n'
        'Risk budget stays 2%. Easy violation counts exclude undefined ratios; full completion support is reported separately. No deployment change.', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .94))
    target = report.run.PRIVATE/'figures/isolated_harm_comparison.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)
    report.run.base.once(report.run.PUBLIC/'figure_receipt.json', dict(path=str(target.relative_to(report.run.ROOT)),
        sha256=report.run.base.sha(target), evidence_summary_sha256=report.run.base.sha(report.evaluation.PUBLIC/'summary.json'),
        source='cached_verified_aggregate_visualization', fresh_training=False, fresh_inference=False))
    print(target)
    reg, _, _ = report.run.admission()
    _, docs = report.evaluation.models(reg)
    traces = {a: [] for a in report.api.ARMS}
    for (_, _, arm), doc in docs.items():
        state = report.api.core.read_checkpoint(report.run.ROOT/doc['checkpoint']['path'])
        traces[arm].append(state['trace'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fields = [('primary_loss', 'Sampled primary moment / score loss'),
              ('regret', 'Sampled decision regret'), ('mean_soft_action', 'Mean soft intervention')]
    for arm, color in zip(report.api.ARMS, colors[-3:]):
        steps = [r['step'] for r in traces[arm][0]]
        for ax, (field, title) in zip(axes, fields):
            values = np.array([[r[field] for r in trace] for trace in traces[arm]])
            low, middle, high = np.quantile(values, [.25, .5, .75], axis=0)
            ax.plot(steps, middle, color=color, label=arm.replace('_', ' '))
            ax.fill_between(steps, low, high, color=color, alpha=.12)
            ax.set_title(title)
            ax.set_xlabel('Additional optimizer updates')
            ax.spines[['top', 'right']].set_visible(False)
    axes[-1].legend(fontsize=8)
    fig.suptitle('18 continuations per arm: median and interquartile range, not confidence intervals')
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.text(.02, .02, 'Minibatch traces, not full-census error or trajectory FDE. Signed dual loss excluded from cross-objective comparison.', fontsize=9)
    path = target.parent/'training_traces.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    report.run.base.once(report.run.PUBLIC/'training_figure_receipt.json', dict(path=str(path.relative_to(report.run.ROOT)),
        sha256=report.run.base.sha(path), source='cached_verified_training_traces', fresh_training=False))
    print(path)


if __name__ == '__main__':
    main()
