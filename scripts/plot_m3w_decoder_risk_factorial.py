"""Render verified aggregate results; binary figures remain outside Git."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts import report_m3w_decoder_risk_factorial as report


def main():
    freeze, complete, summary, detail = report.validate()
    root = report.run.PRIVATE/'figures'
    root.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    arms = report.evaluation.ARMS
    labels = ['Local', 'Shared\nrisk', 'Shared\nutility', 'Source\nrisk', 'Source\nutility']
    x = np.arange(len(arms))
    colors = ('#176B83', '#C84B31')
    for scope, shift, color in (('fit', -.18, colors[0]), ('held', .18, colors[1])):
        values = [detail[scope][a]['mean_view_signed_MSE'] for a in arms]
        axs[0, 0].bar(x+shift, values, width=.34, color=color, label=scope.title())
        bad = [summary['scopes'][scope]['safety'][a]['known_easy_violations'] for a in arms]
        axs[0, 1].bar(x+shift, bad, width=.34, color=color, label=scope.title())
    axs[0, 0].set(yscale='log', ylabel='Mean view signed-score MSE (log scale)',
                  title='Cost prediction: fitting versus recording transfer')
    axs[0, 1].set(ylim=(0, 76), ylabel='Known easy-risk violations / 72 views',
                  title='Unchanged 2% selected positive-harm budget')
    for ax in axs[0]:
        ax.set_xticks(x, labels)
        ax.legend(frameon=False)
    pair_names = [right+'_minus_'+left for left, right in report.evaluation.PAIRS]
    pair_labels = ['Source risk - shared risk', 'Source utility - shared utility',
                   'Shared risk - shared utility', 'Source risk - source utility',
                   'Source risk - local', 'Source utility - local']
    for ax, kind, title in ((axs[1, 0], 'full', 'Full policy utility'),
                            (axs[1, 1], 'matched', 'Same-count utility')):
        for i, key in enumerate(pair_names):
            value = summary['scopes']['held']['contrasts'][key+'_'+kind]
            if value['CI95'] is None:
                ax.text(0, i, 'undefined', va='center')
                continue
            low, high = value['CI95']
            mean = value['mean']
            ax.errorbar(mean, i, xerr=[[max(0., mean-low)], [max(0., high-mean)]],
                        fmt='o', color='#343A40', capsize=4)
        ax.axvline(0, color='#888888', linewidth=1, linestyle='--')
        ax.set_yticks(range(len(pair_labels)), pair_labels)
        ax.invert_yaxis()
        ax.set(title=title, xlabel='Paired completion-lower utility, pp (not FDE)')
    fig.suptitle('Decoder sharing x risk objective: 72 fresh cost-head fits\n'
                 'Exposed TRAIN-internal evidence; 3,000 locality bootstrap draws; no deployment claim', fontsize=13)
    fig.savefig(root/'factorial_evidence.png', dpi=160)
    fig.savefig(root/'factorial_evidence.pdf')
    plt.close(fig)
    receipt = dict(summary_sha256=report.run.base.sha(report.evaluation.PUBLIC/'summary.json'),
        files={str(p.relative_to(report.run.ROOT)): report.run.base.sha(p)
               for p in (root/'factorial_evidence.png', root/'factorial_evidence.pdf')},
        result_source='cached_verified_aggregate_render', independent_roles_read=False)
    report.run.base.once(report.run.PUBLIC/'figure_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
