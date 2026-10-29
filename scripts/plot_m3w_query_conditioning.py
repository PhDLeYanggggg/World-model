"""Render aggregate diagnostic evidence, not an independent performance figure."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts import audit_m3w_query_conditioning as runner


def execute():
    summary = json.loads((runner.PUBLIC/'summary.json').read_text())
    verification = json.loads((runner.PUBLIC/'verification.json').read_text())
    arms = ['self', 'peer']; names = ['Self-only', 'Peer context']
    fig, grid = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes = grid.ravel()
    x = np.arange(2); width = .32
    cohort = [summary['arms'][a]['cohorts']['self'] for a in arms]
    axes[0].bar(x-width/2, [c['easy']['predicted_harm'] for c in cohort], width,
                label='Predicted', color='#21867a')
    axes[0].bar(x+width/2, [c['easy']['observed_harm'] for c in cohort], width,
                label='Observed', color='#bd4c48')
    axes[0].set(title='Easy harm on identical self-selected rows',
                ylabel='Sum of source-normalized harm', xticks=x, xticklabels=names)
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2)
    axes[1].bar(x-width/2, [100*c['easy']['near_zero_jacobian_rows']/c['rows'] for c in cohort],
                width, label='Near-zero Jacobian rows', color='#6a5b96')
    axes[1].bar(x+width/2, [100*c['easy']['underprediction_near_zero_jacobian']/c['easy']['positive_underprediction']
                           for c in cohort], width, label='Underprediction at near-zero J', color='#c39b32')
    axes[1].set(title='Remaining local decoder obstruction', ylabel='Percent',
                xticks=x, xticklabels=names, ylim=(0, 100))
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.12), fontsize=8, ncol=2)
    for ax, metric, title in [(axes[2], 'total', 'One disposable AdamW step: total loss'),
                              (axes[3], 'common_self_easy_MSE', 'Same step: fixed-cohort easy-risk MSE')]:
        bottom = np.zeros(2)
        for key, label, color in [('better', 'Unclipped better', '#21867a'),
                               ('tied', 'Tied', '#aeb5bb'), ('worse', 'Unclipped worse', '#bd4c48')]:
            values = [verification['details'][a][metric][key] for a in arms]
            ax.bar(x, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set(title=title, ylabel='Probe batches (144 per arm)', xticks=x, xticklabels=names)
        ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen fitting-only diagnosis | repeated occurrences, not held performance', fontsize=13)
    directory = runner.PRIVATE/'figures'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/'conditioning_v2.png'; fig.savefig(path, dpi=160); plt.close(fig)
    runner.run.once(runner.PUBLIC/'figure_v2_receipt.json', dict(path=str(path.relative_to(runner.run.ROOT)),
        sha256=runner.run.sha(path), summary_sha256=runner.run.sha(runner.PUBLIC/'summary.json'),
        verification_sha256=runner.run.sha(runner.PUBLIC/'verification.json'),
        raw_data_in_figure=False, evidence_scope='fit_only_aggregate_diagnostic'))
    print(path)


if __name__ == '__main__':
    execute()
