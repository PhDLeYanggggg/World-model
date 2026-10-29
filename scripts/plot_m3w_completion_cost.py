"""Public aggregate policy and training-loss figures; private raster previews."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import run_m3w_completion_cost as run


def save(fig, name):
    directory = run.PUBLIC/'figures'; directory.mkdir(exist_ok=True)
    path = directory/(name+'.svg'); fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/(name+'.png'), dpi=150); plt.close(fig)


def main():
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    check = json.loads((run.PUBLIC/'verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256'] != run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Verified full readout required')
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': run.NAME, 'axes.spines.top': False, 'axes.spines.right': False})
    modes = run.api.MODES; colors = ['#2864a0', '#bb4b36']; labels = ['Known-only', 'Completion endpoints']
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8))
    for i, mode in enumerate(modes):
        x = np.arange(2)+(i-.5)*.28; data = summary['modes'][mode]
        point = np.array([data['accuracy'][bank]['point'] for bank in run.api.BANKS])
        ci = np.array([data['accuracy'][bank]['CI95'] for bank in run.api.BANKS])
        axes[0].errorbar(x, point, yerr=np.stack((point-ci[:, 0], ci[:, 1]-point)), fmt='o',
                         capsize=4, color=colors[i], label=labels[i])
        risk = [100*data['safety'][bank]['supported_record_views']/data['safety'][bank]['required_record_views'] for bank in run.api.BANKS]
        axes[1].bar(x, risk, width=.24, color=colors[i], label=labels[i])
    axes[0].axhline(0, color='#777777', linewidth=.8); axes[0].set_ylabel('Known ADE gain over floor (%)')
    axes[1].axhline(100, color='#777777', linestyle=':', linewidth=.9)
    axes[1].set_ylim(0, 105); axes[1].set_ylabel('Required risk views supported (%)')
    for axis in axes:
        axis.set_xticks([0, 1], ['Physical bank', 'Physical + neural bank'])
        axis.grid(axis='y', color='#dddddd', linewidth=.6); axis.set_axisbelow(True)
    fig.suptitle('Completion-aware supervision fails the fitting safety gate', fontsize=13, y=.97)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .90), ncol=2, frameon=False)
    fig.text(.5, .035, 'Intervals: 3,000 locality-bootstrap draws, fitting only. Risk retains all 444 dependent recording views per mode.',
             ha='center', fontsize=8.6)
    fig.subplots_adjust(left=.095, right=.98, bottom=.17, top=.76, wspace=.29); save(fig, 'policy_comparison')

    traces = json.loads((run.PUBLIC/'training_traces.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5), sharex=True)
    for i, mode in enumerate(modes):
        rows = [r for r in traces['rows'] if r['mode'] == mode]
        steps = [r['step'] for r in rows[0]['trace']]
        if len(rows) != 36 or any([r['step'] for r in row['trace']] != steps for row in rows):
            raise ValueError('All 36 fixed complete traces required per supervision mode')
        losses = np.array([[v['loss'] for v in row['trace']] for row in rows])
        if not np.isfinite(losses).all() or (losses <= 0).any():
            raise ValueError('Positive finite losses required for logarithmic display')
        low, mid, high = np.quantile(losses, [.1, .5, .9], axis=0)
        axes[i].plot(steps, mid, color=colors[i]); axes[i].fill_between(steps, low, high, color=colors[i], alpha=.15)
        axes[i].set_title(labels[i]); axes[i].set_yscale('log'); axes[i].set_xlabel('Optimizer update')
        axes[i].grid(axis='y', color='#dddddd', linewidth=.6)
    axes[0].set_ylabel('Minibatch objective (log scale)')
    fig.suptitle('Real training traces: different targets, not comparable absolute losses', fontsize=12, y=.97)
    fig.text(.5, .035, 'Median and 10th-90th percentiles across 36 cost heads; descriptive spread, not a confidence interval.', ha='center', fontsize=8.6)
    fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.17, wspace=.22); save(fig, 'loss_traces')
    print(json.dumps(dict(public_generated_SVG=2, raw_data_images_exported=False)))


if __name__ == '__main__':
    main()
