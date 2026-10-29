"""Aggregate fitting statistics only; no dataset image or trajectory is exported."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import diagnose_m3w_forecast_amplitude as run


def main():
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    verification = json.loads((run.PUBLIC/'verification.json').read_text())
    if not verification['exact_replay'] or verification['summary_sha256'] != run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Verified complete readout required')
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': run.NAME, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), sharex=True)
    alpha = np.asarray(sorted(summary['amplitudes']))
    colors = {'physical_neural': '#2268ac', 'factorized': '#227b52'}
    labels = {'physical_neural': 'Mixed estimator', 'factorized': 'Separate experts'}
    for col, family in enumerate(('forest', 'neural')):
        for bank in colors:
            for suffix, style in (('', '-'), ('_same_set', '--')):
                name = bank + suffix
                stats = [summary['families'][family][str(a)] for a in alpha]
                values = [s['accuracy'][name]['point'] for s in stats]
                ci = np.asarray([s['accuracy'][name]['CI95'] for s in stats])
                label = labels[bank] + (' / fixed set' if suffix else ' / reselect')
                axes[0, col].plot(alpha, values, style, marker='o' if not suffix else 's',
                                  markersize=4, color=colors[bank], label=label)
                if not suffix:
                    axes[0, col].fill_between(alpha, ci[:, 0], ci[:, 1], color=colors[bank], alpha=.10)
                risk = [100*s['safety'][name]['supported_record_views']/s['safety'][name]['required_record_views'] for s in stats]
                axes[1, col].plot(alpha, risk, style, marker='o' if not suffix else 's', markersize=4, color=colors[bank])
        axes[0, col].set_title('Forest cost heads' if family == 'forest' else 'Neural cost heads')
        axes[0, col].axhline(0, color='#707070', linewidth=.8)
        axes[1, col].axhline(100, color='#a72c32', linewidth=.9, linestyle=':')
        axes[1, col].set_ylim(-2, 106)
        axes[1, col].set_xlabel('Fixed output amplitude alpha')
        for axis in axes[:, col]:
            axis.set_xscale('log', base=2)
            axis.set_xticks(alpha, ['1/16', '1/8', '1/4', '1/2', '1'])
            axis.grid(axis='y', color='#dddddd', linewidth=.6)
    axes[0, 0].set_ylabel('Known ADE gain over floor (%)')
    axes[1, 0].set_ylabel('Required risk views supported (%)')
    fig.suptitle('Frozen output amplitude: fitting evidence, not deployment', y=.985, fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .948), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Shading: 3,000 locality-bootstrap draws (fitting only). Risk requires every recording; unknown/empty views remain.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.095, right=.98, bottom=.11, top=.82, hspace=.17, wspace=.22)
    public = run.PUBLIC/'figures'; public.mkdir(exist_ok=True)
    path = public/'amplitude_controls.svg'
    fig.savefig(path, metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    preview = run.PRIVATE/'amplitude_controls.png'
    fig.savefig(preview, dpi=150); plt.close(fig)
    print(json.dumps(dict(public_SVG=str(path), private_preview=str(preview), raw_media_exported=False)))


if __name__ == '__main__':
    main()
