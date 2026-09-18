"""Render aggregate tables and a figure without altering frozen scientific output."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/source_motion_quality_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/source_motion_quality_v1'
os.environ.setdefault('MPLCONFIGDIR', str(PRIVATE / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'source_motion_quality_v1'


def csv_write(name, rows):
    with (PUBLIC / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    artifacts = [PUBLIC / (name + '.json') for name in
                 ('preparation', 'probes', 'replay', 'analysis', 'verification')]
    digest = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    before = digest()
    a = json.loads((PUBLIC / 'analysis.json').read_text())
    probes = json.loads((PUBLIC / 'probes.json').read_text())
    magnitude = a['strata']['max_pixel_displacement']
    rows = []
    for label, value in magnitude.items():
        rows.append(dict(bin=label, rows=value['rows'], tracks=value['tracks'],
            recordings=value['records'], cv_error_share_percent=100*value['cv_error_share'],
            control_gain_percent=value['candidate']['unconditional']['gain_percent'],
            motion_loss_gain_percent=value['candidate']['motion_loss']['gain_percent']))
    csv_write('magnitude_table.csv', rows)
    csv_write('probe_table.csv', [dict(trial=t['trial'], site=t['identity']['site'],
        label=t['identity']['label'], arm=t['identity']['arm'], seed=t['identity']['seed'],
        training_rows=t['training_rows'], training_positive_rate=t['training_positive_rate'],
        **{k: v for k, v in t['metrics'].items() if k != 'calibration_curve'}) for t in probes['trials']])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    labels = list(magnitude)
    x = np.arange(len(labels))
    for ax, key, title, ylabel in [
            (axes[0, 0], 'rows', 'A. Future annotation excursion', 'Overlapping queries'),
            (axes[0, 1], 'cv_error_share_percent', 'B. Stationary CV error contribution', 'Share of summed normalized ADE (%)')]:
        values = [row[key] for row in rows]
        bars = ax.bar(x, values, color=['#999999'] + ['#267f87']*5)
        ax.bar_label(bars, fmt='%.0f' if key == 'rows' else '%.1f', padding=3, fontsize=10)
        ax.set(xticks=x, xticklabels=labels, xlabel='Maximum future displacement (annotation pixels)',
               ylabel=ylabel, title=title, ylim=(0, max(values)*1.18))
    for ax, label, title in [(axes[1, 0], 'nonzero', 'C. Any nonzero future change'),
                              (axes[1, 1], 'half_box_excursion', 'D. Excursion >= half an observed box diagonal')]:
        display_scale = 10000 if label == 'half_box_excursion' else 1
        for pos, arm in enumerate(('geometry', 'geometry_past_box')):
            row = a['probability_probe_summary'][label+'_'+arm]
            point = row['equal_site_brier_lift'] * display_scale
            low, high = np.asarray(row['conditional_four_site_ci95']) * display_scale
            ax.errorbar(point, pos, xerr=[[point-low], [high-point]], fmt='o',
                        color=['#267f87', '#b34c5c'][pos], capsize=5)
        ax.axvline(0, color='#777777', linewidth=1)
        ax.set(yticks=[0, 1], yticklabels=['Geometry', 'Geometry + past box'], ylim=(1.7, -.7),
               xlabel=('Equal-site Brier lift' + (' (units of 1e-4)' if display_scale > 1 else '')
                       + '\nvs training prevalence; higher is better'), title=title)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Source motion-quality diagnostic: 15,430 queries, four explored sites', fontsize=15)
    svg = PUBLIC / 'evidence.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(PRIVATE / 'evidence.png', dpi=150)
    plt.close(fig)
    assert before == digest(), 'Reporting changed frozen evidence'
    print(json.dumps(dict(frozen_artifacts_unchanged=5, rows=a['rows'],
                         probe_models=probes['models'], figure=str(PRIVATE/'evidence.png'))))


if __name__ == '__main__':
    main()
