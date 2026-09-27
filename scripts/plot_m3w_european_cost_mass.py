"""All registered cost-readout contrasts; no favorable assignment selection."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cost_mass as run
from scripts.report_m3w_european_cap_auxiliary_cost import PRIMARY, GUARDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def panels(doc, pairs, metric, title, name):
    fig, axes = plt.subplots(2, len(pairs), figsize=(6*len(pairs), 9),
        squeeze=False, constrained_layout=True)
    for row, family in enumerate(('full', 'motion_only')):
        for col, (contrast, label) in enumerate(pairs):
            ax = axes[row, col]; values = doc['contrasts'][contrast][family]
            for i, metrics in enumerate(values.values()):
                m = metrics[metric]
                if 'CI' not in m:
                    ax.text(0, i, 'not estimable'); continue
                p, (lo, hi) = m['point'], m['CI']
                ax.errorbar(p, i, xerr=np.array([[p-lo], [hi-p]]), fmt='o',
                    color='#00796b' if lo > 0 else '#b23b3b' if hi < 0 else '#555555', capsize=4)
            ax.axvline(0, color='black', linewidth=.8)
            ax.set_yticks(range(len(values)),
                [k.replace('producer','P').replace('_controller',' / C') for k in values])
            ax.set_title(f'{family}\n{label}')
            ax.set_xlabel('Paired improvement (metric-native units)')
            ax.grid(axis='x', alpha=.2)
    fig.suptitle(title+'\n3 seeds; 3,000 four-locality resamples; exposed source development', fontsize=13)
    fig.savefig(run.PUBLIC/(name+'.svg'), metadata={'Date': None})
    fig.savefig(run.PRIVATE/(name+'.png'), dpi=120)
    plt.close(fig)


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt':'m3w-cost-mass-v1','font.size':10})
    for reference in ('raw','scaled'):
        panels(doc, [(f'mass_{arm}_vs_{reference}_{arm}', f'{arm}: mass vs {reference}')
            for arm in ('cost_only','cap_aux','shuffled_aux')], PRIMARY,
            'Expected easy-harm MSE improvement (%), not trajectory gain', 'mass_vs_'+reference)
    panels(doc, [('mass_true_vs_mass_cost','True auxiliary vs cost-only'),
        ('mass_true_vs_mass_shuffled','True auxiliary vs shuffled')], PRIMARY,
        'Auxiliary information under matched moment readouts: MSE gain (%)', 'auxiliary_mass')
    for i, metric in enumerate(GUARDS):
        panels(doc, [('mass_cost_only_vs_raw_cost_only','Cost-only mass vs raw'),
            ('mass_cost_only_vs_scaled_cost_only','Cost-only mass vs L2')], metric,
            metric+' (positive favors mass; no physical-safety claim)', f'mass_guard_{i}')


if __name__ == '__main__': main()
