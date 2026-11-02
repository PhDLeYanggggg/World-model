"""Render only the public, outcome-free calibration cohort inventory."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_independent_risk_calibration_v1'


def main():
    path = PUBLIC/'input_inventory.json'; inventory = json.loads(path.read_text())
    groups = sorted(inventory['by_locality']); values = inventory['by_locality']
    targets = np.array([values[g]['target_rows'] for g in groups])
    records = np.array([values[g]['recordings'] for g in groups])
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': 'm3w_calibration_cohort_v1', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.5), sharey=True,
                              gridspec_kw={'width_ratios': [1.4, 1]})
    y = np.arange(len(groups)); colors = ['#be493e' if t == targets.max() else '#337a72' for t in targets]
    axes[0].barh(y, targets, color=colors, height=.65)
    axes[0].set_xscale('log'); axes[0].set_xlim(900, 900000)
    axes[0].set_yticks(y, [g.replace('eu-locality-', '') for g in groups]); axes[0].invert_yaxis()
    axes[0].set_ylabel('Locality group'); axes[0].set_xlabel('Past-supported target windows (log scale)')
    for i, count in enumerate(targets):
        axes[0].text(count*1.08, i, f'{count:,}', va='center', fontsize=8)
    axes[1].plot(100*targets/targets.sum(), y, 'o', color='#be493e', label='Target-window share')
    axes[1].plot(100*records/records.sum(), y, 's', color='#375a8c', label='Recording share')
    axes[1].set_xlim(-2, 100); axes[1].set_xlabel('Share of registered cohort (%)')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(axis='x', color='#dddddd', linewidth=.5); ax.set_axisbelow(True)
    fig.suptitle('Risk-calibration cohort: 102 recordings in 12 locality groups', fontsize=11)
    fig.text(.5, .015, 'Obs8/pred12; raw stride 12; at most 256 query frames/recording. Windows are not independent units.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .94))
    public = PUBLIC/'figures'; private = PRIVATE/'figures'
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    svg, png = public/'calibration_locality_balance.svg', private/'calibration_locality_balance.png'
    fig.savefig(svg, metadata={'Date': None}); fig.savefig(png, dpi=170); plt.close(fig)
    result = dict(source='fresh_run_public_input_inventory_only_not_predictive_evidence',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        svg_sha256=hashlib.sha256(svg.read_bytes()).hexdigest(),
        target_rows=int(targets.sum()), locality_groups=len(groups),
        largest_locality_target_share=float(targets.max()/targets.sum()),
        labels_or_private_arrays_read=False)
    (public/'cohort_figure_receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(svg=str(svg), preview=str(png), **result)))


if __name__ == '__main__':
    main()
