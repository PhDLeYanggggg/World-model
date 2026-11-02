"""Plot the complete public calibration readout without fitting or selecting."""
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
ARMS = ('independent', 'frame_uniform', 'joint_matched', 'joint_free')
LABELS = ('Independent', 'Frame-uniform', 'Joint, matched count', 'Joint, free count')
COLORS = ('#345f9c', '#ad7918', '#258476', '#b9474b')


def validate(summary, allocation, taxonomy, paired):
    expected = [2**i for i in range(13)]
    if ([r['multiplier'] for r in summary['curves']] != expected
            or [r['multiplier'] for r in allocation['rows']] != expected
            or allocation['full_model_record_grids'] != 918
            or taxonomy['repeated_pipeline_record_views'] != 918
            or paired['complete_pipeline_record_views'] != 918):
        raise ValueError('Complete original calibration grid required')
    for curve in summary['curves']:
        if set(curve['policies']) != set(ARMS):
            raise ValueError('Every policy is shown, without post-outcome filtering')
        for arm, row in curve['policies'].items():
            if row['required_record_views'] != 918:
                raise ValueError('Every original recording/pipeline view required')
            matches = [r for r in taxonomy['rows']
                       if r['multiplier'] == curve['multiplier'] and r['arm'] == arm]
            if (len(matches) != 1 or sum(matches[0]['exclusive'].values()) != 918
                    or matches[0]['exclusive']['supported'] != row['supported_record_views']):
                raise ValueError('Failure categories must reconcile to the original gate')
    contrasts = {a+'_minus_independent' for a in ARMS if a != 'independent'}
    if (len(paired['comparisons']) != 39
            or {(r['multiplier'], r['contrast']) for r in paired['comparisons']}
            != {(m, c) for m in expected for c in contrasts}):
        raise ValueError('All prereported descriptive paired contrasts required')


def axis_style(ax):
    ax.set_xscale('log', base=2)
    ticks = [1, 4, 16, 64, 256, 1024, 4096]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.set_xlabel('Fixed risk multiplier')
    ax.grid(color='#dddddd', linewidth=.5)
    ax.set_axisbelow(True)


def series(ax, xs, statistics, color, label):
    values = np.array([np.nan if r['point'] is None else r['point'] for r in statistics])
    lows = np.array([np.nan if r['CI95'] is None else r['CI95'][0] for r in statistics])
    highs = np.array([np.nan if r['CI95'] is None else r['CI95'][1] for r in statistics])
    ax.plot(xs, values, 'o-', markersize=3, color=color, label=label, linewidth=1.3)
    ax.fill_between(xs, lows, highs, color=color, alpha=.08)


def main():
    names = ('summary.json', 'causal_allocation_report.json',
             'risk_failure_taxonomy.json', 'paired_calibration_effects.json')
    documents = [json.loads((PUBLIC/n).read_text()) for n in names]
    summary, allocation, taxonomy, paired = documents
    validate(*documents)
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none',
        'svg.hashsalt': 'm3w_independent_calibration_v1',
        'axes.spines.top': False, 'axes.spines.right': False})
    output = PUBLIC/'figures'; previews = PRIVATE/'figures'
    output.mkdir(parents=True, exist_ok=True); previews.mkdir(parents=True, exist_ok=True)
    generated = []

    def save(fig, name):
        svg, png = output/(name+'.svg'), previews/(name+'.png')
        fig.savefig(svg, metadata={'Date': None}); fig.savefig(png, dpi=170)
        plt.close(fig)
        generated.append(dict(path=str(svg.relative_to(ROOT)),
            sha256=hashlib.sha256(svg.read_bytes()).hexdigest(), private_preview=str(png.relative_to(ROOT))))

    xs = np.array([r['multiplier'] for r in summary['curves']])
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        rows = [r['policies'][arm] for r in summary['curves']]
        for ax, field in zip(axes[0], ('known_gain_percent', 'lower_gain_percent')):
            series(ax, xs, [r[field] for r in rows], color, label)
        axes[1, 0].plot(xs, [r['supported_record_views'] for r in rows], 'o-',
                       color=color, markersize=3, label=label)
        axes[1, 1].plot(xs, [100*r['actions'][arm]['switch_rate'] for r in allocation['rows']],
                       'o-', color=color, markersize=3, label=label)
    for ax in axes.flat:
        axis_style(ax)
    for ax in axes[0]:
        ax.axhline(0, color='#555555', linestyle=':', linewidth=.8)
    axes[0, 0].set_title('Available-label ADE gain')
    axes[0, 1].set_title('Wholly unknown-row completion lower bound')
    for ax in axes[0]:
        ax.set_ylabel('Equal-locality gain (%)')
    axes[1, 0].set_title('Original 2% gate and finite-support checks')
    axes[1, 0].set_ylabel('Supported recording/pipeline views')
    axes[1, 0].axhline(918, color='#555555', linestyle=':', linewidth=.8)
    axes[1, 0].set_ylim(-15, 950)
    axes[1, 1].set_title('Allocation-coded intervention rate')
    axes[1, 1].set_ylabel('Selected / repeated target views (%)')
    axes[1, 1].set_ylim(-1, 101)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .965), ncol=4, frameon=False)
    fig.suptitle('Dedicated-role calibration: complete frozen grid', fontsize=12, y=.995)
    fig.text(.5, .016, 'Conditional calibration readout, not independent confirmation. Bands: 3,000 locality resamples; 12 groups.\n'
             'Available-label ADE is not full12 ADE; completion bounds omit missing steps inside partial trajectories.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .925)); save(fig, 'calibration_risk_utility')

    categories = taxonomy['categories']
    shades = ('#b8b8b8', '#b9474b', '#ad7918', '#6a5e9c', '#258476')
    labels = ('Undefined reference', 'Observed error harm', 'Easy net only',
              'Unknown-bound only', 'Supported')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    for ax, arm, title in zip(axes, ('independent', 'joint_matched'), ('Independent', 'Joint, matched count')):
        rows = {r['multiplier']:r['exclusive'] for r in taxonomy['rows'] if r['arm'] == arm}
        bottom = np.zeros(len(xs))
        for key, color, label in zip(categories, shades, labels):
            counts = np.array([rows[int(m)][key] for m in xs])
            ax.bar(np.arange(len(xs)), counts, bottom=bottom, color=color, label=label, width=.82)
            bottom += counts
        ax.set_xticks(np.arange(len(xs)), [str(v) for v in xs], rotation=55)
        ax.set_xlabel('Fixed risk multiplier'); ax.set_title(title); ax.set_ylim(0, 930)
        ax.grid(axis='y', color='#dddddd', linewidth=.5); ax.set_axisbelow(True)
    axes[0].set_ylabel('Recording/pipeline views (total 918)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .95), ncol=5, frameon=False, fontsize=8)
    fig.suptitle('Exclusive failure priority; overlapping causes retained in the JSON', fontsize=11)
    fig.text(.5, .015, 'Undefined reference is not a safe zero. Observed harm means increased forecast error, not physical damage.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .88)); save(fig, 'calibration_failure_taxonomy')

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for arm, label, color in zip(ARMS[1:], LABELS[1:], COLORS[1:]):
        rows = {r['multiplier']:r['effects'] for r in paired['comparisons']
                if r['contrast'] == arm+'_minus_independent'}
        for ax, field in zip(axes, ('known_pp', 'completion_lower_pp')):
            series(ax, xs, [rows[int(m)][field] for m in xs], color, label)
    for ax in axes:
        axis_style(ax); ax.axhline(0, color='#555555', linestyle=':', linewidth=.8)
        ax.set_ylabel('Paired gain difference (percentage points)')
    axes[0].set_title('Available-label utility vs independent')
    axes[1].set_title('Direct policy-pair completion lower bound')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.suptitle('Same-bank descriptive contrasts, not another parameter search', fontsize=11)
    fig.text(.5, .015, 'Bands: paired locality bootstrap (3,000 draws). Unknown-row disagreement is computed directly between policies.\n'
             'Only the joint-matched arm has the same per-query intervention count as independent.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .87)); save(fig, 'calibration_paired_utility')
    receipt = dict(result_source='fresh_run_visualization_of_public_calibration_statistics',
        inputs={n:hashlib.sha256((PUBLIC/n).read_bytes()).hexdigest() for n in names},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), figures=generated,
        selected_multiplier_unchanged=summary['selection']['selected_multiplier'],
        policy_or_threshold_fit=False, labels_or_private_arrays_read=False,
        confirmation_access=False, deployment_changed=False)
    (output/'calibration_readout_figure_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(figures=generated)))


if __name__ == '__main__':
    main()
