"""Plot public fitting traces and complete development effects, without fitting."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selection_harm_deviance_v1'
CONTROL = ROOT/'outputs/publication_readiness_2026_09/european_selection_cardinal_cost_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selection_harm_deviance_v1'
ARMS = ('mean_absolute', 'easy_deviance', 'dual_deviance')
LABELS = ('Mean MSE control', 'Easy-only deviance', 'Dual H/EH deviance')
COLORS = ('#5a6677', '#258476', '#b9474b')


def main():
    read = lambda p:json.loads(Path(p).read_text())
    summary = read(PUBLIC/'summary.json'); verified = read(PUBLIC/'verification.json')
    if (not verified['all54_metric_groups_exact'] or verified['summary_sha256'] != hashlib.sha256((PUBLIC/'summary.json').read_bytes()).hexdigest()):
        raise ValueError('Verified complete readout required')
    fits, bindings = {}, {}
    trace_path = CONTROL/'training_traces.json'; export = read(trace_path)
    if export['training_freeze_sha256'] != hashlib.sha256((CONTROL/'training_freeze.json').read_bytes()).hexdigest():
        raise ValueError('Original full control trace freeze changed')
    control_traces = {r['trial']:r for r in export['models'] if r['arm'] == 'mean_absolute'}
    bindings[str(trace_path.relative_to(ROOT))] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    for arm in ARMS:
        folder = CONTROL if arm == 'mean_absolute' else PUBLIC
        paths = sorted((folder/'fits').glob('*_'+arm+'.json'))
        if len(paths) != 9: raise ValueError('All9real training traces required')
        fits[arm] = [read(p) for p in paths]
        if arm == 'mean_absolute':
            for fit in fits[arm]:
                trace = control_traces[fit['trial']]
                if (trace['checkpoint_sha256'] != fit['checkpoint']['sha256']
                        or trace['trace'][0] != fit['first_trace'] or trace['trace'][-1] != fit['final_trace']):
                    raise ValueError('Full original control trace identity differs')
                fit['trace'] = trace['trace']
        bindings.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plt.rcParams.update({'font.size':9, 'svg.fonttype':'none', 'svg.hashsalt':'m3w_harm_loss_v1',
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        for i, fit in enumerate(fits[arm]):
            trace = fit['trace'] if 'trace' in fit else None
            if trace is None:
                raise ValueError('Full public control trace export required, not interpolated first/last points')
            x = [r['step']/fit['budget'] for r in trace]
            y = [r['total'] if arm == 'mean_absolute' else r['original_total'] for r in trace]
            axes[0, 0].plot(x, y, color=color, alpha=.25, linewidth=.7, label=label if i == 0 else None)
    axes[0, 0].set_yscale('symlog', linthresh=.01)
    axes[0, 0].set_xlabel('Fraction of the complete query-coverage budget')
    axes[0, 0].set_ylabel('Original MSE + gain fitting objective')
    axes[0, 0].set_title('Common fitting loss, not validation')
    legend = axes[0, 0].legend(frameon=False, fontsize=8)
    for line in legend.get_lines(): line.set_alpha(1); line.set_linewidth(1.5)
    pos = np.arange(3)
    for i, (arm, color) in enumerate(zip(ARMS, COLORS)):
        stat = summary['accuracy'][arm]
        if stat['CI95'] is not None:
            lo, hi = stat['CI95']; point = stat['point']
            axes[0, 1].errorbar(i, point, yerr=[[point-lo], [hi-point]], fmt='o', color=color, capsize=4)
        safety = summary['safety'][arm]; value = safety['worst_known_easy_net_degradation_percent']
        if value is not None: axes[1, 1].bar(i, value, color=color, width=.55)
    axes[0, 1].set_xticks(pos, LABELS, rotation=15, ha='right')
    axes[0, 1].set_xlim(-.6, 2.6)
    axes[0, 1].set_ylabel('Equal-locality ADE gain over floor (%)')
    axes[0, 1].set_title('Complete exposed-development readout')
    axes[0, 1].axhline(0, color='#555555', linestyle=':', linewidth=.8)
    for i, (arm, color) in enumerate(zip(ARMS[1:], COLORS[1:])):
        stat = summary['effects'][arm]['completion_lower_pp']
        if stat['CI95'] is not None:
            lo, hi = stat['CI95']; point = stat['point']
            axes[1, 0].errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt='o', color=color, capsize=4)
    axes[1, 0].set_yticks([0, 1], LABELS[1:])
    axes[1, 0].axvline(0, color='#555555', linestyle=':', linewidth=.8)
    axes[1, 0].set_xlabel('Paired completion-lower effect vs control (pp)')
    axes[1, 0].set_title('Registered primary: dual arm, not best-arm selection')
    compact = ('Mean MSE\ncontrol', 'Easy-only\ndeviance', 'Dual H/EH\ndeviance')
    axes[1, 1].set_xticks(pos, [f'{label}\n{s["supported_record_views"]}/{s["required_record_views"]} pass'
        for label,s in ((label,summary['safety'][a]) for a,label in zip(ARMS,compact))], fontsize=8)
    axes[1, 1].axhline(2, color='#555555', linestyle='--', linewidth=1)
    axes[1, 1].set_xlim(-.6, 2.6)
    axes[1, 1].set_ylabel('Worst recording signed easy-net degradation (%)')
    axes[1, 1].set_title('Easy-net is only one required risk check')
    for ax in axes.flat: ax.grid(color='#dddddd', linewidth=.5); ax.set_axisbelow(True)
    fig.suptitle('Matched SOURCE harm-loss training: utility and original risk together', fontsize=12)
    fig.text(.5, .015, 'Thin loss lines: actual frozen-pipeline fits, not independent replications. Error bars: 3,000 locality resamples, 6 exposed groups.\n'
             'Available-label ADE is not full12 ADE. Undefined selected support is not a safe zero. No deployment promotion.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.18, right=.97, bottom=.17, top=.86, wspace=.48, hspace=.62)
    folder = PUBLIC/'figures'; private = PRIVATE/'figures'
    folder.mkdir(exist_ok=True); private.mkdir(exist_ok=True)
    svg = folder/'harm_loss_evidence.svg'; png = private/'harm_loss_evidence.png'
    fig.savefig(svg, metadata={'Date':None}); fig.savefig(png, dpi=170); plt.close(fig)
    (folder/'figure_receipt.json').write_text(json.dumps(dict(
        result_source='fresh_plot_of_verified_public_traces_and_development_statistics',
        input_hashes=bindings, summary_sha256=verified['summary_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        svg_sha256=hashlib.sha256(svg.read_bytes()).hexdigest(), private_arrays_or_weights_read=False,
        training_or_threshold_change=False, independent_confirmation=False), indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(svg=str(svg), preview=str(png))))


if __name__ == '__main__': main()
