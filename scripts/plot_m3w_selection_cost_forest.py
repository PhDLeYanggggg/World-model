"""Plot complete matched forest evidence using public scalars only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selection_cost_forest_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selection_cost_forest_v1'
ARMS = ('mean_absolute','row_forest','context_forest')
LABELS = ('Neural mean\ncontrol','Row\nforest','Context\nforest')
COLORS = ('#5c6673','#2c8476','#b74b4d')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    read = lambda name:json.loads((PUBLIC/name).read_text())
    summary = read('summary.json'); verified = read('verification.json'); freeze = read('training_freeze.json')
    if (verified['summary_sha256'] != sha(PUBLIC/'summary.json') or not all(verified[k] for k in
            ('all18_SOURCE_cost_inferences_exact','all18_selection_cost_inferences_exact','all54_metric_groups_exact'))):
        raise ValueError('Verified complete result required')
    traces = {a:[] for a in ARMS[1:]}; hashes = {}
    for ref in freeze['fits']:
        p = ROOT/ref['path']
        if not p.is_relative_to(PUBLIC/'fits') or sha(p) != ref['sha256']:
            raise ValueError('Original public fit receipts required')
        r = json.loads(p.read_text()); traces[r['arm']].append(r['trace']); hashes[ref['path']] = ref['sha256']
    if any(len(cells) != 9 for cells in traces.values()):
        raise ValueError('All eighteen complete fits required')
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':'m3w_selection_cost_forest_v1',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(12.4,4.5))
    for arm,color,label in zip(ARMS[1:],COLORS[1:],LABELS[1:]):
        for i,trace in enumerate(traces[arm]):
            axes[0].plot([r['trees'] for r in trace],[r['fixed_SOURCE_monitor_objective'] for r in trace],
                         color=color,alpha=.35,linewidth=.8,label=label.replace('\n',' ') if i == 0 else None)
    axes[0].set_xlabel('Trees per complete 128-tree fit')
    axes[0].set_ylabel('Fixed SOURCE monitor objective')
    axes[0].set_title('Fitting monitor, not validation')
    legend = axes[0].legend(frameon=False,fontsize=8)
    for line in legend.get_lines(): line.set_alpha(1)
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        r = summary['accuracy'][arm]
        if r['CI95'] is not None:
            lo,hi = r['CI95']; value = r['point']
            axes[1].errorbar(i,value,yerr=[[value-lo],[hi-value]],color=color,fmt='o',capsize=3)
    axes[1].axhline(0,color='#555555',linestyle=':',linewidth=.8)
    axes[1].set_xticks(range(3),LABELS); axes[1].set_xlim(-.6,2.6)
    axes[1].set_ylabel('Equal-locality ADE gain over floor (%)')
    axes[1].set_title('Exposed-development utility')
    for i,(arm,color) in enumerate(zip(ARMS[1:],COLORS[1:])):
        r = summary['effects'][arm]['completion_lower_pp']
        if r['CI95'] is not None:
            lo,hi = r['CI95']; value = r['point']
            axes[2].errorbar(value,i,xerr=[[value-lo],[hi-value]],color=color,fmt='o',capsize=3)
    axes[2].axvline(0,color='#555555',linestyle=':',linewidth=.8)
    axes[2].set_yticks([0,1],['Row','Context'])
    axes[2].set_xlabel('Paired completion-lower effect (pp)')
    axes[2].set_title('Primary: context minus control')
    for ax in axes: ax.grid(color='#dddddd',linewidth=.5); ax.set_axisbelow(True)
    risk = '; '.join(f'{label.replace(chr(10)," ")}: {summary["safety"][a]["supported_record_views"]}/'
        f'{summary["safety"][a]["required_record_views"]}' for a,label in zip(ARMS,LABELS))
    fig.suptitle('Matched current-bank nonlinear cost control',fontsize=12)
    fig.text(.5,.07,'Original complete recording-risk views supported: '+risk,ha='center',fontsize=8)
    fig.text(.5,.025,'3,000 locality resamples / 6 exposed groups. Repeated pipeline views are not independent. '
             'Available-label ADE is not full12 ADE.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.98,bottom=.26,top=.8,wspace=.47)
    folder = PUBLIC/'figures'; preview = PRIVATE/'figures'
    folder.mkdir(exist_ok=True); preview.mkdir(exist_ok=True)
    svg = folder/'forest_cost_evidence.svg'; png = preview/'forest_cost_evidence.png'
    fig.savefig(svg,metadata={'Date':None}); fig.savefig(png,dpi=170); plt.close(fig)
    (folder/'figure_receipt.json').write_text(json.dumps(dict(result_source='fresh_plot_verified_public_scalars',
        fit_receipt_hashes=hashes,summary_sha256=sha(PUBLIC/'summary.json'),script_sha256=sha(__file__),
        svg_sha256=sha(svg),private_preview_sha256=sha(png),private_arrays_or_weights_read=False,
        training_or_policy_change=False,independent_confirmation=False),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(svg=str(svg),preview=str(png))))


if __name__ == '__main__': main()
