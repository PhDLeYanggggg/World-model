"""Fixed-bank statistical figure from the verified SOURCE estimator control."""
import json
import hashlib
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_multiaction_forest_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_multiaction_forest_v1'


def intervals(axis, x, values, color, label, marker='o'):
    for i, value in enumerate(values):
        if value['point'] is None or value['CI95'] is None:
            axis.text(x[i], .02, 'not estimable', rotation=90, va='bottom', transform=axis.get_xaxis_transform())
            continue
        point, (low, high) = value['point'], value['CI95']
        axis.vlines(x[i], low, high, color=color, linewidth=2)
        axis.hlines([low,high], x[i]-.035, x[i]+.035, color=color, linewidth=1.5)
        axis.plot(x[i], point, marker=marker, color=color, label=label if i == 0 else None, markersize=6)


def main():
    evidence = json.loads((PUBLIC/'evidence.json').read_text())
    if not evidence['gates']['exact144group_replay']:raise ValueError('Verified evidence required')
    held = evidence['held']; arms = ('physical_only','physical_neural','neural_only')
    names = ('Physical (6)','Mixed (7)','Neural (1)'); x = np.arange(3)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'svg.fonttype':'none','svg.hashsalt':'m3w_multiaction_forest_v1'})
    fig, axes = plt.subplots(1,3,figsize=(14,5.5))
    intervals(axes[0], x-.09, [held['accuracy'][a] for a in arms], '#287D63','ExtraTrees cost estimator')
    intervals(axes[0], x+.09, [held['accuracy'][a+'_neural_parent'] for a in arms], '#93546E','Frozen neural cost head','s')
    axes[0].set_title('Known-outcome accuracy'); axes[0].set_ylabel('ADE gain over floor (%)')
    intervals(axes[1],x-.09,[held['forest_minus_neural'][a]['full_known_pp'] for a in arms], '#287D63','Known paired difference')
    intervals(axes[1],x+.09,[held['forest_minus_neural'][a]['full_lower_pp'] for a in arms], '#93546E','Completion-lower contrast','s')
    axes[1].set_title('Forest minus neural estimator'); axes[1].set_ylabel('Known-floor-normalized utility (pp)')
    for offset,suffix,color,label in ((-.16,'','#287D63','ExtraTrees'),(.16,'_neural_parent','#93546E','Neural head')):
        rows=[held['safety'][a+suffix] for a in arms]
        bars=axes[2].bar(x+offset,[r['budget_record_views'] for r in rows],width=.28,color=color,label=label)
        for bar,row in zip(bars,rows):
            axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,
                str(row['budget_record_views'])+'/'+str(row['required_record_views']),ha='center',fontsize=8)
    maximum=max(held['safety'][a]['required_record_views'] for a in arms)
    axes[2].set_ylim(0,maximum*1.12);axes[2].axhline(maximum,color='#555555',linestyle=':',linewidth=1)
    axes[2].set_title('Full all/easy 2% risk support');axes[2].set_ylabel('Supported repeated recording views')
    for axis in axes:
        axis.set_xticks(x,names);axis.spines[['top','right']].set_visible(False)
        axis.set_xlabel('Forecast candidate bank')
        axis.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.23))
    for axis in axes[:2]:axis.axhline(0,color='#777777',linewidth=.8,linestyle='--')
    fig.suptitle('A matched conventional estimator does not change the evidence standard',fontsize=15,y=.98)
    fig.text(.5,.025,'Exposed SOURCE development; 3,000 locality resamples, not independent confirmation. '
        'Banks share frozen forecasts; estimator objectives and compute differ.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.8,bottom=.28,left=.065,right=.985,wspace=.34)
    (PUBLIC/'figures').mkdir(exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    fig.savefig(PUBLIC/'figures/estimator_comparison.svg',metadata={'Date':None})
    fig.savefig(PRIVATE/'estimator_comparison_preview.png',dpi=160)
    plt.close(fig)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (PUBLIC/'figure_manifest.json').write_text(json.dumps(dict(
        source_evidence_sha256=digest(PUBLIC/'evidence.json'),generator_sha256=digest(Path(__file__)),
        svg_sha256=digest(PUBLIC/'figures/estimator_comparison.svg'),
        kind='generated_statistical_figure_not_third_party_media',raw_data_or_images_included=False,
        preview_png_private=True),indent=2)+'\n')
    print(str(PRIVATE/'estimator_comparison_preview.png'))


if __name__ == '__main__':main()
