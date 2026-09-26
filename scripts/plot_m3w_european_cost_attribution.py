"""Signed algebraic components, not independent causal contributions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_cost_attribution_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_cost_attribution_v1'


def main():
    d=json.loads((PUBLIC/'aggregate_metrics.json').read_text())['contrasts']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'cost-attribution-v1'}):
        fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True)
        for ax,pair in zip(axes,('full','motion_only')):
            rs=d[pair]['disagreement']; labels=sorted(rs); y=np.arange(len(labels))
            for shift,key,label,color in ((-.24,'membership_over_MSE','Membership squared','#186d89'),
                (0,'severity_over_MSE','Severity squared','#b3712d'),(.24,'cross_over_MSE','Signed cross term','#a94156')):
                ax.barh(y+shift,[rs[r][key]['point'] for r in labels],height=.22,color=color,label=label)
            ax.axvline(0,color='#555',lw=.7); ax.set_title(pair); ax.grid(axis='x',alpha=.2)
            ax.set_xlabel('Term / composed MSE'); ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(y,labels); axes[0].invert_yaxis(); axes[1].legend(frameon=False,fontsize=9)
        fig.suptitle('Offline factor attribution / signed terms sum to 1, not causal shares')
        fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(PUBLIC/'factor_terms.svg',metadata={'Date':None})
        fig.savefig(PRIVATE/'factor_terms_preview.png',dpi=130); plt.close(fig)


if __name__=='__main__': main()
