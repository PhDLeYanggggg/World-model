"""Plot all predeclared pointwise candidate contrasts, without selecting a seed."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_protected_motion_v1'


def main():
    raw=(PUBLIC/'analysis.json').read_bytes()
    r=json.loads(raw)
    audit=json.loads((PUBLIC/'accounting_audit.json').read_text())
    if not audit['all_passed'] or audit['analysis_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Complete verified accounting required')
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for i,event in enumerate(('all','easy')):
        for j,guard in enumerate(('no_guard','source_zero_guard')):
            ax=axes[i,j]
            labels=[]
            for y,(arm,seed) in enumerate((a,s) for a in ('ridge','neural_underharm4') for s in (17,29,43)):
                m=r['neural_vs_damping'][f'{seed}_{event}_{arm}_{guard}']['pointwise']
                labels.append(('Ridge' if arm=='ridge' else 'Neural risk')+f' / seed {seed}')
                v,ci=m['equal_scene_gain_percent'],m['scene_bootstrap_ci95']
                if v is None or ci is None:ax.text(0,y,'undefined',ha='left');continue
                ax.errorbar(v,y,xerr=np.array([[v-ci[0]],[ci[1]-v]]),fmt='o' if arm=='ridge' else 's',
                    color='#24758b' if arm=='ridge' else '#a75339',capsize=3,markersize=5)
            ax.axvline(0,color='#555555',lw=1)
            ax.set_yticks(range(len(labels)),labels)
            ax.invert_yaxis()
            ax.grid(axis='x',alpha=.18)
            ax.set_title(('All-event' if event=='all' else 'Easy-event')+' / '+('no support guard' if j==0 else 'source-support guard'))
            ax.set_xlabel('Neural trajectory gain over protected damping (%)')
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same predicted-risk budget, candidate-specific matched training\n'
        '3,000-locality bootstrap; source-development evidence, not independent safety certification',fontsize=14)
    fig.savefig(PUBLIC/'candidate_contrasts.png',dpi=160)
    fig.savefig(PUBLIC/'candidate_contrasts.svg')
    path=PUBLIC/'candidate_contrasts.svg'
    path.write_text(re.sub(r'[ \t]+\n','\n',path.read_text()))
    print('Rendered candidate_contrasts PNG/SVG; visual inspection still required.')


if __name__=='__main__':main()
