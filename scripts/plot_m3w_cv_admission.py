"""Plot matched-risk readout and separate hindsight capacity."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    r=json.loads((a.report/'readout.json').read_text())
    d=json.loads((a.report/'unpruned_capacity.json').read_text())
    if r['a_labels_opened'] or d['views'] != r['summary']['views']:
        raise ValueError('Fitting-only roster required')
    risk=r['summary']['risk']
    keys=('relaxed_gain_match','relaxed','relaxed_joint')
    if len({risk[k]['switches'] for k in keys}) != 1:
        raise ValueError('Expected matched counts')
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained')
    panels=(('Causal diagnostic, 2,317 edits each',('Gain only','Risk screen','Coupled screen'),
             [risk[k]['supported_views'] for k in keys], '#346389'),
            ('Hindsight, both references and risk',('Forced nominees','Unpruned nominees','OLS4 retained'),
             [d['status'][k]['verified'] for k in ('forced_pool','unpruned','ols4_addback')],'#9b4d34'))
    for ax,(title,labels,values,color) in zip(axes,panels,strict=True):
        bars=ax.barh(labels,values,color=color,height=.52)
        ax.bar_label(bars,padding=4)
        ax.invert_yaxis()
        ax.set(xlim=(0,72),xticks=[0,18,36,54,72],xlabel='Verified recording-seed views (of 72)',title=title)
        ax.grid(axis='x',alpha=.17)
        ax.spines[['right','top']].set_visible(False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=200)


if __name__=='__main__':
    main()
