"""Render the support recovery and count-matched utility contrasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def draw(r,m,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,(a,b) = plt.subplots(2,1,figsize=(3.25,3.55),layout='constrained')
    names = ('Frozen','Direct','Conditional')
    keys = ('old','direct','conditional')
    donors = r['donors']
    n = [donors[k]['same_action_overlap'] for k in keys]
    a.bar(names,n,color=['#81909a','#355b70','#1d706d'],width=.56)
    a.axhline(donors['old']['actual'],ls='--',lw=.8,color='#555555')
    a.set_ylim(0,2700)
    a.set_ylabel('True seed queries')
    a.set_title('A. Correctly recovered seeds',loc='left')
    for i,v in enumerate(n):
        a.text(i,v+80,f'{v:,}',ha='center',fontsize=8)
    for tag,off,color,metric in (('Known labels',.12,'#236b87','known_pp'),
                                 ('Full-grid lower',-.12,'#bb542e','completion_lower_pp')):
        for j,k in enumerate(('direct','conditional')):
            s = m['summary']['effects'][k][metric]
            v,(l,h) = s['point'],s['interval95']
            b.errorbar(v,j+off,xerr=[[v-l],[h-v]],fmt='o',ms=4,capthick=.8,capsize=2,
                       color=color,label=tag if j==0 else None)
    b.set_yticks([0,1],['Direct','Conditional'])
    b.axvline(0,lw=.8,color='#555555')
    b.set_ylim(-.45,1.45)
    b.set_xlabel('Swap minus frozen gain (pp)')
    b.set_title('B. Matched-count gain',loc='left')
    handles,labels = b.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False,fontsize=7)
    fig.savefig(out,metadata={'CreationDate':None,'ModDate':None,'Title':'Signed coverage ablation'})
    plt.close(fig)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('out',type=Path)
    a = p.parse_args()
    draw(*(json.loads((a.results/f'{k}.json').read_text()) for k in ('readout','matched_gain')),a.out)
