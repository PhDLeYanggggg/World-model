"""Render a private aggregate figure; no row data or image is committed."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts import run_m3w_policy_conditional_calibration as run


def main():
    s=json.loads((run.PUBLIC/'summary.json').read_text())
    out=run.PRIVATE/'figures/comparison.png';out.parent.mkdir(parents=True,exist_ok=True)
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
    keys=['raw','global','support_only'];names=['Raw rational','Global correction','Support-only veto']
    values=[s['contrasts'][k+'_matched'] for k in keys]
    mean=np.array([v['mean'] for v in values]);ci=np.array([v['CI95'] for v in values])
    a.errorbar(mean,np.arange(3),xerr=np.vstack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=5,color='#af3b46')
    a.axvline(0,color='#333333',linewidth=1);a.set_yticks(np.arange(3),names);a.invert_yaxis()
    a.set_xlabel('Conditional minus control utility (pp)\nMatched counts; nominal 95% locality interval')
    a.set_title('Conditional correction loses useful selection',fontsize=12)
    arms=['raw','global','support_only','conditional'];x=np.arange(4)
    bars=b.bar(x,[s['safety'][k]['full_support'] for k in arms],color=['#667983','#378478','#b48c38','#4d6d9f'])
    b.bar_label(bars,padding=3);b.set_ylim(0,76);b.set_xticks(x,['Raw','Global','Support\nonly','Conditional'])
    b.set_ylabel('Fully supported views out of 72')
    b.set_title('20 supported views still fail the full gate',fontsize=12)
    b.axhline(72,linestyle='--',color='#333333',linewidth=1)
    for ax in (a,b):
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exposed TRAIN-only mechanism study, not deployment evidence',fontsize=13)
    fig.savefig(out,dpi=180);plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json',dict(path=str(out.relative_to(run.ROOT)),sha256=run.sha(out),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'),aggregate_only=True,private_image=True))
    print(out)


if __name__=='__main__':main()
