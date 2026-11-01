"""Render a paper-scale plot from complete public scalar results,not private data."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    from scripts import report_m3w_own_decision_cost as own
    if (own.run.PUBLIC/'verification.json').exists():
        v=own.build();s=v['summary'];names=('frozen_projected_joint','fixed_joint','own_joint')
        labels=('Frozen','Fixed-choice','Own-choice');contrast=s['effects']['own_minus_fixed']['full']
        title='Same forecasts; fixed versus own decisions';xlabel='Own minus fixed utility (pp)';source=own.run.NAME
    else:
        v=own.run.prior_report.build();s=v['summary'];names=('raw_locked_joint','locked_joint','raw_learned_joint','learned_joint')
        labels=('Locked\nraw','Locked\nproj.','Learned\nraw','Learned\nproj.');contrast=s['effects']['learned_projection']['full']
        title='Zero-step projection; no new training';xlabel='Projected minus raw utility (pp)';source=own.run.previous.NAME
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8}):
        fig,axes=plt.subplots(1,2,figsize=(7,2.55),layout='constrained')
        for i,name in enumerate(names):
            x=s['accuracy'][name];axes[0].bar(i,x['point'],width=.6,color=('#326a95','#5b8b70','#9a6a16','#b85042')[i])
            if x['CI95'] is not None:axes[0].vlines(i,*x['CI95'],color='black',linewidth=.8)
        axes[0].set_xticks(range(len(names)),labels);axes[0].set_ylabel('Known ADE gain over floor (%)');axes[0].set_title(title)
        for i,key in enumerate(('known_pp','lower_pp')):
            x=contrast[key]
            if x['CI95'] is not None:axes[1].hlines(i,*x['CI95'],color='#326a95',linewidth=1);axes[1].plot(x['point'],i,'o',color='#326a95',markersize=4)
        axes[1].axvline(0,color='black',linewidth=.6);axes[1].set_yticks([0,1],['Known','Completion\nlower']);axes[1].set_xlabel(xlabel)
        axes[1].set_title('Paired conditional 95% intervals');axes[1].set_ylim(-.4,1.4)
        for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        a.output.mkdir(parents=True,exist_ok=True);fig.savefig(a.output/'cost_evidence.pdf');fig.savefig(a.output/'cost_evidence.png',dpi=240);plt.close(fig)
        from scripts import report_m3w_feature_cost_evidence as historical
        value=historical.build();names=('Continuous cost','Recording policy','Published deferral','Support-unit repair','Shared reference','Static context')
        fig,axes=plt.subplots(1,2,figsize=(7,3.1),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
        y=range(6)
        for i,study in enumerate(historical.STUDIES):
            x=value['inner_held_mixed_minus_physical'][study]['full_known_pp']
            axes[0].hlines(i,*x['CI95'],color='#326a95');axes[0].plot(x['point'],i,'o',color='#326a95',markersize=3)
        axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].axvline(0,color='black',linestyle='--',linewidth=.6)
        axes[0].set_title('Known neural increment');axes[0].set_xlabel('Mixed minus physical gain (pp)')
        for offset,arm,label,color in ((-.16,'physical_only','Physical','#326a95'),(.16,'physical_neural','Mixed','#b85042')):
            rows=[next(r for r in value['policies'] if r['study']==study and r['scope']=='inner_held' and r['policy']==arm) for study in historical.STUDIES]
            if any(r['required_record_head_views']!=186 for r in rows):raise ValueError('Historical complete roster changed')
            counts=[r['supported_record_head_views'] for r in rows]
            axes[1].barh([i+offset for i in y],counts,height=.29,color=color,label=label)
            for i,count in enumerate(counts):axes[1].text(count+2,i+offset,str(count),va='center',fontsize=8)
        axes[1].set_yticks(y,[]);axes[1].invert_yaxis();axes[1].set_xlim(0,205);axes[1].set_xticks([0,62,124,186])
        axes[1].axvline(186,color='black',linestyle='--',linewidth=.6);axes[1].set_xlabel('Supported record/head views / 186');axes[1].set_title('Recording risk support')
        axes[1].legend(loc='upper right',frameon=False,fontsize=8)
        for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        fig.savefig(a.output/'historical_evidence.pdf');fig.savefig(a.output/'historical_evidence.png',dpi=240);plt.close(fig)
    source_dir=own.run.PUBLIC if source==own.run.NAME else own.run.previous.PUBLIC
    receipt=dict(source=source,summary_sha256=own.run.sha(source_dir/'summary.json'),
        historical_summary_sha256={study:own.run.sha(own.run.PUBLIC.parent/study/'summary.json') for study in historical.STUDIES},
        figures={name:own.run.sha(a.output/name) for name in ('cost_evidence.pdf','historical_evidence.pdf')},
        exposed_SOURCE_only=True,independent_confirmation=False)
    (a.output/'plot_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
