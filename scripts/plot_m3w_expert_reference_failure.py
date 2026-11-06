"""Posthoc worst-record illustration; labels never enter inference or tuning."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_expert_reference as run


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=run.read(run.PUBLIC/'analysis.json');_,reg=run.register()
    views=[(v['known_easy_net_degradation'],r,rec) for r in a['rows'] if r['arm']=='expert_context'
           for rec,v in r['policies']['joint_free']['recordings'].items() if v['known_easy_net_degradation'] is not None]
    worst,row,recording=max(views,key=lambda v:v[0]);node=next(n for n in reg['jobs'] if n['job']['key']==row['job'])
    _,old=run.parent.own_and_floor(node);data=run.parent.full.load_data();ix=old['local_ids']
    target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
    bank=run.parent.load_cache(run.read(run.PUBLIC/'inference'/f'{row["job"]}_expert_context_seed{row["seed"]}.json')['cache'])
    choice=bank['joint_free'];packet=dict(paths=old['paths'],floor=old['floor'],cv=old['cv'],
        causal=dict(envelope=run.np.linalg.norm(old['paths']-old['floor'][:,None],axis=-1).max(-1)))
    truth=run.parent.previous.method.amplitude.targets(packet,target,valid,node['job']['easy_cut'])['moments']
    at=run.np.flatnonzero((old['recordings'].astype(str)==recording)&(choice>=0)&valid.any(1))
    at=at[(truth[at,choice[at],3]>0)&(truth[at,choice[at],1]>0)]
    if not len(at):raise ValueError('Worst signed easy record must supply a real selected easy-harm case')
    selected=int(at[run.np.argmax(truth[at,choice[at],1])]);action=int(choice[selected]);global_row=int(ix[selected])
    history=data['history'][global_row]-data['origin'][global_row]
    actual=target[selected];floor=old['floor'][selected];prediction=old['paths'][selected,action]
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    ax.plot(history[:,0],history[:,1],'o-',color='#525252',markersize=3,label='Observed history (8)')
    ax.plot(floor[:,0],floor[:,1],'-',color='#2563eb',label='Causal CV floor')
    ax.plot(old['paths'][selected,6,:,0],old['paths'][selected,6,:,1],'--',color='#7c3aed',label='Frozen full expert')
    ax.plot(prediction[:,0],prediction[:,1],'o-',color='#15803d',markersize=3,label='Frozen chosen amplitude')
    gt=actual.copy();gt[~valid[selected]]=run.np.nan
    ax.plot(gt[:,0],gt[:,1],'s-',color='#dc2626',markersize=4,label='Available future labels (evaluation only)')
    ax.scatter([0],[0],c='black',marker='x',s=35,label='Current point')
    ax.set(xlabel='Relative image x (pixel)',ylabel='Relative image y (pixel)',
        title='Posthoc easy-harm case, native obs8/pred12\nNot a representative sample or a physical safety claim')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=8)
    out=run.PRIVATE/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'worst_record_easy_case.png',dpi=170);fig.savefig(out/'worst_record_easy_case.pdf');plt.close(fig)
    value=truth[selected,action]
    run.once(run.PUBLIC/'failure_case_summary.json',run.stable(dict(source=row['source'],recording=recording,seed=row['seed'],
        selection='largest_absolute_selected_easy_harm_within_worst_signed_easy_record',
        worst_record_signed_easy_degradation=worst,available_future_labels=int(valid[selected].sum()),
        selected_action=action,amplitude=run.parent.donor.FRACTIONS[action-1],floor_ADE=float(value[2]),
        selected_ADE=float(value[2]+value[1]-value[0]),positive_harm=float(value[1]),
        predicted_moments=bank['moments'][selected,action].tolist(),
        result_source='fresh_run_posthoc_illustration_only',used_for_inference_or_tuning=False,
        figure_and_raw_trajectories_private=True)))
    print('Actual posthoc easy-harm illustration saved privately; scalar summary public.')


if __name__=='__main__':main()
