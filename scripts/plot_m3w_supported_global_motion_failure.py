"""Posthoc worst-easy recording case; raw paths/figures remain private."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_supported_global_motion as run
np=run.np


def main():
    a=run.read(run.PUBLIC/'analysis.json');reg=run.read(run.REG);cases=[]
    for row in a['rows']:
        if row['arm']!='augmented_global':continue
        for record,v in row['policies']['joint_free']['recordings'].items():
            if v['known_easy_net_degradation'] is not None:
                cases.append((v['known_easy_net_degradation'],row['source'],row['seed'],record))
    worst,source,seed,record=max(cases)
    node=next(n for n in reg['jobs'] if source in n['part']['names']);own,old=run.parent.own_and_floor(node)
    data=run.parent.full.load_data();ix=old['local_ids'];target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
    def bank(arm):return run.parent.load_cache(run.read(run.PUBLIC/'inference'/f'{node["job"]["key"]}_{arm}_seed{seed}.json')['cache'])
    g,l=bank('augmented_global'),bank('augmented_local')
    prediction=run.parent.full.method.paths_for(old['paths'],old['floor'],g['joint_free'])
    local=run.parent.full.method.paths_for(old['paths'],old['floor'],l['joint_free'])
    error=run.parent.full.method.original.errors
    pe,_=error(prediction,target,valid);be,_=error(old['floor'],target,valid);ce,_=error(old['cv'],target,valid)
    take=(old['sites']==source)&(old['recordings'].astype(str)==record)&valid.any(1)&(ce>0)&(ce<=node['job']['easy_cut'])&(g['joint_free']>=0)
    candidates=np.flatnonzero(take)
    if not len(candidates):raise ValueError('Worst easy record must contain a selected known-easy case')
    at=candidates[np.argmax(pe[candidates]-be[candidates])];action=int(g['joint_free'][at])
    if pe[at]<=be[at]:raise ValueError('Positive realized harm case required')
    local_error,_=error(local[at:at+1],target[at:at+1],valid[at:at+1])
    moment=g['moments'][at,action]
    case=dict(source='fresh_run_posthoc_case_from_cached_verified_readout',source_component=source,
        risk_seed=seed,forecaster_seed=17,recording=record,current_frame=int(old['frames'][at]),
        agent_window_id=int(old['ids'][at]),action=action,alpha=run.controls.method.FRACTIONS[action-1],
        future_labels=int(valid[at].sum()),prediction_horizon=12,known_easy=True,
        floor_available_label_ADE_pixel=float(be[at]),global_available_label_ADE_pixel=float(pe[at]),
        local_available_label_ADE_pixel=float(local_error[0]),realized_positive_harm_pixel=float(pe[at]-be[at]),
        predicted_moments=dict(zip(('B','H','R','ER','EH'),map(float,moment))),
        record_signed_easy_degradation=worst,
        case_selection='worst_signed_easy_record_then_largest_positive_easy_harm_case_not_representative_sampling',
        joint_point_not_complete_query_budget=True,future_is_label_only=True,independent_confirmation=False,
        original_rules_unchanged=True,deployment_changed=False)
    run.once(run.PUBLIC/'failure_case.json',case)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=data['history'][ix[at]]-data['origin'][ix[at],None]
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    ax.plot(history[:,0],history[:,1],'o-',color='#525252',label='Observed 8 states',markersize=3)
    ax.plot(old['floor'][at,:,0],old['floor'][at,:,1],'--',color='#2563eb',label=f'CV floor ADE {be[at]:.3f}')
    ax.plot(local[at,:,0],local[at,:,1],':',color='#7c3aed',label=f'Local policy ADE {local_error[0]:.3f}')
    ax.plot(prediction[at,:,0],prediction[at,:,1],'-',color='#dc2626',label=f'Global policy ADE {pe[at]:.3f}')
    observed=np.flatnonzero(valid[at]);parts=np.split(observed,np.flatnonzero(np.diff(observed)>1)+1)
    for j,part in enumerate(parts):
        ax.plot(target[at,part,0],target[at,part,1],'o-',color='#15803d',label='Available future labels' if j==0 else None)
    ax.scatter([0],[0],marker='s',color='#111827',label='Current origin',zorder=8)
    ax.set(xlabel='Relative image x (pixels)',ylabel='Relative image y (pixels)',
        title=f'Posthoc failure: {source.removeprefix("moving_source_component_")}, recording {record}, risk seed {seed}\n'
            f'alpha={case["alpha"]:g}; {len(observed)}/12 future labels; no metric/time calibration')
    ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=8)
    path=run.PRIVATE/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/'worst_easy_case.png',dpi=170);fig.savefig(path/'worst_easy_case.pdf');plt.close(fig)
    print('Posthoc actual failure case saved; raw paths and figures stay private.')


if __name__=='__main__':main()
