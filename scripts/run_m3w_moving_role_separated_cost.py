"""Role-separated cost learning and current-frame joint amplitude decisions."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_complete_risk_donors as previous
from src.world_model import m3w_role_separated_cost as method
from src.world_model import m3w_frame_action_allocation as allocation
from src.data_unification import m3w_full_action_features as basis

np,torch=previous.np,previous.torch
full,donor,source,neural=previous.full,previous.donor,previous.source,previous.neural
read,stable,sha,once,checked,artifact=previous.read,previous.stable,previous.sha,previous.once,previous.checked,previous.artifact
load_cache=previous.load_cache
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_role_separated_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_role_separated_cost_v1'
CONFIG=ROOT/'configs/m3w_moving_role_separated_cost_v1.json'
TRAINING_REG=PUBLIC/'registration_v2.json'
REG=PUBLIC/'registration_v3.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_role_separated_cost.py',
       ROOT/'src/data_unification/m3w_full_action_features.py',ROOT/'tests/test_m3w_role_separated_cost.py',
       ROOT/'tests/test_m3w_full_action_features.py',PUBLIC/'reference_invariance_amendment.md',
       PUBLIC/'verification_only_amendment.md')
POLICIES=('proposal_only',*allocation.ARMS)


def register():
    cfg=read(CONFIG);_,parent=previous.register()
    frozen=read(TRAINING_REG)
    for p in (CONFIG,ROOT/'src/world_model/m3w_role_separated_cost.py',
              ROOT/'src/data_unification/m3w_full_action_features.py',PUBLIC/'protocol.md'):
        if sha(p)!=frozen['files'][str(p.relative_to(ROOT))]:
            raise ValueError('Verification amendment cannot change model/data/training definitions')
    if cfg!=frozen['config'] or method.SETTINGS!=frozen['settings']:
        raise ValueError('Frozen fitting settings cannot change')
    if (cfg['arms']!=list(method.ARMS) or cfg['risk_head_seeds']!=[17,29,43]
            or cfg['fresh_models']!=48 or cfg['steps_per_model']!=4096 or cfg['risk_budget']!=.02
            or cfg['policies']!=list(POLICIES) or cfg['primary_policy']!='joint_free'
            or cfg['expected_evaluation_rows']!=62466 or cfg['augmentation_mass']!=.5
            or any(cfg[k] for k in ('threshold_search','new_outer_C_H_readout','identity_certification',
                'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed matched loss comparison, all views and original rules required')
    value=dict(config=cfg,settings=method.SETTINGS,jobs=parent['jobs'],definition_version=3,
        training_registration=artifact(TRAINING_REG),training_model_and_settings_unchanged=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration=artifact(previous.REG),parent_training=artifact(previous.PUBLIC/'training_freeze.json'),
        full_population=artifact(full.PUBLIC/'population.json'),full_inference=artifact(full.PUBLIC/'inference_freeze_v2.json'),
        previous_result=artifact(previous.PUBLIC/'analysis.json'),
        earlier_negative_observation_result=artifact(ROOT/'outputs/publication_readiness_2026_09/european_observation_cost_v1/readout.md'),
        parent_code={str(p.relative_to(ROOT)):sha(p) for p in (Path(allocation.__file__),
            Path(allocation.select_actions.__globals__['__file__']),Path(full.method.original.__file__),
            Path(previous.method.amplitude.__file__),Path(basis.packet.__file__),Path(method.storage.__file__))},
        role='exposed_own_B_full_candidate_source_held_comparison',original_rules_unchanged=True,
        independent_confirmation=False,new_outer_C_H_readout=False)
    once(REG,stable(value));return cfg,value


def committed():
    for p in (*FILES,REG):
        body=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(body).hexdigest()!=sha(p):raise ValueError('Commit exact study before real preparation/fitting')


def runtime(cfg,phase):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume saved model/allocation')))
    start,last=time.monotonic(),[0.]
    def guard():
        now=time.monotonic()
        if now-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('Resume same study after12h')
        if now-last[0]>1:
            last[0]=now
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep5GiB reserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>cfg['artifact_cap_bytes']:
                raise OSError('New private artifact cap exceeded')
    def beat(**values):
        guard();event=dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
        source.atomic(PRIVATE/'heartbeat.json',event)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    beat(state='ready',machine=platform.machine(),device='cpu',torch_threads=4,interop_threads=1,workers=0)
    return start,guard,beat


def normalized_moments(paths,floor,cv,target,valid,easy_cut,scales):
    radius=np.linalg.norm(paths-floor[:,None],axis=-1).max(-1)
    packet=dict(paths=paths,floor=floor,cv=cv,causal=dict(envelope=radius))
    truth=previous.method.amplitude.targets(packet,target,valid,easy_cut)
    return (truth['moments']/scales[:,None,None]).astype(np.float32),truth['observed_future'],radius


def prepare_own(reg,guard,beat):
    data=full.load_data();refs=[]
    for node in reg['jobs']:
        job=node['job'];key=job['key'];old=load_cache(read(full.PUBLIC/'inference_v2'/f'{key}.json')['cache'])
        ix=old['local_ids'];guard()
        packet=basis.packet.packet(geometry=data['geometry'][ix],shared_context=data['shared_context'][ix],
            history=data['history'][ix],recordings=old['recordings'],frames=old['frames'],ids=old['ids'],
            sites=old['sites'],relative_forecast=old['paths'][:,6],floor_index=job['floor_index'])
        x=basis.summary(packet['causal']);scales=packet['causal']['scales']
        np.testing.assert_array_equal(packet['paths'],old['paths'])
        active=old['active'];np.testing.assert_array_equal(x[active,old['proposal_self'][active]],old['summary120'])
        y,known,radius=normalized_moments(packet['paths'],packet['floor'],packet['cv'],
            data['target_eval'][ix]-data['origin'][ix,None],data['valid'][ix],job['easy_cut'],scales)
        row,action=np.nonzero(packet['causal']['allowed'])
        own=dict(x=x[row,action],moments=y[row,action],observed=known[row],
            envelope=(radius[row,action]/scales[row]).astype(np.float32),scales=scales[row],
            sites=old['sites'][row],query=data['queries'][ix][row],ids=old['ids'][row]*7+action,
            row=row,action=action)
        own['weights']=method.storage.base.balanced_weights(own['sites'],own['query'])
        path=PRIVATE/'own'/f'{key}.npz';full.save_arrays(path,own)
        refs.append(dict(job=key,cache=artifact(path),agent_rows=len(ix),candidate_rows=len(row),
            known_candidates=int(own['observed'].sum()),unknown_candidates=int((~own['observed']).sum()),
            original_proposal_features_exact=True,first50_action_invariant=True))
        beat(state='all_action_own_prepared',job=key,rows=len(ix),candidates=len(row))
    once(PUBLIC/'own.json',stable(dict(banks=refs,features_before_future_labels=True)))


def added_data(fold):
    ref=next(r for r in read(previous.PUBLIC/'unknown_A.json')['banks'] if r['producer_fold']==fold)
    added=previous.method.append_unknown(donor.load_added(fold),load_cache(ref['cache']))
    side=next(r for r in read(PUBLIC/'A_moments.json')['banks'] if r['producer_fold']==fold)
    labels=load_cache(side['cache']);order=np.argsort(labels['ids']);positions=order[np.searchsorted(labels['ids'][order],added['ids'])]
    np.testing.assert_array_equal(labels['ids'][positions],added['ids'])
    for k in ('moments','envelope','observed'):added[k]=labels[k][positions]
    signed=torch.from_numpy(added['moments'])
    np.testing.assert_array_equal((signed[:,[1,4]]-.02*signed[:,[2,3]]).numpy(),added['target'])
    return added


def prepare_A(reg,guard,beat):
    data=source.load_data();forecasts=read(source.PUBLIC/'analysis.json')['forecasts'];refs=[]
    for fold in (0,1):
        job=next(n['job'] for n in reg['jobs'] if n['job']['producer_fold']==fold)
        chunks={k:[] for k in ('moments','envelope','observed','ids')}
        for item in forecasts:
            if item['job']['producer_fold']!=fold:continue
            cached=load_cache(item['forecast']);all_ids=cached['local_ids']
            np.testing.assert_array_equal(data['global_ids'][all_ids],cached['global_ids'])
            if not set(data['sites'][all_ids])<=set(job['producer_sites']):raise ValueError('Wrong parent-A ancestry')
            for begin in range(0,len(all_ids),1024):
                guard();ix=all_ids[begin:begin+1024];neural_path=cached['relative_prediction'][begin:begin+1024].astype(float)
                physical=np.stack([donor.baseline_numpy(data['history'][ix],k)-data['origin'][ix,None] for k in range(6)],1)
                floor=physical[:,job['floor_index']]
                paths=np.stack([floor,*[floor+alpha*(neural_path-floor) if alpha!=1 else neural_path
                                       for alpha in donor.FRACTIONS]],1)
                scales=previous.causal_cost_features(data['geometry'][ix],floor,floor)[1].astype(np.float32)
                y,known,radius=normalized_moments(paths,floor,physical[:,1],
                    data['target_eval'][ix]-data['origin'][ix,None],data['valid'][ix],job['easy_cut'],scales)
                legal=radius>0;legal[:,0]=False;row,action=np.nonzero(legal)
                for k,v in dict(moments=y[row,action],envelope=(radius[row,action]/scales[row]).astype(np.float32),
                    observed=known[row],ids=data['global_ids'][ix][row]*7+action).items():chunks[k].append(v)
            beat(state='A_forecast_moments_prepared',forecast=item['job']['key'])
        result={k:np.concatenate(v) for k,v in chunks.items()}
        path=PRIVATE/'A_moments'/f'A{fold}.npz';full.save_arrays(path,result)
        refs.append(dict(producer_fold=fold,cache=artifact(path),candidates=len(result['ids']),
                         unknown=int((~result['observed']).sum())))
    once(PUBLIC/'A_moments.json',stable(dict(banks=refs)))
    for fold in (0,1):
        added=added_data(fold)
        beat(state='all_A_signed_targets_exact',producer_fold=fold,candidates=len(added['ids']))


def mixed(own,added,held):
    if set(own['sites'])&set(added['sites']) or held in set(added['sites']):raise ValueError('Disjoint parent pools required')
    fit=own['sites']!=held
    if not fit.any() or fit.all():raise ValueError('Whole query source excluded')
    ow=np.zeros(len(fit));ow[fit]=.5*own['weights'][fit]/own['weights'][fit].sum()
    aw=.5*added['weights']/added['weights'].sum()
    d={k:np.concatenate((own[k],added[k])) for k in ('x','moments','observed','envelope','sites','ids')}
    d['weights']=np.r_[ow,aw]
    if len(np.unique(d['ids']))!=len(d['ids']):raise ValueError('Unique original row/action namespace required')
    return d


def entries(reg):
    for node in reg['jobs']:
        key=node['job']['key'];ref=next(r for r in read(PUBLIC/'own.json')['banks'] if r['job']==key)
        own=load_cache(ref['cache']);added=added_data(node['job']['producer_fold'])
        for site in node['part']['names']:
            data=mixed(own,added,site)
            for seed in (17,29,43):
                for arm in method.ARMS:
                    identity=dict(registration_sha256=sha(TRAINING_REG),job=key,source=site,risk_seed=seed,arm=arm)
                    yield node,site,seed,arm,data,identity


def fit_path(key,site,seed,arm,kind='fits'):
    return PRIVATE/kind/f'{key}_{site}_seed{seed}_{arm}.pt.gz'


def pilot(reg,guard,beat,resume=False):
    node,site,seed,_,data,identity=next(entries(reg));identity=dict(identity,arm='role_separated',pilot=True)
    p=PRIVATE/'pilot/resumed.pt.gz';q=PRIVATE/'pilot/direct.pt.gz'
    if not p.exists():method.fit(data,site,identity,p,arm='role_separated',seed=seed,stop_at=128,heartbeat=beat,guard=guard)
    a=method.fit(data,site,identity,p,arm='role_separated',seed=seed,stop_at=256,resume=True,heartbeat=beat,guard=guard)
    b=method.fit(data,site,identity,q,arm='role_separated',seed=seed,stop_at=256,resume=resume,heartbeat=beat,guard=guard)
    method.exact(a,b)
    once(PUBLIC/'pilot.json',stable(dict(real128plus128_vs256_exact=True,pilot_updates=512,
        fit_rows=a['fit_rows'],parameters=a['parameters'],direct256_seconds=b['seconds'],
        projected48_primary_fit_seconds=b['seconds']*16*48,projection_excludes_preparation_refits_and_allocation=True)))


def train(reg,guard,beat,*,resume=False,verify=False):
    if not read(PUBLIC/'pilot.json')['real128plus128_vs256_exact']:raise ValueError('Real recovery/resource pilot first')
    refs=[];count=0;matched={}
    for node,site,seed,arm,data,identity in entries(reg):
        key=node['job']['key'];path=fit_path(key,site,seed,arm,'verify' if verify else 'fits')
        state=method.fit(data,site,identity,path,arm=arm,seed=seed,resume=resume,guard=guard,
            heartbeat=lambda **kw:beat(job=key,source=site,seed=seed,**kw))
        if not state['complete'] or state['step']!=4096 or site in state['preprocess']['train_sources']:
            raise ValueError('Complete source-excluded fixed budget required')
        if verify:method.exact(neural.rel.base.read_checkpoint(fit_path(key,site,seed,arm)),state)
        else:
            out=PUBLIC/'fits'/f'{key}_{site}_seed{seed}_{arm}.json'
            once(out,stable(dict(job=key,source=site,seed=seed,arm=arm,checkpoint=artifact(path),
                steps=state['step'],fit_rows=state['fit_rows'],parameters=state['parameters'],
                seconds=state['seconds'],trace=state['trace'],fitting_sources=state['preprocess']['train_sources'],result_source='fresh_run')))
            refs.append(artifact(out))
        pair=key,site,seed
        if arm=='composed_anchor':matched[pair]=state
        else:
            control=matched.pop(pair)
            for field in ('preprocess','initial_model','generator','visits','parameters','step'):
                neural.rel.base.exact(control[field],state[field])
        count+=1;beat(state='full_refit_exact' if verify else 'fit_complete',complete=count,total=48,job=key,source=site,seed=seed,arm=arm)
    if verify:once(PUBLIC/'verification.json',dict(all48_full_refits_exact=True,matched_initialization_and_draws=True,repeat_updates=196608))
    else:once(PUBLIC/'training_freeze.json',dict(fits=refs,models=48,primary_updates=196608,all_before_readout=True,final_checkpoint_only=True))


def own_and_floor(node):
    key=node['job']['key'];own=load_cache(next(r['cache'] for r in read(PUBLIC/'own.json')['banks'] if r['job']==key))
    old=load_cache(read(full.PUBLIC/'inference_v2'/f'{key}.json')['cache'])
    return own,old


def predict_one(node,arm,seed):
    own,old=own_and_floor(node);n=len(old['ids']);p=np.zeros((n,7,5));conditional=np.zeros_like(p);prob=np.zeros((n,7))
    for site in node['part']['names']:
        at=np.flatnonzero(own['sites']==site)
        state=neural.rel.base.read_checkpoint(checked(read(PUBLIC/'fits'/f'{node["job"]["key"]}_{site}_seed{seed}_{arm}.json')['checkpoint']))
        if site in state['preprocess']['train_sources']:raise ValueError('Query source entered preprocessing')
        output=method.predict(state,own['x'][at],own['envelope'][at]);row,action=own['row'][at],own['action'][at]
        p[row,action]=output['moments']*own['scales'][at,None]
        conditional[row,action]=output['conditional']*own['scales'][at,None]
        prob[row,action]=output['observation_probability']
    return dict(moments=p,conditional=conditional,probability=prob),own,old


def infer(reg,cfg,beat,*,resume=False,replay=False):
    if not read(PUBLIC/'verification.json')['all48_full_refits_exact']:raise ValueError('All fits/refits before inference')
    refs=[]
    for node in reg['jobs']:
        for seed in (17,29,43):
            for arm in method.ARMS:
                key=f'{node["job"]["key"]}_{arm}_seed{seed}';out=PUBLIC/'inference'/f'{key}.json'
                if resume and out.exists() and not replay:checked(read(out)['cache']);refs.append(artifact(out));continue
                prediction,own,old=predict_one(node,arm,seed);p=prediction['moments']
                allowed=np.zeros(p.shape[:2],bool);allowed[own['row'],own['action']]=True
                proposal=np.zeros_like(allowed);at=old['active'];proposal[at,old['proposal_self'][at]]=True
                proposal_only,_=allocation.select_actions(p,proposal,arm='physical_neural')
                path=PRIVATE/'inference'/f'{key}.npz'
                if replay:
                    cached=load_cache(read(out)['cache'])
                    for k,v in prediction.items():np.testing.assert_array_equal(v,cached[k])
                    np.testing.assert_array_equal(proposal_only,cached['proposal_only'])
                    independent,_=allocation.select_actions(p,allowed,arm='physical_neural')
                    np.testing.assert_array_equal(independent,cached['independent'])
                    for group in allocation.query_groups(old['recordings'],old['frames'],old['ids']):
                        for name in allocation.ARMS:
                            if not allocation.feasible(p[group],allowed[group],cached[name][group]):raise ValueError('Frozen solver choice violates predicted budgets')
                        if (cached['joint_matched'][group]>=0).sum()!=(independent[group]>=0).sum():raise ValueError('Matched count changed')
                else:
                    checkpoint=PRIVATE/'allocation'/f'{key}.pt.gz'
                    restored=neural.rel.base.read_checkpoint(checkpoint) if resume and checkpoint.exists() else None
                    choices,_,info=allocation.allocate(p,allowed,old['recordings'],old['frames'],old['ids'],
                        seconds=cfg['solver_seconds_per_query'],heartbeat=lambda **kw:beat(job=key,state='joint_allocation',**kw),
                        checkpoint=lambda state:neural.rel.base.save_checkpoint(checkpoint,state),resume_state=restored)
                    full.save_arrays(path,dict(**prediction,proposal_only=proposal_only,**choices))
                    once(out,stable(dict(job=node['job']['key'],seed=seed,arm=arm,cache=artifact(path),allocation=info,
                                         all_choices_before_labels=True,result_source='fresh_run')))
                refs.append(artifact(out));beat(state='inference_replayed' if replay else 'causal_choices_frozen',job=key)
    if not replay:once(PUBLIC/'inference_freeze.json',dict(contexts=refs,all18_context_seed_arm_choices_before_readout=True))


def interval(rows,key,cfg):
    values=[]
    for site in sorted(set(r['source'] for r in rows)):
        xx=[r[key] for r in rows if r['source']==site]
        if any(v is None for v in xx):return dict(point=None,interval95=None,status='required_view_not_estimable')
        values.append(np.mean(xx))
    if len(values)!=8:raise ValueError('All eight source blocks required')
    x=np.asarray(values);pick=np.random.default_rng(cfg['bootstrap_seed']).integers(0,8,(3000,8))
    return dict(point=float(x.mean()),interval95=np.quantile(x[pick].mean(1),[.025,.975]).tolist(),
                source_blocks=8,risk_seeds_averaged_before_resampling=True,independent_confirmation=False)


def evaluate(reg,cfg,beat,*,replay=False):
    if not read(PUBLIC/'inference_freeze.json')['all18_context_seed_arm_choices_before_readout']:raise ValueError('Freeze all choices first')
    data=full.load_data();rows=[];primary=[];effects=[]
    for node in reg['jobs']:
        own,old=own_and_floor(node);job=node['job'];ix=old['local_ids'];target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        for seed in (17,29,43):
            bank={arm:load_cache(read(PUBLIC/'inference'/f'{job["key"]}_{arm}_seed{seed}.json')['cache']) for arm in method.ARMS}
            paths={arm:{name:full.method.paths_for(old['paths'],old['floor'],bank[arm][name]) for name in POLICIES} for arm in method.ARMS}
            for site in node['part']['names']:
                at=old['sites']==site
                for arm in method.ARMS:
                    policies={name:full.method.original.policy(p[at],old['floor'][at],old['cv'][at],target[at],valid[at],old['recordings'][at],
                        easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in paths[arm].items()}
                    rows.append(dict(job=job['key'],source=site,seed=seed,arm=arm,policies=policies))
                    for first,second in (('proposal_only','independent'),('independent','joint_matched'),('independent','joint_free')):
                        effects.append(dict(source=site,seed=seed,arm=arm,contrast=second+'_minus_'+first,
                            **full.method.original.paired(paths[arm][first][at],paths[arm][second][at],old['floor'][at],target[at],valid[at])))
                primary.append(dict(source=site,seed=seed,
                    **full.method.original.paired(paths['composed_anchor']['joint_free'][at],paths['role_separated']['joint_free'][at],
                                                 old['floor'][at],target[at],valid[at])))
        beat(state='complete_original_metrics',job=job['key'])
    summaries={}
    for arm in method.ARMS:
        subset=[r for r in rows if r['arm']==arm];summaries[arm]={}
        for name in POLICIES:
            policies=[r['policies'][name] for r in subset]
            summaries[arm][name]=dict(safety=full.method.safety(policies),
                source_seed_macro_known_gain=float(np.mean([p['known_ADE_gain_floor_percent'] for p in policies])),
                source_seed_macro_completion_lower_gain=float(np.mean([p['completion_lower_gain_floor_percent'] for p in policies])),
                safety_by_seed={str(s):full.method.safety([r['policies'][name] for r in subset if r['seed']==s]) for s in (17,29,43)},
                selected_by_seed={str(s):sum(round(r['policies'][name]['switch_rate']*r['policies'][name]['rows']) for r in subset if r['seed']==s) for s in (17,29,43)})
    primary_summary={k:interval(primary,k,cfg) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    effect_summary={arm:{contrast:{k:interval([r for r in effects if r['arm']==arm and r['contrast']==contrast],k,cfg)
        for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for contrast in sorted(set(r['contrast'] for r in effects))} for arm in method.ARMS}
    result=stable(dict(rows=rows,policies=summaries,primary=primary_summary,within_arm_effects=effect_summary,
        source_blocks=8,unique_recordings=24,required_record_views_per_arm_policy=72,
        risk_head_seeds=[17,29,43],forecaster_seeds=[17],fresh_models=48,result_source='fresh_run',
        independent_confirmation=False,original24group_primary='not_estimable_not_passed',
        original_rules_unchanged=True,new_outer_C_H_readout=False,identity_certification=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay:neural.rel.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('register','prepare','pilot','train','verify','infer','evaluate','replay'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,reg=register()
        if args.phase=='register':print('Matched role-separated full-action study registered.');return
        committed();start,guard,beat=runtime(cfg,args.phase)
        before={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase=='prepare':prepare_own(reg,guard,beat);prepare_A(reg,guard,beat)
        elif args.phase=='pilot':pilot(reg,guard,beat,resume=args.resume)
        elif args.phase in ('train','verify'):train(reg,guard,beat,resume=args.resume,verify=args.phase=='verify')
        elif args.phase=='infer':infer(reg,cfg,beat,resume=args.resume)
        elif args.phase=='evaluate':evaluate(reg,cfg,beat)
        else:
            infer(reg,cfg,beat,replay=True);evaluate(reg,cfg,beat,replay=True)
            neural.rel.base.exact(before,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        where=PRIVATE/'replays' if args.phase=='replay' else PUBLIC/'executions'
        once(where/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,phase=args.phase,
            predictions_recomputed=args.phase=='replay',timed_solver_recomputed=False if args.phase=='replay' else None))
        beat(state='phase_complete')


if __name__=='__main__':main()
