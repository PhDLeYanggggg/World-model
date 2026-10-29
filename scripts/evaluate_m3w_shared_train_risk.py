"""One fixed-final TRAIN-internal readout, never official/test selection."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts import run_m3w_shared_train_risk as run
from src.world_model import m3w_shared_train_risk as api
from src.world_model import m3w_scaled_risk_readout as readout

PUBLIC=run.PUBLIC/'readout';REG=PUBLIC/'registration.json'
ARMS=('local','shared_global','shared_source')
PAIRS=(('local','shared_global'),('local','shared_source'),('shared_global','shared_source'))


def evaluate(args,predictions):
    if set(predictions)!=set(ARMS): raise ValueError('All fixed arms required')
    x,env,y,_,sites,recs,frames,pr=args
    valid=api.eligible(args);n=len(x);ids=api.np.arange(n)
    actions={name:valid & (api.old.core.signed(p)[:,0]>0) &
             (api.old.core.signed(p)[:,1:]<=0).all(1) for name,p in predictions.items()}
    policies={name:readout.policy_score(y,a,env) for name,a in actions.items()}
    scores={name:readout.previous.score(p,y,pr,sites,recs,frames,api.np.ones(n,bool))['conditional_MSE']
            for name,p in predictions.items()}
    known=api.np.isfinite(y).all(1);den=float(y[known,2].sum())
    contrasts={};pairs={};matched={}
    for left,right in PAIRS:
        name=right+'_minus_'+left
        contrasts[name+'_MSE']=scores[right]-scores[left] if scores[right] is not None and scores[left] is not None else None
        a,b=readout.previous.policy.matched(actions[left],actions[right],
            api.old.core.signed(predictions[left])[:,0],api.old.core.signed(predictions[right])[:,0],recs,frames,ids)
        matched[name]={left:readout.policy_score(y,a,env),right:readout.policy_score(y,b,env)}
        for mode,aa,bb in [('full',actions[left],actions[right]),('matched',a,b)]:
            pair=readout.previous.paired_completion(y,aa,bb,env)
            pairs[name+'_'+mode]=pair
            contrasts[name+'_'+mode]=100*pair['lower_mass']/den if den>0 else None
    per_recording={str(r):{name:readout.policy_score(y[recs==r],a[recs==r],env[recs==r])
                          for name,a in actions.items()} for r in api.np.unique(recs)}
    return dict(rows=n,known_rows=int(known.sum()),unknown_rows=int((~known).sum()),
        full_known_reference_mass=den,policies=policies,scores=scores,contrasts=contrasts,
        pairs=pairs,matched_policies=matched,recordings=per_recording,
        diagnostics={k:readout.selected_diagnostics(p,y,actions[k],pr['scale']) for k,p in predictions.items()},
        inference_label_filter=False),actions


def summarize(rows,reg,cfg):
    expected={(context,item['packet']['identity']['source'],seed)
              for context,items in reg['contexts'].items() for item in items for seed in reg['seeds']}
    actual=[(r['context'],r['source'],r['seed']) for r in rows]
    if set(actual)!=expected or len(actual)!=len(expected): raise ValueError('Every72 fixed view required')
    contrasts={field:readout.previous.interval(rows,field,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
               for field in rows[0]['result']['contrasts']}
    for v in contrasts.values():
        v['status']='nominal_TRAIN_internal_not_independent_confirmation' if v['CI95'] is not None else v['status']
    safety={}
    for arm in ARMS:
        policies=[r['result']['policies'][arm] for r in rows]
        safety[arm]=dict(views=len(policies),selected_occurrences=sum(p['selected_count'] for p in policies),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk']>.02+1e-12 for p in policies),
            easy_upper_violations=sum(p['easy_selected_risk_upper'] is not None and p['easy_selected_risk_upper']>.02+1e-12 for p in policies),
            undefined_easy_risk=sum(p['easy_selected_risk_upper'] is None for p in policies),
            completion_supported=sum(p['finite_completion_supported'] for p in policies),
            empty_full_policy=sum(p['selected_count']==0 for p in policies))
    routing={}
    for arm in ARMS[1:]:
        failures=[]
        for mode in ('full','matched'):
            v=contrasts[arm+'_minus_local_'+mode]
            if v['CI95'] is None or v['CI95'][0]<=0: failures.append(mode+'_utility_not_supported')
        mse=contrasts[arm+'_minus_local_MSE']
        if mse['CI95'] is None or mse['CI95'][0]>0:failures.append('MSE_missing_or_demonstrably_worse')
        if safety[arm]['completion_supported']!=72:failures.append('full_policy_safety_support_failed')
        if arm=='shared_source':
            for mode in ('full','matched'):
                v=contrasts['shared_source_minus_shared_global_'+mode]
                if v['CI95'] is None or v['CI95'][0]<0:failures.append(mode+'_no_nondegradation_support_vs_shared_global')
        routing[arm]=dict(eligible_for_later_development_design=not failures,failure_reasons=failures)
    return dict(views=72,source_localities=len({r['source'] for r in rows}),contrasts=contrasts,safety=safety,
        routing=routing,outer_development_read=False,independent_roles_read=False,deployment_changed=False,
        new_forecasters=False,scientific_success=False)


def register():
    reg,cfg,_=run.admission()
    files=[Path(__file__).resolve(),run.ROOT/'tests/test_m3w_shared_train_readout.py']
    value=dict(bindings={str(p.relative_to(run.ROOT)):run.base.sha(p) for p in files},
        training_registration_sha256=run.base.sha(run.REG),expected_views=72,
        arms=list(ARMS),pairs=[list(p) for p in PAIRS],risk_budget=.02,
        matched_risk_is_reporting_only=True,outer_development_read=False,independent_roles_read=False)
    run.base.once(REG,value);return reg,cfg


def frozen_models():
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text());models={}
    if freeze['fresh_fits']!=108 or freeze['registration_sha256']!=run.base.sha(run.REG):
        raise ValueError('All finals must be frozen before inner-held inference')
    for ref in freeze['fits']:
        receipt=run.ROOT/ref['path']
        if run.base.sha(receipt)!=ref['sha256']:raise ValueError('Changed fit receipt')
        doc=json.loads(receipt.read_text());i=doc['identity'];key=i['context'],i['seed'],i['mode'],i['source']
        path=run.ROOT/doc['checkpoint']['path']
        if path.is_symlink() or not path.resolve().is_relative_to((run.PRIVATE/'heads').resolve()) or run.base.sha(path)!=doc['checkpoint']['sha256']:
            raise ValueError('Changed model payload')
        if key in models or doc['step']!=2000 or doc['inner_held_scored'] or doc['independent_roles_read']:
            raise ValueError('Invalid frozen fit identity')
        models[key]=doc
    if len(models)!=108:raise ValueError('All108 distinct models required')
    return models


def execute(resume=False):
    reg,cfg=register();_,_,parent=run.admission()
    for rel in [*json.loads(REG.read_text())['bindings'],str(REG.relative_to(run.ROOT))]:
        p=subprocess.run(['git','show','HEAD:'+rel],cwd=run.ROOT,capture_output=True)
        if p.returncode or p.stdout!=(run.ROOT/rel).read_bytes():raise ValueError('Commit readout before held inference')
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1)
    models=frozen_models();rows=[];refs=[];started=time.monotonic()
    if (PUBLIC/'complete.json').exists():raise ValueError('Completed readout immutable')
    def beat(**kw):
        row=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),phase='inner_readout',**kw)
        tmp=run.PRIVATE/'readout_heartbeat.tmp';tmp.write_text(json.dumps(row)+'\n');os.replace(tmp,run.PRIVATE/'readout_heartbeat.json')
        print(json.dumps(row),flush=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context,items in sorted(reg['contexts'].items()):
            _,held=run.packets(items,parent,beat)
            for seed in reg['seeds']:
                shared={mode:api.old.core.read_checkpoint(run.ROOT/models[context,seed,mode,None]['checkpoint']['path']) for mode in ARMS[1:]}
                for source,args in sorted(held.items()):
                    path=PUBLIC/'groups'/(context+'_'+source+'_seed'+str(seed)+'.json')
                    if path.exists() and not resume:raise ValueError('Explicit readout resume required')
                    binding=dict(readout_registration_sha256=run.base.sha(REG),training_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'))
                    if path.exists():
                        row=json.loads(path.read_text())
                        if any(row[k]!=v for k,v in binding.items()):raise ValueError('Changed readout receipt')
                    else:
                        local=api.old.core.read_checkpoint(run.ROOT/models[context,seed,'local',source]['checkpoint']['path'])
                        api.old.core.exact(local['preprocess'],args[-1])
                        prediction={'local':api.old.predict(local,args[0],args[1])[0]}
                        api.np.testing.assert_array_equal(prediction['local'],api.old.predict(local,args[0],args[1])[0])
                        for mode,state in shared.items():
                            api.old.core.exact(state['source_preprocess'][source],args[-1])
                            prediction[mode]=api.predict_shared(state,source,args[0],args[1])
                            api.np.testing.assert_array_equal(prediction[mode],api.predict_shared(state,source,args[0],args[1]))
                        result,actions=evaluate(args,prediction)
                        row=dict(**binding,context=context,source=source,seed=seed,result=result,
                            prediction_hashes={k:api.old.parent.forest.fingerprint(v) for k,v in prediction.items()},
                            action_hashes={k:api.old.parent.forest.fingerprint(v) for k,v in actions.items()},
                            exact_inference_replay=True,result_source='fresh_run_fixed_final_inner_TRAIN_holdout',
                            outer_development_read=False,independent_roles_read=False,deployment_changed=False)
                        run.base.once(path,row)
                    rows.append(row);refs.append(dict(path=str(path.relative_to(run.ROOT)),sha256=run.base.sha(path)))
                    beat(state='view_complete',views=len(rows),total=72)
        run.base.once(PUBLIC/'summary.json',summarize(rows,reg,cfg))
        run.base.once(PUBLIC/'complete.json',dict(groups=refs,seconds=time.monotonic()-started,
            summary_sha256=run.base.sha(PUBLIC/'summary.json'),registration_sha256=run.base.sha(REG),
            training_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'),
            outer_development_read=False,independent_roles_read=False,deployment_changed=False))
        beat(state='complete',views=72)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','run']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register':register();print('Registered fixed TRAIN-internal readout')
    else:execute(a.resume)
