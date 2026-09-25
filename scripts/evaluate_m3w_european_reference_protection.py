"""Predeclared reference-protection comparisons on historically opened source C."""
import json
import numpy as np
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from scripts.audit_m3w_selected_query_budget import account
from scripts.audit_m3w_easy_harm_fitting import summarize_fit
from scripts.audit_m3w_easy_harm_transport import evaluation_weights
from src.evaluation.m3w_bridge_risk_calibration import evidence
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics


def evaluate(run,cfg,identity):
    freeze=run.PUBLIC/'decision_freeze.json'; run.parent.previous.require_committed(freeze)
    run.checked_training(identity)
    assert json.loads(freeze.read_text())['manifest']==run.artifact(run.PRIVATE/'training_complete.json')
    finish=run.PUBLIC/'completion_checks.json'
    if finish.exists():
        old=json.loads(finish.read_text()); assert old['identity']==identity and old['all_passed']
        for r in old['groups']+[old['seeds']]: assert run.artifact(run.ROOT/r['path'])==r
        run.beat('cached_verified_C_readout'); return
    rows={p:[] for p in cfg['pairs']}; refs=[]
    for g,data,pairs in run.parent.contexts(identity['parent']['parent']):
        ci=pairs['C']['ids']; name=g['group']; sites=data['sites'][ci]; roster=sorted(set(sites))
        queries=run.parent.query_groups(sites,data['recordings'][ci],data['frames'][ci]); cv=data['baseline_ade'][ci,1]
        masks=dict(all=np.ones(len(ci),bool),easy=(cv>0)&(cv<=g['easy_cut']),hard=cv>=g['hard_cut'],complete=data['valid'][ci].all(1))
        for pair in cfg['pairs']:
            path=run.PUBLIC/'groups'/(name+'_'+pair+'.json'); rp=run.PRIVATE/'eval_receipts'/path.name
            if path.exists():
                assert run.artifact(path)==json.loads(rp.read_text()); row=json.loads(path.read_text())
                assert row['identity']==identity; rows[pair].append(row); refs.append(run.artifact(path)); continue
            run.beat('C_readout',group=name,pair=pair); x,env,r,p=pairs['C'][pair]; checks=dict(coordinates=0,metrics=0)
            def errors(v):
                pred=v.astype(float)+data['origin'][ci,None]
                a=native_errors(pred,data['target_eval'][ci],data['valid'][ci],np.ones(len(ci)))
                b=run.parent.base.cross.independent.coordinate_errors(pred,data['target_eval'][ci],data['valid'][ci])
                for u,w in zip(a,b): run.parent.base.cross.independent.close(u,w); checks['coordinates']+=1
                return a
            ra,rf=errors(r); pa,pf=errors(p)
            source=run.parent.previous.PRIVATE/'source'/(name+'_'+pair)
            with np.load(source/'labels.npz',allow_pickle=False) as z:
                for k,v in (('reference',ra),('candidate',pa),('cv',cv)): np.testing.assert_array_equal(v,z[k])
            with np.load(source/'scores.npz',allow_pickle=False) as z: oldscore={k:z[k].copy() for k in z.files}
            scores={}; states={}
            for arm in cfg['arms']:
                directory=run.PRIVATE/'heads'/(name+'_'+pair+'_'+arm)
                _,states[arm]=run.method.restore(directory)
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ci); scores[arm]=z['scores'].copy()
            with np.load(run.prior.PRIVATE/'decisions'/(name+'_'+pair+'.npz'),allow_pickle=False) as z:
                old={v:z[v].copy() for v in run.CONTROLS.values()}
            bits=run.actions(old,scores,oldscore['neural__utility'],oldscore['moving'],env,queries,ci)
            with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'.npz'),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci)
                for k,v in bits.items(): np.testing.assert_array_equal(v,z[k])
            predictions={k:(np.where(v,pa,ra),np.where(v,pf,rf)) for k,v in bits.items()}
            def metric(a,b,mask):
                v=paired_scene_metrics(a[mask],b[mask],sites[mask],expected_scenes=roster,
                    dataset='EuropeanSquares_source_development',coordinate_unit='image_pixel',
                    bootstrap_resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
                run.parent.base.cross.independent.check_metric(a,b,sites,mask,v,cfg); checks['metrics']+=1
                return v
            oldrow=json.loads((run.prior.PUBLIC/'groups'/path.name).read_text())
            views={k:oldrow['views'][v] for k,v in run.CONTROLS.items()}
            for key in run.NEW_POLICIES:
                ade,fde=predictions[key]
                views[key]=dict(ADE_vs_raw={s:metric(ade,predictions['raw_neural'][0],m) for s,m in masks.items()},
                    ADE_vs_reference={s:metric(ade,ra,m) for s,m in masks.items()},easy_vs_CV=metric(ade,cv,masks['easy']),
                    FDE_vs_raw=metric(fde,predictions['raw_neural'][1],masks['all']),switch_rate=float(bits[key].mean()),
                    risk=evidence(bits[key],cv,ra,pa,sites,easy_cut=g['easy_cut']),
                    tails={s:dict(p95=float(np.nanquantile(ade[sites==s],.95)),p99=float(np.nanquantile(ade[sites==s],.99))) for s in roster})
            comparisons=[('protected_'+m,'continued_'+m) for m in ('all','dual','scene','joint')]
            comparisons += [('continued_joint','mean_joint')]
            comparisons += [('protected_joint',k) for k in ('mean_joint','sampling_joint','raw_neural','raw_ridge','protected_dual','protected_hash_matched')]
            contrasts={a+'_vs_'+b:{s:metric(predictions[a][0],predictions[b][0],m) for s,m in masks.items()} for a,b in comparisons}
            target=run.parent.method.event_targets(cv,ra,pa,g['easy_cut']); weights=evaluation_weights(target,sites)
            diagnostics={a:summarize_fit(scores[a],target,weights,states[a]['preprocess']['cost_scale'],
                states[a]['loss_scales'],bits['raw_neural']) for a in cfg['arms']}
            row=dict(identity=identity,group=name,pair=pair,producer=g['producer'],controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]),views=views,contrasts=contrasts,
                query_budget={a:account(scores[a],target,bits[a+'_joint'],queries,sites) for a in cfg['arms']},
                matched_action_diagnostics=diagnostics,checks=checks,rows=len(ci),queries=len(queries),verified=True,
                result_source='fresh_run_new_policies_cached_verified_old_controls',
                readout_role='historically_opened_source_C_not_independent_confirmation',
                parent_readout=run.artifact(run.prior.PUBLIC/'groups'/path.name))
            run.immutable_json(path,row); run.immutable_json(rp,run.artifact(path)); rows[pair].append(row); refs.append(run.artifact(path))
    seed=run.PUBLIC/'seed_averaged_metrics.json'; run.immutable_json(seed,{p:seed_summary(v,cfg) for p,v in rows.items()})
    run.immutable_json(finish,dict(identity=identity,groups=refs,seeds=run.artifact(seed),all_passed=True,
        checks={k:sum(v['checks'][k] for groups in rows.values() for v in groups) for k in ('coordinates','metrics')},
        selection_access=False,reserved_calibration_access=False,confirmation_access=False))
    run.beat('C_readout_complete',groups=len(refs))
