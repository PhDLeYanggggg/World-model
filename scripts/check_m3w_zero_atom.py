"""Separate exact-count/risk arithmetic, not independent scientific validation."""
from pathlib import Path
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from scripts import run_m3w_zero_atom as run


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    ctx=run.load(); cfg=ctx['cfg']; root=ROOT/cfg['output']; public=ROOT/cfg['reports']
    identity=run.read(root/'identity.json'); assert identity==ctx['identity']
    a=run.read(public/'analysis.json'); done=run.read(root/'decisions_complete.json')
    assert a['experiment_sha256']==run.base.file_digest(root/'identity.json')
    assert a['decisions_sha256']==run.base.file_digest(root/'decisions_complete.json')
    data=ctx['parent']['pack'][1][1]; fits=0
    for r in a['fits']:
        assert run.base.file_digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        cp=joblib.load(ROOT/r['checkpoint']); assert cp['identity']==r['identity']
        old=ctx['refs'][r['view'],r['action']]
        frozen=joblib.load(ROOT/old['checkpoint'])
        assert len(cp['entries'])==len(frozen['model'].estimators_)==128
        assert r['view'].split('_seed')[0] not in r['identity']['training_sites']
        assert r['support']['unknown_draws']==0
        for tree, entry in zip(frozen['model'].estimators_,cp['entries']):
            leaf=tree.tree_.children_left==-1
            np.testing.assert_array_equal(entry['total'][leaf],tree.tree_.weighted_n_node_samples[leaf])
            assert (entry['positive']>=0).all() and (entry['positive']<=entry['total']).all()
            assert (entry['unique'][leaf]>=64).all()
        fits+=1
    counts,risk_checks=0,0; archives={}
    for ref in done['archives']:
        assert run.base.file_digest(ROOT/ref['path'])==ref['sha256']
        r=run.read(ROOT/ref['path']); assert run.base.file_digest(ROOT/r['path'])==r['sha256']
        archives[r['view'],r['action']]=r['path']
        old=ctx['decisions'][r['view'],r['action']]
        with np.load(ROOT/old['path'],allow_pickle=False) as z:
            ids,f,d=z['ids'],z['fractions'],z['distance']
        with np.load(ROOT/r['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids']); bits=z['choices']
            admit=z['supported']&(z['probability']==0)
            assert not (bits[:,:3]&~admit[:,None]).any()
            assert (z['minimum_unique_rows']>=64).all()
        gain=(f[:,0]-f[:,1])*d; risk=(f[:,2]-f[:,3])*d; den=f[:,4]*old['cutoff']
        h=data['geometry'][ids,:16].reshape(-1,8,2)
        eligible=(gain>0)&(d>0)&(den>0)&np.any(h[:,-1]!=h[:,-2],axis=1)
        assert not (bits&~eligible[:,None]).any()
        assert not (bits[:,[0,3]]&(risk>cfg['easy_rho']*den)[:,None]).any()
        groups={}
        for i,idx in enumerate(ids): groups.setdefault((str(data['recordings'][idx]),int(data['frames'][idx])),[]).append(i)
        assert len(groups)==len(r['queries'])
        for indexes in groups.values():
            q=np.array(indexes); np.testing.assert_array_equal(bits[q,:3].sum(0),bits[q,3:].sum(0)); counts+=3
            for col in (1,4):
                assert math.fsum(risk[q][bits[q,col]])<=cfg['easy_rho']*float(den[q].sum()); risk_checks+=1
            rr=np.maximum(risk[q],0)-cfg['easy_rho']*den[q]
            for col in (2,5): assert math.fsum(rr[bits[q,col]])<=0; risk_checks+=1
            for col in range(3):
                assert math.fsum(gain[q][bits[q,col+3]])>=math.fsum(gain[q][bits[q,col]])
    reductions=0; harm_details=[]; event_diagnostics=[]; unique_harms={}
    for action in cfg['actions']:
        errors={p:[] for p in a['policies'] if p.startswith(('atom_','matched_'))}
        for seed in cfg['seeds']:
            ref=next(r for r in ctx['parent']['pack'][1][3]['outcome_archives'] if r['path'].endswith(f'{action}_seed{seed}.npz'))
            assert run.base.file_digest(ROOT/ref['path'])==ref['sha256']
            with np.load(ROOT/ref['path'],allow_pickle=False) as z: o={k:z[k] for k in z.files}
            bits=np.zeros((len(data['sites']),6),bool)
            original=np.zeros((len(data['sites']),3),bool)
            probability=np.zeros(len(bits)); distance=np.zeros(len(bits))
            for site in cfg['sites']:
                view=f'{site}_seed{seed}'
                with np.load(ROOT/archives[view,action],allow_pickle=False) as z:
                    bits[z['ids']]=z['choices']; probability[z['ids']]=z['probability']
                    original[z['ids']]=z['original']
                with np.load(ROOT/ctx['decisions'][view,action]['path'],allow_pickle=False) as z:
                    distance[z['ids']]=z['distance']
            h=data['geometry'][:,:16].reshape(-1,8,2)
            moving=np.any(h[:,-1]!=h[:,-2],axis=1)
            use=o['complete']&moving&(distance>0)
            zero=use&o['zero_CV']
            event_diagnostics.append(dict(action=action,seed=seed,complete_moving_effective=int(use.sum()),
                zero_moving_effective=int(zero.sum()),zero_predicted_absent=int((zero&(probability==0)).sum()),
                brier_complete_moving_effective=float(np.mean((probability[use]-o['zero_CV'][use])**2))))
            for col,p in enumerate(errors):
                e=np.where(bits[:,col],o['candidate_ade'],o['cv']); errors[p].append(e)
                r=a['summary'][action+'__'+p]['seeds'][str(seed)]
                assert r['selected']==int(bits[:,col].sum())
                assert r['zero_CV_harmed']==int((bits[:,col]&o['zero_CV']&(e>0)).sum())
                harmed=bits[:,col]&o['zero_CV']&(e>0)
                reference_harmed=original[:,col%3]&o['zero_CV']&(o['candidate_ade']>0)
                key=action+'__'+p
                unique_harms.setdefault(key,set()).update(np.flatnonzero(harmed).tolist())
                for site in cfg['sites']:
                    ix=harmed&(data['sites']==site)
                    if not ix.any(): continue
                    harm_details.append(dict(action=action,policy=p,seed=seed,site=site,count=int(ix.sum()),
                        not_harmed_by_original=int((ix&~reference_harmed).sum()),
                        harm_ade_range=[float(e[ix].min()),float(e[ix].max())],
                        zero_probability_range=[float(probability[ix].min()),float(probability[ix].max())]))
        for p,arrays in errors.items():
            e=np.mean(arrays,axis=0); r=a['summary'][action+'__'+p]
            for subset,mask in [('all',np.ones(len(e),bool))]+[(k,o[k]) for k in ('complete','hard','positive_easy','zero_CV')]:
                report=r['ADE'] if subset=='all' else r['subsets'][subset]
                for site in cfg['sites']:
                    use=mask&(data['sites']==site)&np.isfinite(o['cv']); row=report['by_scene'][site]
                    assert int(use.sum())==row['rows']
                    if use.any(): np.testing.assert_allclose(e[use].mean(),row['model_error'],rtol=1e-12)
                    reductions+=1
    for action in cfg['actions']:
        for p in ('old_strict','cutoff_point','cutoff_population','cutoff_selected'):
            assert a['summary'][action+'__'+p]==ctx['old']['summary'][action+'__'+p]
    run.base.assert_current(identity)
    report=dict(result_source='fresh_run',method='separate_arithmetic_same_executor',
        analysis_sha256=run.base.file_digest(public/'analysis.json'),code_sha256=run.base.file_digest(Path(__file__)),
        fits=fits,matched_query_counts=counts,risk_checks=risk_checks,scene_reductions=reductions,
        original_rows_exact=12,exhaustive_optimality='not_run',independent_confirmation=False)
    diagnostic=dict(result_source='fresh_run',analysis_sha256=report['analysis_sha256'],
        scope='descriptive_postreadout_no_policy_change',unique_harmed_windows={k:len(v) for k,v in unique_harms.items()},
        harm_details=harm_details,event_diagnostics=event_diagnostics,
        full_complete_brier_in_analysis_includes_stopped_rows_outside_policy_support=True,
        low_brier_is_not_rare_event_recall_or_calibration_guarantee=True)
    run.base.immutable_json(public/'postreadout_diagnostics.json',diagnostic)
    run.base.immutable_json(public/'separate_checks.json',report); print(json.dumps(report))


if __name__=='__main__': main()
