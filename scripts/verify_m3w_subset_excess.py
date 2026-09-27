"""Independent subset, fit, decision and fixed-denominator readout verification."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_subset_excess as run
from scripts.verify_m3w_query_excess_refit import check_selected_costs
from scripts.verify_m3w_fixed_floor_tail import reduce_check
TESTS=['tests/test_m3w_subset_excess.py','tests/test_m3w_subset_excess_verification.py',
       'tests/test_m3w_query_excess.py','tests/test_m3w_query_utility.py',
       'tests/test_m3w_causal_descriptor_head.py','tests/test_m3w_causal_descriptor_protocol.py',
       'tests/test_m3w_fixed_floor_probe.py','tests/test_m3w_query_excess_verification.py']


def reconstruct_bank(sites,recordings,frames,ids,moving,controller,envelope):
    bank=np.zeros((len(ids),3),bool);groups={}
    for i in range(len(ids)):
        groups.setdefault((str(sites[i]),str(recordings[i]),int(frames[i])),[]).append(i)
        bank[i,0]=bool(moving[i] and controller[i])
    for positions in groups.values():
        ordered=sorted([i for i in positions if moving[i]],key=lambda i:(float(envelope[i]),int(ids[i])))
        for rank,i in enumerate(ordered):bank[i,1 if 2*rank<len(ordered) else 2]=True
    return bank


def check_fixed_denominator(metric,floor,neural,known,take):
    selected=[i for i in range(len(take)) if take[i] and known[i]]
    harm=sum(max(float(neural[i])-float(floor[i]),0.) for i in selected)
    benefit=sum(max(float(floor[i])-float(neural[i]),0.) for i in selected)
    den=sum(float(floor[i]) for i in range(len(take)) if known[i])
    for key,value in [('positive_harm_sum',harm),('benefit_sum',benefit),
            ('selected_reference_error',sum(float(floor[i]) for i in selected)),('selected_known_count',len(selected))]:
        np.testing.assert_allclose(metric[key],value,rtol=1e-12,atol=1e-9)
    if den>0:np.testing.assert_allclose(metric['positive_harm_over_all_floor'],harm/den,rtol=1e-12,atol=1e-12)
    else:assert metric['positive_harm_over_all_floor'] is None
    np.testing.assert_allclose(metric['floor_error_sum']-metric['error_sum'],benefit-harm,rtol=1e-10,atol=1e-8)


def main():
    run.base.torch.set_num_threads(4);run.base.torch.set_num_interop_threads(1)
    cfg,data,jobs,oid,pid,pbound,bound,di,pi,identity=run.load()
    causal={k:data[k] for k in run.CAUSAL_KEYS}
    frozen=json.loads((run.PUBLIC/'decision_freeze.json').read_text());assert frozen['identity']==identity
    refs={Path(r['path']).stem:r for r in frozen['groups']}
    raw=json.loads((run.PRIVATE/'details.json').read_text())
    readout={(r['group'],r['site'],r['policy']):r['metric'] for r in raw['rows']}
    cost_views=queries=groups_checked=matched_queries=bank_rows=0
    totals={a:dict(heads=0,updates=0,training_queries=0,singleton_queries=0,
        loss_declined_heads=0,unknown_draws=0) for a in run.api.ARMS}
    for c in run.base.floor_api.contexts(causal,jobs,oid):
        ids=c['ids'];bank=reconstruct_bank(data['sites'][ids],data['recordings'][ids],data['frames'][ids],
            ids,c['moving'],c['neural_mask'],c['env'])
        np.testing.assert_array_equal(bank,run.masks(c,causal));bank_rows+=len(ids)
        cv,_,(floor,_),(neural,_)=run.base.floor_api.costs(c,data,np.arange(len(ids)))
        for pair in range(6):
            name=c['name']+f'_pair{pair}';r=refs[name];assert run.base.artifact(ROOT/r['path'])==r
            d=json.loads((ROOT/r['path']).read_text());assert d['future_fields_removed'] and not d['held_outcomes_used']
            for key in ('arrays','fit','parent_action'):assert run.base.artifact(ROOT/d[key]['path'])==d[key]
            fit=json.loads((ROOT/d['fit']['path']).read_text());roles=fit['identity']['roles_and_targets']
            run.base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
            with np.load(ROOT/d['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            held,p,pr=run.predictions(c,causal,d['fit'],identity);np.testing.assert_array_equal(a['ids'],ids[held])
            for site in roles['held_sites']:
                at=data['sites'][a['ids']]==site;ix=held[at]
                for policy in run.POLICIES:
                    take=np.zeros(len(ix),bool) if policy=='floor' else a[policy][at]
                    metric=readout[(name,site,policy)]
                    check_selected_costs(metric,cv[ix],floor[ix],neural[ix],take)
                    check_fixed_denominator(metric,floor[ix],neural[ix],np.isfinite(cv[ix]),take);cost_views+=1
            fit_at=np.isin(data['sites'][ids],roles['training_sites']);fit_ids=ids[fit_at];states={}
            parent_fit=fit['identity']['parent_fit'];assert run.base.artifact(ROOT/parent_fit['path'])==parent_fit
            pf=json.loads((ROOT/parent_fit['path']).read_text())
            old=run.api.head.read_checkpoint(ROOT/pf['artifacts']['pointwise']['path'])
            for arm,ref in fit['artifacts'].items():
                assert run.base.artifact(ROOT/ref['path'])==ref
                state=run.api.head.read_checkpoint(ROOT/ref['path']);states[arm]=state
                assert state['step']==2000 and state['settings']==cfg['head_training']
                np.testing.assert_array_equal(bank[fit_at],state['subsets'])
                assert run.base.inter.array_hash(state['subsets'])==fit['identity']['fitting_subset_hash']
                assert not state['draws'][~state['preprocess']['known']].any()
                for k in ('initial_model','draws','query_draws','query_keys','sampler_rng','torch_rng','step','settings','preprocess'):
                    run.api.exact(old[k],state[k])
                groups={}
                for i in np.flatnonzero(state['preprocess']['known']):
                    rid=fit_ids[i];key=(str(data['sites'][rid]),str(data['recordings'][rid]),int(data['frames'][rid]))
                    groups.setdefault(key,[]).append(i)
                assert state['query_keys']==sorted(groups)
                for site in roles['training_sites']:
                    qix=[i for i,k in enumerate(state['query_keys']) if k[0]==site]
                    assert state['query_draws'][qix].sum()==32000
                info=fit['fits'][arm];assert info['complete'] and info['unknown_rows_sampled']==0
                t=totals[arm];t['heads']+=1;t['updates']+=state['step'];t['training_queries']+=len(groups)
                t['singleton_queries']+=sum(len(g)==1 for g in groups.values())
                t['loss_declined_heads']+=info['trace'][-1]['monitor'][arm]<info['trace'][0]['monitor'][arm]
            run.api.assert_matched(*[states[a] for a in run.api.ARMS]);matched_queries+=len(old['query_keys'])
            parent=json.loads((ROOT/d['parent_action']['path']).read_text())
            assert run.base.artifact(ROOT/parent['arrays']['path'])==parent['arrays']
            with np.load(ROOT/parent['arrays']['path'],allow_pickle=False) as z:
                for new,oldkey in {'ids':'ids','eligible':'eligible','parent_independent':'parent_independent',**run.CACHED}.items():
                    np.testing.assert_array_equal(a[new],z[oldkey])
            by_query={}
            for i,rid in enumerate(a['ids']):
                by_query.setdefault((str(data['sites'][rid]),str(data['recordings'][rid]),int(data['frames'][rid])),[]).append(i)
            for positions in by_query.values():
                at=np.array(positions);k=int(a['parent_independent'][at].sum())
                for arm in run.api.ARMS:
                    q=np.column_stack((p[arm][at,1].astype(float)-.02*p[arm][at,0].astype(float),
                        p[arm][at,3].astype(float)-.02*p[arm][at,2].astype(float)))/pr['cost_scale']
                    ix=a[arm+'_independent'][at];jx=a[arm+'_joint'][at]
                    np.testing.assert_array_equal(ix,a['eligible'][at] & (q<=0).all(1))
                    assert ix.sum()==jx.sum() and not (jx & ~a['eligible'][at]).any()
                    scale=np.maximum(np.max(abs(q),axis=0),1e-12)
                    assert np.all((q/scale)[jx].sum(0)<=1e-10)
                    order=sorted([i for i in range(len(at)) if a['eligible'][at][i]],key=lambda i:(float(q[i].max()),int(a['ids'][at][i])))
                    expected=np.zeros(len(at),bool);expected[order[:k]]=True
                    np.testing.assert_array_equal(expected,a[arm+'_rank'][at])
            queries+=len(by_query);groups_checked+=1
            if groups_checked%18==0:print(json.dumps(dict(state='independent_verified',groups=groups_checked,queries=queries)),flush=True)
    assert json.loads((run.PUBLIC/'training_summary.json').read_text())==totals
    for name in ('prediction_replay','fit_replay','evaluation_replay'):
        assert json.loads((run.PUBLIC/(name+'.json')).read_text())['exact']
    er=json.loads((run.PUBLIC/'evaluation_replay.json').read_text())
    for k in ('summary','details'):assert run.base.artifact(ROOT/er[k]['path'])==er[k]
    s=json.loads((run.PUBLIC/'summary.json').read_text());reductions=0
    for section,key in [('summary','rows'),('paired','contrasts'),('quality','quality')]:
        for policy,metrics in s[section].items():
            rr=[r for r in raw[key] if r['policy']==policy]
            for k,v in metrics.items():reductions+=reduce_check(v,rr,k,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in s['by_seed'].items():
        for policy,metrics in policies.items():
            rr=[r for r in raw['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for k,v in metrics.items():reductions+=reduce_check(v,rr,k,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    old=json.loads((run.previous.PUBLIC/'summary.json').read_text())
    for new,parent in {'floor':'floor',**run.CACHED}.items():
        for k,v in old['summary'][parent].items():assert s['summary'][new][k]==v
    empty=json.loads((run.PUBLIC/'preflight.json').read_text())['fixed_parent_zero_action_views']
    assert len(empty)==10
    for r in empty:
        for arm in run.api.ARMS:
            m=readout[(r['group'],r['site'],arm+'_rank')]
            assert m['intervention_rate']==0 and m['selected_positive_harm_ratio'] is None and m['positive_harm_over_all_floor']==0
    assert s['paired']['subset_aggregate_rank_vs_subset_pointwise_rank']['selected_harm_reduction_pp']['ci95'] is None
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    proc=subprocess.run([sys.executable,'scripts/report_m3w_subset_excess.py'],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    assert all(run.base.digest(run.PUBLIC/p)==h for p,h in artifacts.items())
    parent=json.loads((run.previous.PUBLIC/'verification.json').read_text());bindings=dict(parent['source_bindings'])
    bindings.update(identity['bindings'])
    bindings.update({p:run.base.digest(ROOT/p) for p in ['scripts/report_m3w_subset_excess.py','scripts/verify_m3w_subset_excess.py',*TESTS]})
    run.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.base.artifact(log),groups=108,heads=216,updates=432000,
        independent_query_checks=queries,independent_subset_bank_rows=bank_rows,matched_fitting_queries=matched_queries,
        independent_locality_reductions=reductions,independently_accounted_cost_views=cost_views,
        first_paired_fit_replay_exact=True,all_prediction_action_replays_exact=True,evaluation_replay_exact=True,
        parent_controls_exact=True,report_figure_byte_reproducible=True,formal_primary_replaced=False,
        fixed_denominator_is_risk_certificate=False,full_legacy_suite='not_run',cold_raw_rebuild='not_run',
        independent_confirmation=False,calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(tests=tests,queries=queries,bank_rows=bank_rows,cost_views=cost_views,reductions=reductions)))


if __name__=='__main__':main()
