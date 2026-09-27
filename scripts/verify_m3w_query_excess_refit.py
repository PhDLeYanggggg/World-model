"""Independent grouping, matched fitting, causal inference and risk-readout checks."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_query_excess_refit as run
from scripts.verify_m3w_fixed_floor_tail import reduce_check
TESTS=['tests/test_m3w_query_excess.py','tests/test_m3w_query_utility.py',
    'tests/test_m3w_causal_descriptor_head.py','tests/test_m3w_causal_descriptor_protocol.py',
    'tests/test_m3w_fixed_floor_probe.py']


def main():
    run.base.torch.set_num_threads(4);run.base.torch.set_num_interop_threads(1)
    cfg,data,jobs,oid,pid,pbound,bound,di,identity=run.load()
    causal={k:data[k] for k in run.CAUSAL_KEYS}
    frozen=json.loads((run.PUBLIC/'decision_freeze.json').read_text());assert frozen['identity']==identity
    refs={Path(r['path']).stem:r for r in frozen['groups']}
    groups_checked=queries=matched_fit_queries=0;training_summary={a:dict(heads=0,updates=0,training_queries=0,
        singleton_queries=0,loss_declined_heads=0,unknown_draws=0) for a in cfg['arms']}
    for c in run.base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}';r=refs[name];assert run.base.artifact(ROOT/r['path'])==r
            d=json.loads((ROOT/r['path']).read_text());assert d['future_fields_removed'] and not d['held_outcomes_used']
            assert run.base.artifact(ROOT/d['arrays']['path'])==d['arrays']
            assert run.base.artifact(ROOT/d['fit']['path'])==d['fit']
            fit=json.loads((ROOT/d['fit']['path']).read_text());roles=fit['identity']['roles_and_targets']
            run.base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
            with np.load(ROOT/d['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            held,p,pr=run.predictions(c,causal,name,d['fit'],identity)
            np.testing.assert_array_equal(a['ids'],c['ids'][held])
            fit_ids=c['ids'][np.isin(data['sites'][c['ids']],roles['training_sites'])]
            states={}
            for arm,ref in fit['artifacts'].items():
                assert run.base.artifact(ROOT/ref['path'])==ref
                state=run.api.head.read_checkpoint(ROOT/ref['path']);states[arm]=state
                assert state['step']==cfg['head_training']['steps'] and state['settings']==cfg['head_training']
                assert not state['draws'][~state['preprocess']['known']].any()
                assert state['query_draws'].sum()==2000*32
                by_key={}
                for i in np.flatnonzero(state['preprocess']['known']):
                    rid=fit_ids[i];key=(str(data['sites'][rid]),str(data['recordings'][rid]),int(data['frames'][rid]))
                    by_key.setdefault(key,[]).append(i)
                assert state['query_keys']==sorted(by_key)
                for site in roles['training_sites']:
                    qix=[i for i,k in enumerate(state['query_keys']) if k[0]==site]
                    assert state['query_draws'][qix].sum()==2000*16
                info=fit['fits'][arm];assert info['complete'] and info['unknown_rows_sampled']==0
                t=training_summary[arm];t['heads']+=1;t['updates']+=state['step'];t['training_queries']+=len(by_key)
                t['singleton_queries']+=sum(len(g)==1 for g in by_key.values())
                t['loss_declined_heads']+=info['trace'][-1]['monitor'][arm]<info['trace'][0]['monitor'][arm]
            run.api.assert_matched(states['pointwise'],states['query']);matched_fit_queries+=len(states['query']['query_keys'])
            parent=d['parent_action'];assert run.base.artifact(ROOT/parent['path'])==parent
            pd=json.loads((ROOT/parent['path']).read_text())
            for ref in pd['artifacts'].values():assert run.base.artifact(ROOT/ref['path'])==ref
            with np.load(ROOT/pd['artifacts']['decisions']['path'],allow_pickle=False) as z:
                for key,old in [('ids','ids'),('eligible','eligible'),('parent_independent','independent'),('parent_joint','joint_utility'),('utility_topk_parent','utility_topk')]:
                    np.testing.assert_array_equal(a[key],z[old])
            by_query={}
            for i,rid in enumerate(a['ids']):
                by_query.setdefault((str(data['sites'][rid]),str(data['recordings'][rid]),int(data['frames'][rid])),[]).append(i)
            for positions in by_query.values():
                at=np.array(positions);k=int(a['parent_independent'][at].sum())
                for arm in cfg['arms']:
                    assert a[arm+'_rank'][at].sum()==k
                    assert not (a[arm+'_rank'][at] & ~a['eligible'][at]).any()
                    q=np.column_stack((p[arm][at,1].astype(float)-.02*p[arm][at,0].astype(float),
                        p[arm][at,3].astype(float)-.02*p[arm][at,2].astype(float)))/pr['cost_scale']
                    ix=a[arm+'_independent'][at];jx=a[arm+'_joint'][at]
                    np.testing.assert_array_equal(ix,a['eligible'][at] & (q<=0).all(1))
                    assert ix.sum()==jx.sum() and not (jx & ~a['eligible'][at]).any()
                    scale=np.maximum(np.max(abs(q),axis=0),1e-12)
                    assert np.all((q/scale)[jx].sum(0)<=1e-10)
                    order=np.lexsort((a['ids'][at],q.max(1)));order=order[a['eligible'][at][order]]
                    expected=np.zeros(len(at),bool);expected[order[:k]]=True
                    np.testing.assert_array_equal(expected,a[arm+'_rank'][at])
            queries+=len(by_query);groups_checked+=1
            if groups_checked%18==0:print(json.dumps(dict(state='independent_verified',groups=groups_checked,queries=queries)),flush=True)
    assert json.loads((run.PUBLIC/'training_summary.json').read_text())==training_summary
    assert json.loads((run.PUBLIC/'prediction_replay.json').read_text())['exact']
    assert json.loads((run.PUBLIC/'fit_replay.json').read_text())['exact']
    er=json.loads((run.PUBLIC/'evaluation_replay.json').read_text());assert er['exact']
    for k in ('summary','details'):assert run.base.artifact(ROOT/er[k]['path'])==er[k]
    s=json.loads((run.PUBLIC/'summary.json').read_text());raw=json.loads((run.PRIVATE/'details.json').read_text());reductions=0
    for section,key in [('summary','rows'),('paired','contrasts'),('quality','quality')]:
        for policy,metrics in s[section].items():
            rr=[r for r in raw[key] if r['policy']==policy]
            for k,v in metrics.items():reductions+=reduce_check(v,rr,k,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in s['by_seed'].items():
        for policy,metrics in policies.items():
            rr=[r for r in raw['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for k,v in metrics.items():reductions+=reduce_check(v,rr,k,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    old=json.loads((run.previous.PUBLIC/'summary.json').read_text())
    for new,parent in [('floor','floor'),('parent_independent','independent'),('parent_joint','joint_utility'),('utility_topk_parent','utility_topk')]:
        assert s['summary'][new]==old['summary'][parent]
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    report=subprocess.run([sys.executable,'scripts/report_m3w_query_excess_refit.py'],cwd=ROOT,capture_output=True,text=True)
    if report.returncode:raise RuntimeError(report.stderr)
    assert all(run.base.digest(run.PUBLIC/p)==h for p,h in artifacts.items())
    parent=json.loads((run.previous.PUBLIC/'verification.json').read_text());bindings=dict(parent['source_bindings']);bindings.update(identity['bindings'])
    bindings.update({p:run.base.digest(ROOT/p) for p in ['scripts/report_m3w_query_excess_refit.py','scripts/verify_m3w_query_excess_refit.py',*TESTS]})
    run.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        test_log=run.base.artifact(log),tests=tests,test_files=len(TESTS),groups=108,heads=216,updates=432000,
        matched_fitting_queries=matched_fit_queries,independent_query_checks=queries,independent_locality_reductions=reductions,
        first_paired_fit_replay_exact=True,all_prediction_action_replays_exact=True,evaluation_replay_exact=True,
        parent_controls_exact=True,report_figure_byte_reproducible=True,full_legacy_suite='not_run',cold_raw_rebuild=False,
        independent_confirmation=False,calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(tests=tests,queries=queries,fit_queries=matched_fit_queries,reductions=reductions)))


if __name__=='__main__':main()
