"""Independently check retained counts, signed constraints and outcome accounting."""
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_centered_risk_policy as run
from scripts.verify_m3w_query_excess_refit import check_selected_costs
from scripts.verify_m3w_subset_excess import check_fixed_denominator
from scripts.verify_m3w_fixed_floor_tail import reduce_check
TESTS=['tests/test_m3w_centered_risk_policy.py','tests/test_m3w_centered_risk_verification.py',
       'tests/test_m3w_query_utility.py','tests/test_m3w_signed_bias_probe.py',
       'tests/test_m3w_signed_bias_verification.py','tests/test_m3w_query_excess_verification.py']
TESTS.append('tests/test_m3w_centered_identity_replay.py')


def validate_actions(a,q,delta,utility,sites,recordings,frames,arm):
    centered=q+np.asarray(delta);groups={};changed=0
    for i in range(len(a['ids'])):
        groups.setdefault((str(sites[i]),str(recordings[i]),int(frames[i])),[]).append(i)
    for positions in groups.values():
        ix=np.asarray(positions);eligible=a['eligible'][ix]
        anchor=eligible & (centered[ix]<=0).all(1)
        np.testing.assert_array_equal(a[arm+'_centered_independent'][ix],anchor)
        np.testing.assert_array_equal(a[arm+'_raw_independent'][ix],eligible & (q[ix]<=0).all(1))
        assert not (anchor & ~a[arm+'_raw_independent'][ix]).any()
        for suffix in ('centered_joint','matched_raw_joint','centered_utility_topk','centered_query_uniform'):
            take=a[arm+'_'+suffix][ix]
            assert take.dtype==bool and not (take & ~eligible).any()
            if suffix!='centered_query_uniform':assert int(take.sum())==int(anchor.sum())
            if suffix!='centered_utility_topk':
                risk=q[ix] if suffix=='matched_raw_joint' else centered[ix]
                scale=np.maximum(abs(risk).max(0),1e-12)
                assert all(sum(float(risk[j,k]/scale[k]) for j in np.flatnonzero(take))<=1e-10 for k in (0,1))
            if suffix in ('centered_joint','matched_raw_joint'):
                assert utility[ix][take].sum()+1e-10>=utility[ix][anchor].sum()
        changed+=not np.array_equal(a[arm+'_centered_joint'][ix],a[arm+'_matched_raw_joint'][ix])
    return len(groups),changed


def main():
    started=time.monotonic();run.beat('verification_started')
    run.base.torch.set_num_threads(4);run.base.torch.set_num_interop_threads(1)
    cfg,ident,data,jobs,oid,pid,parents,biases=run.load()
    causal={k:data[k] for k in run.parent.CAUSAL_KEYS}
    frozen=json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    refs={Path(r['path']).stem:r for r in frozen['groups']}
    details=json.loads((run.PRIVATE/'details.json').read_text())
    readout={(r['group'],r['site'],r['policy']):r['metric'] for r in details['rows']}
    exchange={(r['group'],r['site'],r['policy']):r['metric'] for r in details['exchanges']}
    queries=changed=views=groups=exchange_views=0;rejections=[]
    for c in run.base.floor_api.contexts(causal,jobs,oid):
        cv,_,(floor,_),(neural,_)=run.base.floor_api.costs(c,data,np.arange(len(c['ids'])))
        for pair in range(6):
            name=c['name']+f'_pair{pair}';r=refs[name];assert run.base.artifact(ROOT/r['path'])==r
            d=json.loads((ROOT/r['path']).read_text());assert d['future_fields_removed'] and not d['held_outcomes_used']
            assert d['parent_action']==parents[name] and d['bias_fit']==biases[name]
            for key in ('arrays','parent_action','bias_fit','utility_scores'):assert run.base.artifact(ROOT/d[key]['path'])==d[key]
            b=json.loads((ROOT/d['bias_fit']['path']).read_text());old=json.loads((ROOT/d['parent_action']['path']).read_text())
            assert not b['held_labels_used'] and not b['policy_changed'] and b['parent_fit']==old['fit']
            roles=b['roles'];run.base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
            with np.load(ROOT/d['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            held,p,pr=run.parent.predictions(c,causal,old['fit'],pid)
            np.testing.assert_array_equal(a['ids'],c['ids'][held])
            assert run.base.artifact(ROOT/old['arrays']['path'])==old['arrays']
            with np.load(ROOT/old['arrays']['path'],allow_pickle=False) as z:
                for key in ('ids','eligible'):np.testing.assert_array_equal(a[key],z[key])
                for arm in run.ARMS:
                    for suffix in ('independent','joint'):np.testing.assert_array_equal(a[arm+'_raw_'+suffix],z[arm+'_'+suffix])
            with np.load(ROOT/d['utility_scores']['path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(a['ids'],z['ids']);u=(z['scores'][:,5].astype(float)-z['scores'][:,6].astype(float))/pr['cost_scale']
            for arm in run.ARMS:
                v=p[arm].astype(float)
                q=np.column_stack((v[:,1]-.02*v[:,0],v[:,3]-.02*v[:,2]))/pr['cost_scale']
                n,chg=validate_actions(a,q,b['fits'][arm]['nonnegative_offset'],u,data['sites'][a['ids']],data['recordings'][a['ids']],data['frames'][a['ids']],arm)
                queries+=n;changed+=chg
                raw=a[arm+'_raw_independent'];risk=q+np.asarray(b['fits'][arm]['nonnegative_offset'])
                fail_all=risk[:,0]>0;fail_easy=risk[:,1]>0
                rejections.append(dict(group=name,arm=arm,raw_admitted=int(raw.sum()),
                    centered_retained=int((raw & ~fail_all & ~fail_easy).sum()),
                    rejected_all_only=int((raw & fail_all & ~fail_easy).sum()),
                    rejected_easy_only=int((raw & ~fail_all & fail_easy).sum()),
                    rejected_both=int((raw & fail_all & fail_easy).sum())))
            for site in roles['held_sites']:
                at=data['sites'][a['ids']]==site;ix=held[at]
                for policy in run.POLICIES:
                    take=np.zeros(len(ix),bool) if policy=='floor' else np.ones(len(ix),bool) if policy=='raw_neural' else a[policy][at]
                    m=readout[(name,site,policy)]
                    check_selected_costs(m,cv[ix],floor[ix],neural[ix],take)
                    check_fixed_denominator(m,floor[ix],neural[ix],np.isfinite(cv[ix]),take);views+=1
                for new,old in run.CONTRASTS:
                    m=exchange[(name,site,new+'_vs_'+old)];before=a[old][at];after=a[new][at]
                    f,n=floor[ix],neural[ix];known=np.isfinite(f);den=sum(float(v) for v in f[known])
                    for key,mask in [('added',after & ~before),('removed',before & ~after),('shared',before & after)]:
                        selected=[j for j in range(len(mask)) if mask[j] and known[j]]
                        assert m[key+'_count']==sum(bool(v) for v in mask)
                        assert m[key+'_unknown']==sum(bool(mask[j] and not known[j]) for j in range(len(mask)))
                        for cost,sign in [('benefit',1),('harm',-1)]:
                            value=sum(max(sign*(float(f[j])-float(n[j])),0.) for j in selected)
                            np.testing.assert_allclose(m[key+'_'+cost],value,rtol=1e-10,atol=1e-9)
                            np.testing.assert_allclose(m[key+'_'+cost+'_over_floor_pp'],100*value/den,rtol=1e-10,atol=1e-9)
                    exchange_views+=1
            groups+=1
            if groups%18==0:run.beat('independent_verified',groups=groups,query_arm_checks=queries)
    for row in details['contrasts']:
        new,old=row['policy'].split('_vs_');x=readout[(row['group'],row['site'],new)];y=readout[(row['group'],row['site'],old)]
        np.testing.assert_allclose(row['metric']['ADE_gain_percent'],100*(y['error_sum']-x['error_sum'])/y['error_sum'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(row['metric']['all_reference_harm_reduction_pp'],100*(y['positive_harm_sum']-x['positive_harm_sum'])/y['floor_error_sum'],rtol=1e-10,atol=1e-12)
    s=json.loads((run.PUBLIC/'summary.json').read_text());reductions=0
    for section,key in [('summary','rows'),('paired','contrasts'),('exchanges','exchanges')]:
        for policy,metrics in s[section].items():
            rows=[r for r in details[key] if r['policy']==policy]
            for k,v in metrics.items():reductions+=reduce_check(v,rows,k,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in s['by_seed'].items():
        for policy,metrics in policies.items():
            rows=[r for r in details['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for k,v in metrics.items():reductions+=reduce_check(v,rows,k,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for key in ('prediction_replay','evaluation_replay'):
        d=json.loads((run.PUBLIC/(key+'.json')).read_text());assert d['exact']
    for key in ('summary','details'):assert run.base.artifact(ROOT/d[key]['path'])==d[key]
    assert groups==108 and views==3456
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    reject_summary={arm:{k:sum(r[k] for r in rejections if r['arm']==arm) for k in ('raw_admitted','centered_retained','rejected_all_only','rejected_easy_only','rejected_both')} for arm in run.ARMS}
    for m in reject_summary.values():assert m['raw_admitted']==sum(v for k,v in m.items() if k!='raw_admitted')
    run.base.immutable_json(run.PUBLIC/'admission_diagnosis.json',dict(result_source='posthoc_causal_score_accounting_no_policy_change',
        counts_are_repeated_context_rows_not_independent_samples=True,summary=reject_summary,groups=rejections))
    report=subprocess.run([sys.executable,'scripts/report_m3w_centered_risk_policy.py'],cwd=ROOT,capture_output=True,text=True)
    if report.returncode:raise RuntimeError(report.stderr)
    runtime=run.PUBLIC/'verification_runtime.json'
    if not runtime.exists():run.base.immutable_json(runtime,dict(pid=os.getpid(),seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=groups,scope='first independent verification'))
    artifacts={p.name:run.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    report=subprocess.run([sys.executable,'scripts/report_m3w_centered_risk_policy.py'],cwd=ROOT,capture_output=True,text=True)
    if report.returncode:raise RuntimeError(report.stderr)
    assert all(run.base.digest(run.PUBLIC/p)==h for p,h in artifacts.items())
    parent=json.loads((run.bias.PUBLIC/'verification.json').read_text());bindings=dict(parent['source_bindings']);bindings.update(ident['bindings'])
    bindings.update({p:run.base.digest(ROOT/p) for p in ['scripts/verify_m3w_centered_risk_policy.py','scripts/report_m3w_centered_risk_policy.py','scripts/replay_m3w_centered_risk_policy.py',*TESTS]})
    run.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.base.artifact(log),groups=108,query_arm_checks=queries,
        changed_same_count_queries=changed,independent_cost_views=views,independent_exchange_views=exchange_views,independent_locality_checks=reductions,
        all_action_replay_exact=True,evaluation_replay_exact=True,raw_controls_unchanged=True,
        reports_byte_reproducible=True,new_neural_training=False,new_parameter_fitting=False,
        full_legacy_suite='not_run',cold_raw_rebuild='not_run',independent_confirmation=False,
        calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,tests=tests,groups=groups,queries=queries,changed_queries=changed,views=views,reductions=reductions)))
    run.beat('verification_complete',groups=groups,tests=tests)


if __name__=='__main__':main()
