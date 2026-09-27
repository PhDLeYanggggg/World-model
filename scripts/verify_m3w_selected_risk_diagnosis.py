"""Independent arithmetic checks of the frozen-action diagnostic."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_m3w_selected_risk as run
TESTS = ['tests/test_m3w_selected_risk_diagnosis.py', 'tests/test_m3w_subset_excess.py',
         'tests/test_m3w_fixed_floor_probe.py', 'tests/test_m3w_query_excess_verification.py']


def check_residual(report, p, truth, mask, bank, outside, size):
    selected = np.flatnonzero(mask).tolist()
    known = [i for i in selected if np.isfinite(truth[i]).all()]
    assert (report['rows'], report['known'], report['unknown']) == (len(selected), len(known), len(selected)-len(known))
    checked = 3
    for axis, j in [('all', 0), ('easy', 1)]:
        diff = [float(truth[i, j])-float(p[i, j]) for i in known]
        expected = {axis+'_optimism_mean':sum(diff)/len(diff) if diff else None,
            axis+'_MSE':sum(x*x for x in diff)/len(diff) if diff else None,
            axis+'_actual_positive_fraction':sum(truth[i,j]>0 for i in known)/len(known) if known else None,
            axis+'_predicted_excess_sum':sum(float(p[i,j]) for i in known),
            axis+'_actual_excess_sum':sum(float(truth[i,j]) for i in known)}
        for key, value in expected.items():
            if value is None: assert report[key] is None
            else: np.testing.assert_allclose(value, report[key], rtol=1e-10, atol=1e-10)
            checked += 1
    for name, values in [('controller_proxy', bank[:,0]), ('low_proxy', bank[:,1]),
            ('high_proxy', bank[:,2]), ('outside_both_fit_sources', outside), ('singleton_query', size==1)]:
        for suffix, positions in [('_fraction', selected), ('_known_fraction', known)]:
            actual = report[name+suffix]
            if not positions: assert actual is None
            else: np.testing.assert_allclose(actual, sum(bool(values[i]) for i in positions)/len(positions))
            checked += 1
    return checked


def check_reduction(report, rows, key, bootstrap):
    vals = []; missing = 0
    for site in sorted(report['by_locality']):
        block = [r['metric'][key] for r in rows if r['site']==site]
        valid = [float(x) for x in block if x is not None and np.isfinite(x)]
        missing += len(block)-len(valid)
        if not valid: assert report['by_locality'][site] is None
        else:
            value = sum(valid)/len(valid); vals.append(value)
            np.testing.assert_allclose(value, report['by_locality'][site], rtol=1e-12, atol=1e-10)
    assert report['undefined_views']==missing and report['observed_localities']==len(vals)
    if vals: np.testing.assert_allclose(report['mean'], sum(vals)/len(vals), rtol=1e-12, atol=1e-10)
    else: assert report['mean'] is None
    if bootstrap and not missing and len(vals)==len(report['by_locality']):
        rng=np.random.default_rng(101531)
        boot=np.array(vals)[rng.integers(0,len(vals),(3000,len(vals)))].mean(1)
        np.testing.assert_allclose(report['ci95'],np.percentile(boot,[2.5,97.5]),rtol=1e-12,atol=1e-10)
    else: assert report['ci95'] is None
    return len(report['by_locality'])+3


def main():
    cfg, ident = run.identity()
    completion = json.loads((run.PUBLIC/'completion.json').read_text())
    replay = json.loads((run.PUBLIC/'replay.json').read_text())
    assert completion['identity']==replay['identity']==ident and replay['exact_replay']
    assert completion['groups']==replay['groups'] and completion['summary']==replay['summary']
    records = {}
    for r in completion['groups']:
        assert run.base.artifact(ROOT/r['path']) == r
        records[Path(r['path']).stem] = json.loads((ROOT/r['path']).read_text())
    run.base.torch.set_num_threads(4); run.base.torch.set_num_interop_threads(1)
    _, data, jobs, oid, _, _, _, _, _, pid = run.parent.load()
    causal={k:data[k] for k in run.parent.CAUSAL_KEYS}
    fields=exchanges=groups=nearest=0
    for c in run.base.floor_api.contexts(causal,jobs,oid):
        ids=c['ids'];cv,_,(floor,_),(neural,_)=run.base.floor_api.costs(c,data,np.arange(len(ids)))
        bank=run.parent.masks(c,causal)
        sizes=run.api.query_sizes(data['sites'][ids],data['recordings'][ids],data['frames'][ids])
        for pair in range(6):
            name=c['name']+f'_pair{pair}';record=records[name]
            ref=record['parent_action'];assert run.base.artifact(ROOT/ref['path'])==ref
            doc=json.loads((ROOT/ref['path']).read_text())
            held,p,pr=run.parent.predictions(c,causal,doc['fit'],pid)
            with np.load(ROOT/doc['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            fitdoc=json.loads((ROOT/doc['fit']['path']).read_text());roles=fitdoc['identity']['roles_and_targets']
            fit=np.flatnonzero(np.isin(data['sites'][ids],roles['training_sites']))
            state=run.parent.api.head.read_checkpoint(ROOT/fitdoc['artifacts']['subset_pointwise']['path'])
            u=run.parent.api.head.descriptors(data['geometry'][ids],c['floor'],c['prediction'],c['x'],pr)
            upr=state['descriptor_preprocess'];sup=run.api.support_from_fit(u[fit],data['sites'][ids[fit]],pr['known'],u[held],upr['mean'],upr['std'])
            # Brute-force distances independently check a deterministic causal ID sample.
            sample=np.linspace(0,len(held)-1,min(16,len(held)),dtype=int)
            for j,site in enumerate(sup['fitting_sites']):
                train=(u[fit][pr['known'] & (data['sites'][ids[fit]]==site)]-upr['mean'])/upr['std']
                test=(u[held[sample]]-upr['mean'])/upr['std']
                for k,row in zip(sample,test):
                    distance=np.sqrt(np.sum((train-row)**2,axis=1)).min()
                    np.testing.assert_allclose(distance,sup['distance'][k,j],rtol=1e-12,atol=1e-12);nearest+=1
            truth=np.full((len(ids),2),np.nan)
            known=np.isfinite(cv);easy=known & (cv>0) & (cv<=c['job']['design']['easy_cut'])
            truth[known,0]=(np.maximum(neural[known]-floor[known],0)-.02*floor[known])/pr['cost_scale']
            truth[known,1]=np.where(easy[known],truth[known,0],0)
            for r in record['rows']:
                at=data['sites'][a['ids']]==r['site'];ix=held[at];b=bank[ix];outside=sup['outside_both'][at]
                eligible=a['eligible'][at];arm=r['arm'];key=r['slice']
                if key=='eligible':mask=eligible
                elif key in run.parent.api.SUBSETS:mask=b[:,run.parent.api.SUBSETS.index(key)]
                else:
                    policy=key.split('_')[0];take=a[arm+'_'+policy][at]
                    mask={'selected':take,'unselected_eligible':eligible & ~take,
                        'selected_in_proxy':take & b[:,0], 'selected_out_proxy':take & ~b[:,0],
                        'selected_in_support':take & ~outside,'selected_out_support':take & outside}[key[len(policy)+1:]]
                pred=p[arm][at].astype(float)
                q=np.stack([pred[:,1]-.02*pred[:,0],pred[:,3]-.02*pred[:,2]],axis=1)/pr['cost_scale']
                fields+=check_residual(r['metric'],q,truth[ix],mask,b,outside,sizes[ix])
            for r in record['exchanges']:
                at=data['sites'][a['ids']]==r['site'];ix=held[at];m=r['metric']
                before=a['subset_pointwise_'+r['policy']][at];after=a['subset_aggregate_'+r['policy']][at]
                for key,predicate in [('added',lambda i:after[i] and not before[i]),
                        ('removed',lambda i:before[i] and not after[i]),('shared',lambda i:before[i] and after[i])]:
                    use=[i for i in range(len(ix)) if predicate(i) and np.isfinite(floor[ix[i]])]
                    np.testing.assert_allclose(m[key+'_benefit'],sum(max(float(floor[ix[i]]-neural[ix[i]]),0) for i in use),rtol=1e-10,atol=1e-8)
                    np.testing.assert_allclose(m[key+'_harm'],sum(max(float(neural[ix[i]]-floor[ix[i]]),0) for i in use),rtol=1e-10,atol=1e-8)
                exchanges+=1
            groups+=1
            if groups%18==0:print(json.dumps(dict(groups=groups,residual_fields=fields,exchange_views=exchanges)),flush=True)
    d=json.loads((run.PUBLIC/'summary.json').read_text());reductions=0
    sections=[('residuals','rows',('arm','slice'),False),('query_residuals','queries',('arm','policy'),False),
              ('fitting_in_sample','fitting',('arm',),False),('exchanges','exchanges',('policy',),True)]
    for section,source,keys,boot in sections:
        allrows=[r for x in records.values() for r in x[source]]
        for key,metrics in d[section].items():
            rr=[r for r in allrows if '/'.join(str(r[k]) for k in keys)==key]
            for m,v in metrics.items():reductions+=check_reduction(v,rr,m,boot)
    contrasts=[]
    for doc in records.values():
        lookup={(r['site'],r['arm'],r['slice']):r for r in doc['rows']}
        for r in doc['rows']:
            if r['slice'] not in ('joint_selected','rank_selected'):continue
            policy=r['slice'].split('_')[0];other=lookup[(r['site'],r['arm'],policy+'_unselected_eligible')]
            metrics={}
            for axis in ('all','easy'):
                a=r['metric'][axis+'_optimism_mean'];b=other['metric'][axis+'_optimism_mean']
                metrics[axis+'_optimism_mean_selected_minus_unselected']=None if a is None or b is None else a-b
            contrasts.append(dict(site=r['site'],arm=r['arm'],policy=policy,metric=metrics))
    for key,metrics in d['selected_residual_contrasts'].items():
        rr=[r for r in contrasts if r['arm']+'/'+r['policy']==key]
        for m,v in metrics.items():reductions+=check_reduction(v,rr,m,True)
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    parent=json.loads((run.parent.PUBLIC/'verification.json').read_text());bindings=dict(parent['source_bindings'])
    bindings.update(ident['bindings'])
    bindings.update({p:run.base.digest(ROOT/p) for p in ['scripts/verify_m3w_selected_risk_diagnosis.py','scripts/report_m3w_selected_risk_diagnosis.py',*TESTS]})
    artifacts={p.name:run.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.json','.md','.png') and p.name!='verification.json'}
    proc=subprocess.run([sys.executable,'scripts/report_m3w_selected_risk_diagnosis.py'],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    assert all(run.base.digest(run.PUBLIC/p)==h for p,h in artifacts.items())
    run.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        groups=groups,independent_residual_fields=fields,independent_exchange_views=exchanges,
        brute_force_support_distances=nearest,independent_locality_checks=reductions,tests=tests,
        test_files=len(TESTS),test_log=run.base.artifact(log),full_diagnostic_replay_exact=True,
        report_byte_reproducible=True,full_legacy_suite='not_run',cold_raw_rebuild='not_run',
        new_training=False,new_policy=False,independent_confirmation=False,uncertainty_calibrated=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,groups=groups,fields=fields,exchanges=exchanges,nearest=nearest,reductions=reductions,tests=tests)))


if __name__=='__main__':main()
