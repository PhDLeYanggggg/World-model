"""Replay severity support, causal predictions, loss identity and cost metrics."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_severity_auxiliary as run
from scripts.report_m3w_european_severity_auxiliary import aggregates
from scripts.verify_m3w_european_harm_tail_crossfit import rank_metrics
import numpy as np
import torch


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'; binding=run.artifact(Path(__file__))
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['source_binding']==binding and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json'); return r
    assert set(inspect.signature(run.method.predict).parameters)=={'model','x','env','pr'}
    support=json.loads((run.PUBLIC/'support_report.json').read_text())
    supported={(r['group'],r['pair'],r['held']):r for r in support['rows']}
    records=[]; checks=0
    for g,data,pairs in run.base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            x,env,by,*_=run.base.pair_inputs(g,data,pairs,pair)
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            previous=json.loads((run.parent.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,_=run.tail.fold_inputs(x,env,by,cv,sites,held)
                easy=run.parent.parent.parent.method.labels(cv[tr],cut)
                target=run.tail.diagnostic.event_targets(by[te],cv[te],cut)
                tag=name+'_'+pair+'_'+held; directory=run.PRIVATE/'heads'/tag; model,s=run.method.restore(directory)
                r=json.loads((directory/'complete.json').read_text()); inp=r['input']
                for k,a in (('train_x_sha256',x[tr]),('held_x_sha256',x[te]),('train_y_sha256',y),
                    ('train_easy_sha256',easy),('train_ids_sha256',bi[tr]),('held_ids_sha256',bi[te])): assert inp[k]==run.array_hash(a)
                assert s['step']==2000 and r['fit']['parameters']==24901 and not inp['held_targets_used_for_fit']
                for k in ('mean','std','known','weights'): np.testing.assert_array_equal(s['preprocess'][k],pr[k])
                norm=np.where(pr['known'][:,None],y/pr['cost_scale'],0).astype(np.float32)
                assert s['mean_harm']==float(pr['weights']@norm[:,1].astype(float))
                sup=run.method.support(y,easy,pr,sites[tr],data['recordings'][bi[tr]],data['agents'][bi[tr]])
                assert sup==supported[name,pair,held]['support']
                tref,href,oldstate,_=run.parent.parent.reference(tag,x,tr,te,pr,env,bi)
                with np.load(run.parent.PRIVATE/'heads'/tag/'cost_only'/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); np.testing.assert_array_equal(z['scores'],href)
                for k in ('draws','fixed_ids','loss_scales'): np.testing.assert_array_equal(s[k],oldstate[k])
                assert torch.equal(s['sampler_rng'],oldstate['sampler_rng']) and s['draws'][~pr['known']].sum()==0
                _,ordinary=run.parent.method.restore(run.parent.PRIVATE/'heads'/tag/'membership_aux')
                for k in ('cost_loss','membership_BCE'): assert s['trace'][0][k]==ordinary['trace'][0][k]
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); pred=z['scores'].copy(); weighted_p=z['weighted_membership'].copy()
                n=min(4096,te.sum()); score,prob=run.method.predict(model,x[te][:n],env[te][:n],pr)
                np.testing.assert_array_equal(score[:,[1,3]],pred[:n][:,[1,3]]); np.testing.assert_array_equal(prob,weighted_p[:n])
                np.testing.assert_array_equal(pred[:,[0,2]],href[:,[0,2]])
                f=next(f for f in row['folds'] if f['held']==held); oldf=next(f for f in previous['folds'] if f['held']==held)
                assert f['target_sha256']==oldf['target_sha256']==run.array_hash(target)
                assert f['held_ids_sha256']==run.array_hash(bi[te]) and f['easy_cut']==cut
                for a in ('original_mean','cost_only','membership_aux'):
                    assert f['metrics'][a]==oldf['metrics'][a] and f['training'][a]==oldf['training'][a]
                fitting,_=run.method.predict(model,x[tr],env[tr],pr); fitting=run.parent.parent.compose(fitting[:,[1,3]],tref)
                edges=run.tail.training_edges(fitting,env[tr],pr,cfg)
                assert edges==f['training']['severity_aux']['edges']
                assert run.tail.diagnostic.summarize(fitting,y,env[tr],sites[tr],edges)==f['training']['severity_aux']['training']
                assert run.parent.parent.measure(pred,target,env[te],sites[te],edges)==f['metrics']['severity_aux']
                for subset,mask in (('all',np.ones(te.sum(),bool)),('envelope_positive',env[te]>0)):
                    known=np.isfinite(target).all(1)&mask
                    if not known.any(): continue
                    pp,tt=pred[known],target[known]; m=f['metrics']['severity_aux'][subset]
                    np.testing.assert_allclose(float(((pp[:,3]-tt[:,3])**2).mean()),m['harm_MSE'],rtol=1e-10,atol=1e-10); checks+=1
                    np.testing.assert_allclose(float(((pp[:,1]-tt[:,1])**2).mean()),m['component_MSE'][1],rtol=1e-10,atol=1e-10); checks+=1
                    au,ap=rank_metrics(pp[:,3],tt[:,3]>0)
                    for v,k in ((au,'AUROC'),(ap,'AUPRC')):
                        if v is None: assert m['scores']['moment'][k] is None
                        else: np.testing.assert_allclose(v,m['scores']['moment'][k],rtol=1e-10,atol=1e-10)
                        checks+=1
                records.append(dict(group=name,pair=pair,held=held,prefix_rows=int(n),checkpoint=run.artifact(directory/'checkpoint.pt')))
                run.beat('verified',completed=len(records),group=name,pair=pair,held=held)
    assert len(records)==144
    r=dict(identity=identity,source_binding=binding,training=run.artifact(run.PRIVATE/'training_complete.json'),
        records=records,independent_cost_rank_checks=checks,support_views_replayed=144,initialization_and_sampler_matched=True,all_passed=True)
    run.immutable_json(path,r); return r


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); run.base.previous.require_committed(run.PUBLIC/'prediction_freeze.json')
    run.parent.checked_training(identity['parent']['parent'])
    done=run.checked_training(identity); receipt=replay(cfg,identity)
    rows=[]
    for ref in json.loads((run.PUBLIC/'completion_checks.json').read_text())['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    a=aggregates(rows,cfg); assert a==json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    assert a['gates']==json.loads((run.PUBLIC/'gates.json').read_text())
    fits=[json.loads((ROOT/ref['path']).read_text()) for ref in done['heads']]
    comp=json.loads((run.PUBLIC/'compute_receipt.json').read_text())
    assert comp['heads']==len(fits)==144 and comp['updates']==sum(r['fit']['step'] for r in fits)==288000
    assert comp['summed_fit_seconds']==sum(r['fit']['seconds'] for r in fits)
    assert comp['unknown_draws']==sum(r['fit']['unknown_rows_sampled'] for r in fits)==0
    tests=['tests/test_m3w_severity_auxiliary.py','tests/test_m3w_severity_auxiliary_reporting.py',
        'tests/test_m3w_membership_auxiliary.py','tests/test_m3w_membership_auxiliary_reporting.py',
        'tests/test_m3w_task_gradients.py']
    xml=run.PRIVATE/'tests.xml'; p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout,flush=True); assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
    required=('conclusions.md','failure_analysis.md','project_gap.md','model_card.md','data_card.md','operation_zh.md',
              'literature_position.md','training_loss.svg','paired_contrasts.svg')
    assert all((run.PUBLIC/f).is_file() for f in required)
    figures={f:run.digest(run.PUBLIC/f) for f in ('training_loss.svg','paired_contrasts.svg')}
    subprocess.run([sys.executable,'scripts/plot_m3w_european_severity_auxiliary.py'],cwd=ROOT,check=True)
    assert figures=={f:run.digest(run.PUBLIC/f) for f in figures}
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<2**20 for f in artifacts)
    bindings=sorted(set(run.FILES+tests+['scripts/plot_m3w_european_severity_auxiliary.py',str(Path(__file__).relative_to(ROOT))]))
    old=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},fresh_tests=count,test_files=tests,
        parent_tests_cached_verified=old['tests'],parent_test_files_cached_verified=len(old['test_files']),full_legacy_suite='not_run',
        checkpoint_prefix_replays=144,independent_cost_rank_checks=receipt['independent_cost_rank_checks'],
        support_replays=144,deterministic_figures=True,policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(fresh_tests=count,files=len(tests),public_artifacts=len(artifacts),source_bindings=len(bindings))))


if __name__=='__main__': main()
