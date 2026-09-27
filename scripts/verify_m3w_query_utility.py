"""Independent current-query counts, risk scores and aggregate readout checks."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_query_utility as run
from scripts.verify_m3w_fixed_floor_tail import reduce_check
TESTS=['tests/test_m3w_query_utility.py','tests/test_m3w_selection_exchange.py',
    'tests/test_m3w_causal_descriptor_head.py','tests/test_m3w_causal_descriptor_protocol.py',
    'tests/test_m3w_fixed_floor_probe.py']


def main():
    run.parent.base.torch.set_num_threads(4);run.parent.base.torch.set_num_interop_threads(1)
    cfg,data,jobs,oid,identity=run.load()
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text());assert freeze['identity']==identity
    refs={Path(r['path']).parent.name:r for r in freeze['groups']};queries=changed=checked_groups=0
    causal={k:data[k] for k in ('sites','history','origin','geometry','frames','recordings')}
    for c in run.parent.base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}';r=refs[name]
            assert run.parent.base.artifact(ROOT/r['path'])==r
            d=json.loads((ROOT/r['path']).read_text())
            assert d['future_fields_removed'] and not d['held_outcomes_used']
            assert run.parent.base.artifact(ROOT/d['parent_head']['path'])==d['parent_head']
            h=json.loads((ROOT/d['parent_head']['path']).read_text())
            for ref in [*d['artifacts'].values(),*h['artifacts'].values()]:assert run.parent.base.artifact(ROOT/ref['path'])==ref
            with np.load(ROOT/d['artifacts']['decisions']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            with np.load(ROOT/h['artifacts']['decisions']['path'],allow_pickle=False) as z:
                for key,old in [('ids','ids'),('eligible','eligible'),('independent','descriptor'),('control_matched_count','control_matched_count')]:
                    np.testing.assert_array_equal(a[key],z[old])
            ids=a['ids'];pos=np.searchsorted(c['ids'],ids);np.testing.assert_array_equal(c['ids'][pos],ids)
            s=run.parent.api.read_checkpoint(ROOT/h['artifacts']['checkpoint']['path']);pr=s['preprocess']
            u=run.parent.api.descriptors(causal['geometry'][ids],c['floor'][pos],c['prediction'][pos],c['x'][pos],pr)
            p=run.parent.api.predict(run.parent.model_from(s),c['x'][pos],u,c['env'][pos],pr,s['descriptor_preprocess'])
            assert run.parent.base.inter.array_hash(p)==h['prediction_sha256']
            q=np.column_stack((p[:,1].astype(float)-.02*p[:,0].astype(float),p[:,3].astype(float)-.02*p[:,2].astype(float)))/pr['cost_scale']
            with np.load(run.parent.base.floor_api.PRIVATE/'fits'/name/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids);utility=(z['scores'][:,5].astype(float)-z['scores'][:,6].astype(float))/pr['cost_scale']
            groups={}
            for i,rid in enumerate(ids):groups.setdefault((str(data['sites'][rid]),str(data['recordings'][rid]),int(data['frames'][rid])),[]).append(i)
            qc=0
            for positions in groups.values():
                at=np.array(positions);joint=a['joint_utility'][at];anchor=a['independent'][at]
                assert joint.sum()==anchor.sum()==a['utility_topk'][at].sum()==a['control_matched_count'][at].sum()
                assert not (joint & ~a['eligible'][at]).any()
                scale=np.maximum(np.max(np.abs(q[at]),axis=0),1e-12)
                assert np.all((q[at]/scale)[joint].sum(0)<=1e-10)
                assert utility[at][joint].sum()>=utility[at][anchor].sum()-1e-10
                qc+=bool((joint!=anchor).any())
                uniform=a['query_uniform'][at]
                assert uniform.all() or not uniform.any()
                if uniform.any():
                    assert a['eligible'][at].all() and np.all((q[at]/scale).sum(0)<=1e-10)
            assert len(groups)==d['queries']['queries'] and qc==d['queries']['changed_queries']
            queries+=len(groups);changed+=qc;checked_groups+=1
            if checked_groups%18==0:
                print(json.dumps(dict(state='independent_constraints_checked',groups=checked_groups,queries=queries)),flush=True)
    assert json.loads((run.PUBLIC/'decision_replay.json').read_text())['exact']
    ev=json.loads((run.PUBLIC/'evaluation_replay.json').read_text());assert ev['exact']
    for key in ('summary','details'):assert run.parent.base.artifact(ROOT/ev[key]['path'])==ev[key]
    summary=json.loads((run.PUBLIC/'summary.json').read_text());details=json.loads((run.PRIVATE/'details.json').read_text());reductions=0
    for section,rowskey in [('summary','rows'),('paired','contrasts')]:
        for policy,metrics in summary[section].items():
            rr=[r for r in details[rowskey] if r['policy']==policy]
            for key,value in metrics.items():reductions+=reduce_check(value,rr,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in summary['by_seed'].items():
        for policy,metrics in policies.items():
            rr=[r for r in details['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for key,value in metrics.items():reductions+=reduce_check(value,rr,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    parent=json.loads((run.parent.PUBLIC/'summary.json').read_text())
    for new,old in [('independent','descriptor'),('control_matched_count','control_matched_count'),('floor','floor')]:
        assert summary['summary'][new]==parent['summary'][old]
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.parent.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    proc=subprocess.run([sys.executable,'scripts/report_m3w_query_utility.py'],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    assert all(run.parent.base.digest(run.PUBLIC/p)==h for p,h in artifacts.items())
    seal=json.loads((run.source.PUBLIC/'verification.json').read_text());bindings=dict(seal['source_bindings']);bindings.update(identity['bindings'])
    extras=['scripts/report_m3w_query_utility.py','scripts/verify_m3w_query_utility.py',*TESTS]
    bindings.update({p:run.parent.base.digest(ROOT/p) for p in extras})
    run.parent.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.parent.base.artifact(log),groups=108,
        decisions_replayed_exact=True,future_removed_prediction_groups=108,evaluation_replay_exact=True,
        independent_query_constraint_checks=queries,changed_query_views=changed,independent_locality_reductions=reductions,
        parent_controls_exact=True,report_figure_byte_reproducible=True,full_legacy_suite='not_run',cold_raw_rebuild=False,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(tests=tests,queries=queries,changed=changed,reductions=reductions)))


if __name__=='__main__':main()
