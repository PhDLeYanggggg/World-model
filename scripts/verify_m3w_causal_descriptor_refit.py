"""Verify provenance, matched training, same-query ranks and independent reductions."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_causal_descriptor_refit as run
from scripts.verify_m3w_fixed_floor_tail import independently_match, reduce_check
TESTS = ['tests/test_m3w_causal_descriptor_head.py','tests/test_m3w_causal_descriptor_protocol.py',
    'tests/test_m3w_fixed_floor_excess.py','tests/test_m3w_fixed_floor_excess_protocol.py',
    'tests/test_m3w_fixed_floor_probe.py','tests/test_m3w_fixed_floor_tail.py',
    'tests/test_m3w_fixed_floor_slices.py','tests/test_m3w_fixed_floor_slices_protocol.py']


def main():
    cfg, data, _, _, _, _, _, identity = run.load()
    freeze = json.loads((run.PUBLIC/'decision_freeze.json').read_text()); assert freeze['identity']==identity
    queries = 0; cached_controls = 0
    for ref in freeze['heads']:
        assert run.base.artifact(ROOT/ref['path'])==ref
        d = json.loads((ROOT/ref['path']).read_text()); ident = d['identity']; roles = ident['roles_and_targets']
        run.base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
        assert ident['experiment']==identity and d['fit']['complete'] and d['fit']['step']==2000
        assert d['fit']['unknown_rows_sampled']==0 and not d['held_labels_read']
        for a in d['artifacts'].values(): assert run.base.artifact(ROOT/a['path'])==a
        assert run.base.artifact(ROOT/ident['control']['path'])==ident['control']
        old = json.loads((ROOT/ident['control']['path']).read_text())
        for a in old['artifacts'].values(): assert run.base.artifact(ROOT/a['path'])==a
        state = run.api.read_checkpoint(ROOT/d['artifacts']['checkpoint']['path'])
        control = run.base.torch.load(ROOT/old['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
        run.api.assert_matched(state,control); cached_controls += 1
        with np.load(ROOT/d['artifacts']['decisions']['path'],allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
        with np.load(ROOT/old['artifacts']['scores']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],a['ids']); p=z['scores'].astype(float)
        ids=a['ids']; q=np.maximum(p[:,1]-.02*p[:,0],p[:,3]-.02*p[:,2])
        keys=[str(s)+'|'+str(rec) for s,rec in zip(data['sites'][ids],data['recordings'][ids])]
        matched,n=independently_match(a['descriptor'],a['eligible'],q,keys,data['frames'][ids],ids)
        np.testing.assert_array_equal(matched,a['control_matched_count']); queries += n
        with np.load(run.base.PRIVATE/'decisions'/(roles['group']+'.npz'),allow_pickle=False) as z:
            for new,oldkey in [('ids','ids'),('eligible','eligible'),('mse','mse'),('ridge','ridge'),('control','excess')]:
                np.testing.assert_array_equal(a[new],z[oldkey])
    for file,key in [('prediction_replay.json','exact'),('training_replay.json','exact'),('causal_replay.json','future_fields_removed'),('evaluation_replay.json','exact')]:
        assert json.loads((run.PUBLIC/file).read_text())[key]
    for file,keys in [('training_replay.json',('reference','replay')),('evaluation_replay.json',('summary','details'))]:
        d=json.loads((run.PUBLIC/file).read_text())
        for k in keys: assert run.base.artifact(ROOT/d[k]['path'])==d[k]
    summary=json.loads((run.PUBLIC/'summary.json').read_text()); details=json.loads((run.PRIVATE/'details.json').read_text()); count=0
    for section,rowskey in [('summary','rows'),('quality','quality'),('paired','contrasts')]:
        for policy,metrics in summary[section].items():
            rr=[r for r in details[rowskey] if r['policy']==policy]
            for key,value in metrics.items(): count += reduce_check(value,rr,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in summary['by_seed'].items():
        for policy,metrics in policies.items():
            rr=[r for r in details['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for key,value in metrics.items(): count += reduce_check(value,rr,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    log=run.PRIVATE/'scoped_pytest.txt'
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.base.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.json','.md','.png') and p.name!='verification.json'}
    proc=subprocess.run([sys.executable,'scripts/report_m3w_causal_descriptor_refit.py'],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(proc.stderr)
    assert artifacts=={p:run.base.digest(run.PUBLIC/p) for p in artifacts}
    seal=json.loads((run.diagnosis.PUBLIC/'verification.json').read_text()); bindings=dict(seal['source_bindings']); bindings.update(identity['bindings'])
    extras=['scripts/report_m3w_causal_descriptor_refit.py','scripts/replay_m3w_causal_descriptor_training.py',
        'scripts/verify_m3w_causal_descriptor_refit.py',*TESTS]
    bindings.update({p:run.base.digest(ROOT/p) for p in extras})
    run.base.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.base.artifact(log),matched_cached_controls=cached_controls,
        exact_full_prediction_replays=108,exact_full_fit_replays=1,real_future_removed_inference_replays=1,
        exact_evaluation_replay=True,independently_reconstructed_query_views=queries,
        independently_reduced_locality_metrics=count,reports_figure_byte_reproducible=True,
        full_legacy_suite='not_run',cold_raw_rebuild=False,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,tests=tests,query_views=queries,metric_reductions=count,source_bindings=len(bindings),artifacts=len(artifacts))))


if __name__=='__main__': main()
