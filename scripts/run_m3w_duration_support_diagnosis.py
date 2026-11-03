"""Decompose frozen fourfold-head fitting harm by label support, without updates."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_feature_duration_probe as duration
from src.evaluation.m3w_duration_support import support_partitions
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

parent = duration.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_duration_support_diagnosis_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_duration_support_diagnosis_v1'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_duration_support.py',
         ROOT/'tests/test_m3w_duration_support.py', PUBLIC/'protocol.md')


def admission():
    values, _ = duration.admission()
    replay = json.loads((duration.PUBLIC/'verification.json').read_text())
    if (not replay['all18_fitting_decompositions_exact']
            or replay['summary_sha256'] != parent.sha(duration.PUBLIC/'summary.json')):
        raise ValueError('Require completed fourfold fitting replay')
    definition = dict(bindings={str(p.relative_to(ROOT)):parent.sha(p) for p in FILES},
        duration_summary=parent.artifact(duration.PUBLIC/'summary.json'),
        duration_replay=parent.artifact(duration.PUBLIC/'verification.json'),
        jobs=values[1], models=18, roles=['critic_fit'],
        sets=['original','extended_fixed_original_selection','extended_own_selection'],
        partitions=['complete','partial','unknown'], optimizer_updates=0,
        post_result_descriptive=True, held_policy_evaluation=False,
        independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    parent.once(PUBLIC/'registration.json', definition)
    return values


def describe(records):
    def quantiles(values):
        a = [v for v in values if v is not None]
        return dict(defined_views=len(a), quantiles_0_25_50_75_100=
                    np.quantile(a, [0,.25,.5,.75,1]).tolist() if a else None)
    def bad(row, scope):
        value = row[scope]['target_easy']
        return value is not None and value > .02
    result = dict(record_views=len(records),
        selected_rows=sum(r['rows'] for r in records),
        whole_easy_violations=sum(bad(r,'whole_risks') for r in records),
        observed_easy_defined=sum(r['observed_only_risks']['target_easy'] is not None for r in records),
        observed_easy_violations=sum(bad(r,'observed_only_risks') for r in records),
        whole_and_observed_easy_violations=sum(bad(r,'whole_risks') and bad(r,'observed_only_risks') for r in records),
        whole_violation_observed_safe=sum(bad(r,'whole_risks') and r['observed_only_risks']['target_easy'] is not None
                                          and not bad(r,'observed_only_risks') for r in records),
        whole_violation_observed_undefined=sum(bad(r,'whole_risks') and r['observed_only_risks']['target_easy'] is None for r in records),
        unknown_share_of_target_EH=quantiles([r['unknown_share_of_target_EH'] for r in records]),
        parts={})
    for part in ('complete','partial','unknown'):
        values = [r['parts'][part] for r in records]
        result['parts'][part] = dict(selected_rows=sum(v['rows'] for v in values),
            nonempty_views=sum(v['rows'] > 0 for v in values),
            EH_prediction_target_ratio=quantiles([v['masses']['EH']['predicted_over_target'] for v in values]),
            EH_target_mass_sum_over_repeated_views=sum(v['masses']['EH']['target_mass'] for v in values),
            EH_predicted_mass_sum_over_repeated_views=sum(v['masses']['EH']['predicted_mass'] for v in values))
    return result


def one(job, src, ids, refs, caches, models):
    data = parent.bank(job,src,ids,refs,caches,training=True)
    frozen = json.loads((duration.PUBLIC/'fits'/(job['key']+'.json')).read_text())
    old = parent.api.base.base.read_checkpoint(parent.checked(models[job['key'],'interaction64']['checkpoint']))
    state = parent.api.base.base.read_checkpoint(parent.checked(frozen['checkpoint']))
    np.testing.assert_array_equal(data['ids'], old['fit_ids'])
    np.testing.assert_array_equal(data['ids'], state['fit_ids'])
    old_pred = parent.api.predict(old,data['causal'])
    pred = parent.api.predict(state,data['causal'])
    # Future arrays are accessed only after both inference calls have returned.
    rows, paths, c = data['ids'], data['paths'], data['causal']
    y = np.load(parent.parent.prior.SOURCE/'target_eval.npy',mmap_mode='r',allow_pickle=False)[rows]-src['origin'][rows,None]
    valid = np.load(parent.parent.prior.SOURCE/'valid.npy',mmap_mode='r',allow_pickle=False)[rows]
    a_old = parent.api.base.amplitudes(old_pred,c['allowed'])
    a_new = parent.api.base.amplitudes(pred,c['allowed'])
    result = {}
    for label, p, alpha in (('original',old_pred[:,6],a_old),
                            ('extended_fixed_original_selection',pred[:,6],a_old),
                            ('extended_own_selection',pred[:,6],a_new)):
        mixed = parent.api.base.prediction(paths[:,job['floor_index']],paths[:,6],alpha)
        selected = np.linalg.norm(mixed-paths[:,job['floor_index']],axis=-1).max(1)>0
        bank = paths.copy(); bank[:,6] = mixed
        target = supported_bound_targets(bank,y,valid,job['floor_index'],job['easy_cut'])['moments'][:,6]
        scaled = p.copy(); scaled[:,[0,1,4]] *= alpha[:,None]
        values = {}
        for rec in sorted(set(data['recordings'])):
            take = selected & (data['recordings']==rec)
            value = support_partitions(scaled,target,take,valid)
            previous = frozen['sets'][label][str(rec)]
            if value['rows'] != previous['selected'] or value['whole'] != previous['selected_actual_interpolation']:
                raise ValueError('Every frozen selected row and unpartitioned mass must match exactly')
            values[str(rec)] = value
        result[label] = values
    return dict(job=job['key'],seed=job['seed'],role='critic_fit',sets=result,
                frozen_unpartitioned_masses_exact=True,optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run','verify'))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,jobs,src,ids,refs,caches = admission()
        if args.phase == 'register':
            return
        for path in (*FILES,PUBLIC/'registration.json'):
            blob = subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,
                                  capture_output=True,check=True).stdout
            if parent.sha(path) != hashlib.sha256(blob).hexdigest():
                raise ValueError('Commit the diagnosis definition before execution')
        end = PUBLIC/('verification.json' if args.phase=='verify' else 'complete.json')
        if end.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase=='verify' and not (PUBLIC/'complete.json').exists():
            raise ValueError('Require completed first execution')
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        start = time.monotonic()
        def beat(**kw):
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
                raise OSError('Keep the unchanged10GiB disk reserve')
            if time.monotonic()-start > 43200:
                raise TimeoutError('Resume unchanged work after12h')
            event = dict(pid=os.getpid(),elapsed_seconds=time.monotonic()-start,**kw)
            parent.write(PRIVATE/'heartbeat.json',event)
            print(json.dumps(event),flush=True)
        models = parent.fitted()
        records = []
        beat(state='runtime_ready',architecture=platform.machine(),threads=4,workers=0)
        for job in jobs:
            path = PUBLIC/'groups'/(job['key']+'.json')
            if args.phase=='run' and path.exists():
                if not args.resume:
                    raise ValueError('Explicit unchanged resume required')
                value = json.loads(path.read_text())
            else:
                value = one(job,src,ids,refs,caches,models)
                parent.once(path,value)
            records.append(value)
            beat(state='support_partition_verified' if args.phase=='verify' else 'support_partition_complete',
                 groups=len(records),total=18)
            gc.collect()
        summary = dict(result_source='fresh_frozen_fitting_label_support_diagnosis',models=18,
            results={label:describe([r for v in records for r in v['sets'][label].values()])
                     for label in records[0]['sets']},
            optimizer_updates=0,held_policy_evaluation=False,independent_confirmation=False,
            deployment_changed=False,stage5c_executed=False,smc_enabled=False)
        parent.once(PUBLIC/'summary.json',summary)
        parent.once(end,dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summary_sha256=parent.sha(PUBLIC/'summary.json'),
            registration_sha256=parent.sha(PUBLIC/'registration.json'),
            all18_partitions_replayed_exact=args.phase=='verify'))
        print(json.dumps(dict(complete=str(end),groups=18)),flush=True)


if __name__=='__main__':
    main()
