"""Full-roster fourfold optimization-adequacy probe on fitting rows only."""
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
    raise RuntimeError('Use native arm64 before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_interaction_feature_cost as parent
from src.evaluation.m3w_cost_mass_decomposition import cost_masses, risks
from src.evaluation.m3w_duration_probe import extended_settings, require_exact_prefix
from src.evaluation.m3w_feature_cost_diagnosis import describe_records
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

NAME = 'european_feature_duration_probe_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
DIAG = ROOT/'outputs/publication_readiness_2026_09/european_feature_cost_diagnosis_v1'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_duration_probe.py',
         ROOT/'tests/test_m3w_duration_probe.py', PUBLIC/'protocol.md')
sha, checked, artifact, once = parent.sha, parent.checked, parent.artifact, parent.once


def admission():
    values = parent.admission()
    cfg, jobs, _, _, _, _ = values
    verification = json.loads((DIAG/'verification.json').read_text())
    if not verification['all36_job_role_summaries_exact'] or verification['summary_sha256'] != sha(DIAG/'summary.json'):
        raise ValueError('Complete current feature-head diagnosis required')
    updates = 4*sum(j['fit_scope']['budget'] for j in jobs)
    definition = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration=artifact(parent.REG), parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        diagnosis=artifact(DIAG/'summary.json'), diagnosis_replay=artifact(DIAG/'verification.json'),
        settings=extended_settings(cfg['training']), factor=4, jobs=jobs, arm='interaction64',
        new_models=18, planned_primary_optimizer_updates=updates,
        planned_first_full_refit_updates=4*jobs[0]['fit_scope']['budget'],
        old_budget_scientific_prefix_must_match=True, fitting_only=True,
        held_policy_evaluation=False, threshold_selection=False, new_trajectory_training=False,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    once(PUBLIC/'registration.json', definition)
    return values, definition


def fitting_scores(pred, old_pred, data, job, src):
    paths, c = data['paths'], data['causal']
    ids = data['ids']
    y = np.load(parent.parent.prior.SOURCE/'target_eval.npy', mmap_mode='r', allow_pickle=False)[ids]-src['origin'][ids, None]
    valid = np.load(parent.parent.prior.SOURCE/'valid.npy', mmap_mode='r', allow_pickle=False)[ids]
    truth = supported_bound_targets(paths, y, valid, job['floor_index'], job['easy_cut'])['moments'][:, 6]
    a_old = parent.api.base.amplitudes(old_pred, c['allowed'])
    a_new = parent.api.base.amplitudes(pred, c['allowed'])
    floor, proposal = paths[:, job['floor_index']], paths[:, 6]
    out = {}
    for label, p, alpha in (('original', old_pred[:, 6], a_old),
                            ('extended_fixed_original_selection', pred[:, 6], a_old),
                            ('extended_own_selection', pred[:, 6], a_new)):
        mixed = parent.api.base.prediction(floor, proposal, alpha)
        take_all = np.linalg.norm(mixed-floor, axis=-1).max(1) > 0
        bank = paths.copy()
        bank[:, 6] = mixed
        target = supported_bound_targets(bank, y, valid, job['floor_index'], job['easy_cut'])['moments'][:, 6]
        scaled = p.copy()
        scaled[:, [0, 1, 4]] *= alpha[:, None]
        records = {}
        for rec in sorted(set(data['recordings'])):
            at = data['recordings'] == rec
            take = at & take_all
            actual = cost_masses(scaled, target, take)
            records[str(rec)] = dict(selected=int(take.sum()), unknown_selected=int((take & ~valid.any(1)).sum()),
                full_legal_neural=cost_masses(p, truth, at & c['allowed'][:, 6]),
                selected_full_proposal=cost_masses(p, truth, take),
                selected_scaled_proposal=cost_masses(p, truth, take, alpha),
                selected_actual_interpolation=actual, selected_actual_risks=risks(actual))
        if label == 'original':
            saved = json.loads((DIAG/'groups'/(job['key']+'_critic_fit.json')).read_text())['arms']['interaction64']['records']
            for rec, row in records.items():
                for scope in ('full_legal_neural','selected_full_proposal','selected_scaled_proposal','selected_actual_interpolation','selected_actual_risks'):
                    if row[scope] != saved[rec][scope]:
                        raise ValueError('Original fitting decomposition must match exactly')
        out[label] = records
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register','train','verify'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (cfg, jobs, src, ids, refs, caches), reg = admission()
        if args.phase == 'register':
            return
        for path in (*FILES, PUBLIC/'registration.json'):
            blob = subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(path):
                raise ValueError('Commit duration definition before execution')
        end = PUBLIC/('verification.json' if args.phase == 'verify' else 'complete.json')
        if end.exists():
            raise ValueError('Completed phase immutable')
        if args.phase == 'verify' and not (PUBLIC/'complete.json').exists():
            raise ValueError('Complete original fitting probe first')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        start = time.monotonic()
        def guard():
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
                raise OSError('Keep existing10GiB reserve')
            if time.monotonic()-start > 43200:
                raise TimeoutError('12h phase boundary; resume unchanged endpoints')
        def beat(**kw):
            guard()
            event = dict(pid=os.getpid(),elapsed_seconds=time.monotonic()-start,**kw)
            parent.write(PRIVATE/'heartbeat.json',event)
            with (PRIVATE/'events.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        models = parent.fitted()
        records = []
        beat(state='runtime_ready',architecture=platform.machine(),threads=4,workers=0,fitting_only=True)
        for number, job in enumerate(jobs):
            data = parent.bank(job,src,ids,refs,caches,training=True)
            old = parent.api.base.base.read_checkpoint(checked(models[job['key'],'interaction64']['checkpoint']))
            path = PRIVATE/'fits'/(job['key']+'.pt.gz')
            fit = lambda dest, **kw:parent.api.fit(data['learning'],arm='interaction64',seed=job['seed'],
                settings=reg['settings'], identity=dict(registration_sha256=sha(PUBLIC/'registration.json'),job=job),
                path=dest,heartbeat=lambda **v:beat(job=job['key'],**v),guard=guard,**kw)
            prefix_file = PUBLIC/'prefixes'/(job['key']+'.json')
            if args.phase == 'train':
                if path.exists() and not args.resume:
                    raise ValueError('Explicit same-identity resume required')
                if not prefix_file.exists():
                    prefix = fit(path,stop_at=old['budget'],resume=path.exists())
                    require_exact_prefix(old,prefix)
                    once(prefix_file,dict(job=job['key'],old_budget=old['budget'],new_budget=prefix['budget'],scientific_prefix_exact=True))
                    del prefix
                state = fit(path,resume=True)
                if number == 0:
                    repeated = fit(PRIVATE/'first_full_refit.pt.gz',resume=args.resume and (PRIVATE/'first_full_refit.pt.gz').exists())
                    parent.parent.recovered(state,repeated)
                    once(PUBLIC/'first_full_refit.json',dict(updates=state['step'],scientific_state_exact=True))
                    del repeated
            else:
                state = parent.api.base.base.read_checkpoint(path)
                saved = json.loads((PUBLIC/'fits'/(job['key']+'.json')).read_text())
                if artifact(path) != saved['checkpoint']:
                    raise ValueError('Frozen extended checkpoint changed')
            if state['step'] != 4*old['budget']:
                raise ValueError('Every fourfold endpoint must finish')
            old_pred = parent.api.predict(old,data['causal'])
            pred = parent.api.predict(state,data['causal'])
            score = fitting_scores(pred,old_pred,data,job,src)
            objectives = parent.parent.objectives(pred,data['learning'],state['sampler']['counts'])
            original_objectives = parent.parent.objectives(old_pred,data['learning'],old['sampler']['counts'])
            value = dict(job=job['key'],seed=job['seed'],sites=job['critic_sites'],
                checkpoint=artifact(path),updates=state['step'],old_budget=old['budget'],seconds=state['seconds'],
                original_objectives=original_objectives,extended_objectives=objectives,
                sets=score,original_fitting_decomposition_exact=True,
                preprocessing_fit_G_only=True,held_predictions_not_run=True)
            output = PUBLIC/'fits'/(job['key']+'.json')
            if args.phase == 'verify' and not output.exists():
                raise ValueError('Missing completed fitting diagnostic')
            once(output,value)
            records.append(value)
            beat(state='extended_fit_verified' if args.phase=='verify' else 'extended_fit_complete',fits=len(records),total=18)
            del data,old,state,pred,old_pred
            gc.collect()
        summaries = {name:describe_records([r for v in records for r in v['sets'][name].values()]) for name in records[0]['sets']}
        summary = dict(result_source='fresh18real_Torch_fourfold_fitting_only_duration_probe',
            models=18,primary_optimizer_updates=sum(v['updates'] for v in records),
            first_full_refit_updates=reg['planned_first_full_refit_updates'],
            all18_old_budget_prefixes_exact=True,all_original_fitting_decompositions_exact=True,
            results=summaries,held_policy_evaluation=False,independent_confirmation=False,
            deployment_changed=False,stage5c_executed=False,smc_enabled=False)
        once(PUBLIC/'summary.json',summary)
        once(end,dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(PUBLIC/'registration.json'),
            all18_fitting_decompositions_exact=args.phase=='verify',held_policy_evaluation=False))
        print(json.dumps(dict(complete=str(end),updates=summary['primary_optimizer_updates'])),flush=True)


if __name__ == '__main__':
    main()
