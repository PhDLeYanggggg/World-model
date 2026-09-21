"""Replay fixed cost heads on their verified OOF fit rows, without training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = 'configs/m3w_cost_head_fit_forensics_v1.json'


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('transformer', 'eqmotion'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Native arm64 environment required before importing Torch')
    cfg = json.loads((ROOT/CONFIG).read_text())
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(cfg['threads'])
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import numpy as np
    import torch
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.evaluation.m3w_frozen_runtime import prepare_runtime, RelocatedCodeContract
    from src.evaluation.m3w_cost_fit_forensics import diagnose_fit
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    parent = json.loads((ROOT/cfg['parent_config']).read_text())
    if file_digest(ROOT/cfg['parent_risk_analysis']) != cfg['parent_risk_sha256']:
        raise ValueError('Parent risk report changed')
    if file_digest(ROOT/parent['protocol']) != parent['protocol_file_sha256']:
        raise ValueError('Original protocol changed')
    protocol = json.loads((ROOT/parent['protocol']).read_text())
    runtime = prepare_runtime(ROOT, protocol, ROOT/parent['output']/'frozen_runtime',
                              revision=parent['historical_model_revision'], install=True)
    from src.evaluation.m3w_development_evaluation import content_digest, validate_plan, _load_cost_head
    from src.world_model.m3w_neural_gain_harm import read_verified_oof_cache, validate_oof_groups
    from src.world_model.m3w_supervised_intervention import predict_linear_gain_harm
    device = cfg['families'][args.family]
    study = ROOT/f'data/stage_cvpr2027_experiments/8to12_{args.family}_v6'
    bindings, jobs = {}, []
    for seed in cfg['seeds']:
        paths = [study/f'seed{seed}_{name}/artifact.json' for name in
                 ('full','hold0','hold1','hold2','ridge','neural_cost')]
        artifacts = [json.loads(p.read_text()) for p in paths]
        contract = RelocatedCodeContract(protocol, ROOT, artifacts, runtime=runtime)
        plan_path = study/f'seed{seed}_development_plan.json'
        plan = json.loads(plan_path.read_text())
        rules, _ = validate_plan(contract, plan)
        groups = read_verified_oof_cache(contract, study/f'seed{seed}_ridge')
        provenance = validate_oof_groups(contract, groups, seed=seed)
        for path in [*paths, plan_path]:
            bindings[str(path.relative_to(ROOT))] = file_digest(path)
        for name in ('ridge','neural_cost'):
            for filename in ('fit_report.json',):
                p = study/f'seed{seed}_{name}'/filename
                bindings[str(p.relative_to(ROOT))] = file_digest(p)
        for artifact in artifacts:
            p = ROOT/artifact['path']
            if file_digest(p) != artifact['sha256']:
                raise ValueError('Artifact bytes changed')
            bindings[artifact['path']] = artifact['sha256']
        for filename in ('run_identity.json','fold_0.json','fold_0.npz','fold_1.json','fold_1.npz','fold_2.json','fold_2.npz'):
            p = study/f'seed{seed}_ridge'/filename
            bindings[str(p.relative_to(ROOT))] = file_digest(p)
        for name in ('ridge','neural_cost'):
            r = json.loads((study/f'seed{seed}_{name}'/'fit_report.json').read_text())
            if r['oof_feature_identity'] != provenance['oof_feature_identity']:
                raise ValueError('Cost heads did not share the verified OOF features')
            if name == 'neural_cost' and (r['group_sha256'] != provenance['group_sha256'] or r['runtime']['device'] != device):
                raise ValueError('Neural cost target provenance or original device changed')
        jobs.append((seed, contract, plan, groups))
    dependencies = [CONFIG, 'scripts/audit_m3w_cost_head_fit.py',
                    'src/evaluation/m3w_cost_fit_forensics.py', 'src/evaluation/m3w_frozen_runtime.py']
    identity = dict(family=args.family, protocol_sha256=contract.digest, source_bindings=bindings,
        code_sha256={p:file_digest(ROOT/p) for p in dependencies}, frozen_runtime=runtime,
        parent_risk_sha256=cfg['parent_risk_sha256'],
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
            numpy=np.__version__, machine=platform.machine(), device=device, threads=cfg['threads'], workers=0),
        new_training=False, cost_head_evaluation_role='in_sample_fit_not_held_out')
    run = content_digest(identity)
    out = ROOT/cfg['private_output']/args.family
    if args.resume:
        if json.loads((out/'identity.json').read_text()) != identity:
            raise ValueError('Run identity changed; do not silently overwrite diagnosis')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'identity.json', identity)
    completed = (out/'completion.json').exists()
    began, new, reused, records, receipts = time.monotonic(), 0, 0, [], []
    for seed, contract, plan, groups in jobs:
        cache, receipt_path = out/f'seed{seed}.json', out/f'seed{seed}.receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt != dict(run_sha256=run, seed=seed, cache_sha256=file_digest(cache)):
                raise ValueError('Seed diagnosis changed')
            result = json.loads(cache.read_text())
            reused += 1
        else:
            if completed:
                raise ValueError('Completed result missing a receipt')
            x, y = (np.concatenate([g[k] for g in groups]) for k in ('features','targets'))
            recordings = [r['recording_id'] for g in groups for r in g['identities']]
            result = []
            for name in ('ridge','neural_cost'):
                candidate = next(c for c in plan['candidates'] if c['risk_head_id'] == f'seed{seed}_{name}')
                head = _load_cost_head(contract, candidate, device=device)
                raw, normalization_exact = None, False
                if name == 'ridge':
                    np.testing.assert_array_equal(head['mean'], x.mean(0))
                    np.testing.assert_array_equal(head['scale'], np.maximum(x.std(0), 1e-6))
                    raw = ((x-head['mean'])/head['scale']) @ head['coef'].T+head['intercept']
                    pred = np.maximum(raw, 0)
                    reference = predict_linear_gain_harm(head, x)
                    np.testing.assert_array_equal(pred, np.column_stack((reference['benefit'],reference['harm'])))
                    normalization_exact = True
                else:
                    tensor = torch.as_tensor(x, dtype=torch.float32)
                    torch.testing.assert_close(head.mean.cpu(), tensor.mean(0), rtol=0, atol=0)
                    torch.testing.assert_close(head.scale.cpu(), tensor.std(0, correction=0).clamp_min(1e-6), rtol=0, atol=0)
                    blocks = []
                    with torch.no_grad():
                        for start in range(0, len(x), cfg['inference_batch_size']):
                            scores = head(tensor[start:start+cfg['inference_batch_size']].to(device))
                            blocks.append(torch.stack((scores['benefit'],scores['harm']),1).cpu().numpy())
                    pred = np.concatenate(blocks)
                    normalization_exact = True
                metrics = diagnose_fit(y, pred, recordings, rules['policies'], raw_ridge=raw,
                                       top_fraction=cfg['top_target_fraction'])
                result.append(dict(family=args.family, seed=seed, head=name, feature_dim=x.shape[1],
                    normalization_exact=normalization_exact, device=device if name=='neural_cost' else 'numpy_ridge', **metrics))
                temp = out/f'seed{seed}_{name}.tmp.npz'
                np.savez(temp, target=y, predicted=pred, recordings=np.asarray(recordings),
                         **({'raw_ridge':raw} if raw is not None else {}))
                os.replace(temp, out/f'seed{seed}_{name}.npz')
            write_json(cache, result)
            write_json(receipt_path, dict(run_sha256=run, seed=seed, cache_sha256=file_digest(cache)))
            new += 1
        records.extend(result)
        receipts.append(dict(path=str(receipt_path.relative_to(ROOT)), sha256=file_digest(receipt_path)))
        for name, expected in zip(('ridge','neural_cost'), result):
            with np.load(out/f'seed{seed}_{name}.npz', allow_pickle=False) as a:
                replay = diagnose_fit(a['target'],a['predicted'],a['recordings'].tolist(),rules['policies'],
                    raw_ridge=a['raw_ridge'] if 'raw_ridge' in a.files else None,top_fraction=cfg['top_target_fraction'])
            if replay != {k:expected[k] for k in replay}:
                raise ValueError('Saved cost arrays do not replay the diagnosis')
            p = out/f'seed{seed}_{name}.npz'
            receipts.append(dict(path=str(p.relative_to(ROOT)), sha256=file_digest(p)))
        if not completed:
            status=dict(pid=os.getpid(), family=args.family, completed_seeds=len(records)//2,
                        new_seeds=new, reused_seeds=reused, elapsed_seconds=time.monotonic()-began, status='running')
            write_json(out/'heartbeat.json',status)
            print(json.dumps(status),flush=True)
            if args.pilot and new == 1:
                return
    report = dict(result_source='fresh_run_cost_head_readout_cached_verified_fit_arrays_and_models',
        run_sha256=run, source_bindings=bindings, results=records,
        new_training=False,new_forecast_inference=False,new_development_labels=False,
        primary_metric_changed=False,policy_selection=False,independent_calibration=False,
        stage5c_executed=False,smc_enabled=False)
    public = ROOT/cfg['output']; public.mkdir(parents=True,exist_ok=True)
    report_path=public/(args.family+'.json')
    if completed:
        old = json.loads((out/'completion.json').read_text())
        if (old['run_sha256'] != run or old['receipts'] != receipts or old['report_sha256'] != file_digest(report_path)
                or json.loads(report_path.read_text()) != report):
            raise ValueError('Completed diagnosis replay changed')
    else:
        write_json(report_path,report)
        write_json(out/'completion.json',dict(run_sha256=run,report_sha256=file_digest(report_path),receipts=receipts,
            elapsed_seconds=time.monotonic()-began,new_seeds=new,reused_seeds=reused))
        write_json(out/'heartbeat.json',dict(pid=os.getpid(),status='complete',completed_seeds=len(records)//2))
    print(json.dumps(dict(status='complete_exact_replay' if completed else 'complete',family=args.family,
                         new_seeds=new,reused_seeds=reused)),flush=True)


if __name__ == '__main__':
    main()
