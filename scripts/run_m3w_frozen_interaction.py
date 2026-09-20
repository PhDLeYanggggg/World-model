"""Replay all frozen v6 predictors with the missing unary-geometry control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CONFIG = 'configs/m3w_frozen_interaction_v1.json'
DEPENDENCIES = [CONFIG,'scripts/run_m3w_frozen_interaction.py','src/evaluation/m3w_frozen_interaction.py',
    'src/world_model/m3w_interaction_controls.py','src/evaluation/m3w_interaction_control_evaluation.py',
    'src/evaluation/m3w_frozen_runtime.py']


def atomic_json(path,value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    os.replace(temp,path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family',choices=('transformer','eqmotion'),required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pilot-queries',type=int)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if args.pilot_queries is not None and args.pilot_queries <= 0:
        raise ValueError('Positive pilot length required')
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
    cfg = json.loads((ROOT/CONFIG).read_text())
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(cfg['threads'])
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import torch
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.evaluation.m3w_frozen_runtime import prepare_runtime,RelocatedCodeContract
    protocol_path = ROOT/cfg['protocol']
    if file_digest(protocol_path) != cfg['protocol_file_sha256']:
        raise ValueError('Frozen parent protocol changed')
    protocol = json.loads(protocol_path.read_text())
    runtime = prepare_runtime(ROOT,protocol,ROOT/cfg['output']/'frozen_runtime',
                              revision=cfg['historical_model_revision'],install=True)
    from src.evaluation.m3w_development_evaluation import (
        content_digest,validate_plan,load_verified_forecaster,_load_cost_head,scene_requests,decide_scene,score_scene)
    from src.evaluation.m3w_interaction_control_evaluation import attach_interaction_controls
    from src.evaluation.m3w_frozen_interaction import verify_original_rows,compact_query,summarize_controls
    from scripts.evaluate_m3w_forecast_supplement import verify_completed_evaluation
    torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1)
    device = cfg['families'][args.family]
    if protocol['seeds'] != cfg['seeds']:
        raise ValueError('Keep all registered seeds')
    study = ROOT/f'data/stage_cvpr2027_experiments/8to12_{args.family}_v6'
    if json.loads((study/'runner_heartbeat.json').read_text()).get('state') != 'requested_seeds_complete':
        raise ValueError('Original study incomplete')
    families,source_hashes = [],{}
    for seed in cfg['seeds']:
        artifacts = [json.loads((study/f'seed{seed}_{name}/artifact.json').read_text())
                     for name in ('full','hold0','hold1','hold2','ridge','neural_cost')]
        contract = RelocatedCodeContract(protocol,ROOT,artifacts,runtime=runtime)
        plan_path = study/f'seed{seed}_development_plan.json'
        plan = json.loads(plan_path.read_text())
        rules,recordings = validate_plan(contract,plan)
        if recordings != cfg['recordings'] or len(plan['candidates']) != 4:
            raise ValueError('Changed admitted development population or candidate family')
        primary = study/f'seed{seed}_development'
        verify_completed_evaluation(primary,file_digest)
        old_identity = json.loads((primary/'run_identity.json').read_text())
        if old_identity['plan'] != plan or old_identity['device'] != device or old_identity['protocol_sha256'] != contract.digest:
            raise ValueError('Original predictor identity changed')
        for candidate in plan['candidates']:
            for recording in recordings:
                key = [candidate['id'],recording];stem = content_digest(key)
                p = primary/(stem+'.rows.json')
                receipt = json.loads((primary/(stem+'.receipt.json')).read_text())
                digest = file_digest(p)
                if receipt != dict(key=key,run_sha256=content_digest(old_identity),cache_sha256=digest):
                    raise ValueError('Original row export changed')
                source_hashes[str(p.relative_to(ROOT))] = digest
        source_hashes[str(plan_path.relative_to(ROOT))] = file_digest(plan_path)
        families.append((seed,contract,plan,rules,recordings))
    identity = dict(family=args.family,protocol_sha256=contract.digest,device=device,threads=cfg['threads'],
        environment=dict(python=platform.python_version(),torch=str(torch.__version__),machine=platform.machine()),
        code_sha256={p:file_digest(ROOT/p) for p in DEPENDENCIES},
        decision_sha256=file_digest(ROOT/cfg['decision']),source_exports=source_hashes,
        artifacts=[c.artifacts for _,c,_,_,_ in families],frozen_runtime=runtime,eligible_for_selection=False)
    if args.preflight_only:
        print(json.dumps(dict(status='frozen_artifacts_checked_no_evaluation',family=args.family,
            original_exports=len(source_hashes),seeds=cfg['seeds'],device=device)),flush=True)
        return
    out = ROOT/cfg['output']/args.family
    run = content_digest(identity)
    if args.resume:
        if json.loads((out/'identity.json').read_text()) != identity:
            raise ValueError('Resume identity changed')
    else:
        out.mkdir(parents=True,exist_ok=False);atomic_json(out/'identity.json',identity)
    already_complete = (out/'completion.json').exists()
    began = time.monotonic();new_queries = 0;reused = 0;all_results = {};all_receipts = []
    for seed,contract,plan,rules,recordings in families:
        for candidate in plan['candidates']:
            model = head = None
            candidate_rows,candidate_queries,candidate_replay = [],[],[]
            for recording in recordings:
                reader,_ = contract.open_recording(recording,purpose='development')
                key = [candidate['id'],recording];stem = content_digest(key)
                old = json.loads((study/f'seed{seed}_development'/(stem+'.rows.json')).read_text())
                offset,chunk_index = 0,0
                iterator = iter(scene_requests(reader,protocol['task'],stride=rules['query_stride']))
                exhausted = False
                while not exhausted:
                    scenes = []
                    for _ in range(cfg['checkpoint_queries']):
                        try: scenes.append(next(iterator))
                        except StopIteration: exhausted = True;break
                    if not scenes: break
                    ck = [seed,*key,chunk_index];name = content_digest(ck)
                    cache,receipt_path = out/(name+'.json'),out/(name+'.receipt.json')
                    if receipt_path.exists():
                        receipt = json.loads(receipt_path.read_text())
                        if receipt != dict(key=ck,run_sha256=run,cache_sha256=file_digest(cache)):
                            raise ValueError('Private decision batch changed')
                        part = json.loads(cache.read_text());reused += 1
                        actual_keys = [(x['recording_id'],x['frame_id'],x['horizon_raw'],len(x['agents'])) for x in scenes]
                        saved_keys = [(x['recording_id'],x['frame_id'],x['horizon_raw'],x['agent_count']) for x in part['queries']]
                        if actual_keys != saved_keys: raise ValueError('Resumed input membership changed')
                    else:
                        if already_complete:raise ValueError('Completed run has a missing batch receipt')
                        if model is None:
                            model = load_verified_forecaster(contract,candidate['forecaster_id'],device=device)
                            if model._fitted_baseline_name != candidate['baseline']:raise ValueError('Baseline mismatch')
                            head = _load_cost_head(contract,candidate,device=device)
                        part = dict(rows=[],queries=[])
                        for scene in scenes:
                            decision = decide_scene(scene,model,head,baseline=candidate['baseline'],
                                policy=rules['policies'][candidate['policy_id']],geometry=rules['geometry_by_recording'][recording],
                                device=device,solver_seconds=rules['solver_seconds'])
                            decision = attach_interaction_controls(scene,decision,
                                policy=rules['policies'][candidate['policy_id']],geometry=rules['geometry_by_recording'][recording],
                                time_limit_seconds=rules['solver_seconds'])
                            # Every control has already decided before this separate label API.
                            labels = reader.get_scene_labels(scene)
                            rows = score_scene(scene,decision,labels,label_policy=rules['label_policy'])
                            verify_original_rows(old[offset+len(part['rows']):offset+len(part['rows'])+len(rows)],rows)
                            part['rows'].extend(rows);part['queries'].append(compact_query(scene,decision))
                            new_queries += 1
                        atomic_json(cache,part)
                        atomic_json(receipt_path,dict(key=ck,run_sha256=run,cache_sha256=file_digest(cache)))
                    verification = verify_original_rows(old[offset:offset+len(part['rows'])],part['rows'])
                    candidate_rows.extend(part['rows']);candidate_queries.extend(part['queries']);candidate_replay.append(verification)
                    offset += len(part['rows']);chunk_index += 1
                    all_receipts.append(dict(path=str(receipt_path.relative_to(ROOT)),sha256=file_digest(receipt_path)))
                    status = dict(pid=os.getpid(),family=args.family,seed=seed,candidate=candidate['id'],recording=recording,
                        batch=chunk_index,new_scene_queries=new_queries,reused_batches=reused,
                        elapsed_seconds=time.monotonic()-began,status='running_not_complete')
                    if not already_complete:
                        atomic_json(out/'heartbeat.json',status);print(json.dumps(status),flush=True)
                    if args.pilot_queries and new_queries >= args.pilot_queries:
                        status['status'] = 'pilot_complete_full_run_not_complete';atomic_json(out/'heartbeat.json',status)
                        return
                if offset != len(old):raise ValueError('Original population not exhausted exactly')
            result = summarize_controls(candidate_rows,candidate_queries,protocol)
            result['original_replay'] = {k:sum(v[k] for v in candidate_replay) for k in candidate_replay[0]}
            all_results[candidate['id']] = result
            del model,head,candidate_rows,candidate_queries
    report = dict(result_source='fresh_run' if new_queries else 'cached_verified',family=args.family,
        identity_sha256=run,results=all_results,source_exports=source_hashes,new_scene_queries=new_queries,
        reused_batches=reused,elapsed_seconds=time.monotonic()-began,primary_metric_changed=False,
        independent_confirmation=False,eligible_for_selection=False,new_training=False,stage5c_executed=False,smc_enabled=False)
    report_path = out/'report.json'
    if (out/'completion.json').exists():
        completed = json.loads((out/'completion.json').read_text())
        if completed['run_sha256'] != run or completed['report_sha256'] != file_digest(report_path) or completed['receipts'] != all_receipts:
            raise ValueError('Completed output identity changed')
        previous = json.loads(report_path.read_text())
        if previous['results'] != all_results:raise ValueError('Completed aggregate replay differs')
        print(json.dumps(dict(status='completed_exact_semantic_replay_zero_new_evaluation',family=args.family)),flush=True)
    else:
        atomic_json(report_path,report)
        atomic_json(out/'completion.json',dict(run_sha256=run,report_sha256=file_digest(report_path),receipts=all_receipts))
        atomic_json(out/'heartbeat.json',dict(status='complete_not_selected',pid=os.getpid(),family=args.family,
            new_scene_queries=new_queries,elapsed_seconds=time.monotonic()-began))
        print(json.dumps(dict(status='all_frozen_candidates_complete_not_selected',family=args.family)),flush=True)


if __name__ == '__main__':
    main()
