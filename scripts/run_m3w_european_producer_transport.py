"""Replace frozen producers on identical excluded rows; no fitting or tuning."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_nested_calibration as previous
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.world_model.m3w_european_source_forecast import SourceForecaster, baseline_numpy
from src.world_model.m3w_native_forecast import predict
from src.world_model.m3w_native_gain_harm import build_head, predict_neural
from src.world_model.m3w_european_source_intervention import causal_cost_features
from src.world_model.m3w_european_conditional_risk import nonnegative_moments
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_producer_transport import validate_roles, score_diagnosis

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_producer_transport_v1'
CONFIG = 'configs/m3w_european_producer_transport_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_producer_transport.py',
    'src/evaluation/m3w_producer_transport.py', 'tests/test_m3w_producer_transport.py',
    'src/evaluation/m3w_opportunity_diagnosis.py',
    'outputs/publication_readiness_2026_09/european_producer_transport_v1/diagnosis_plan.md')


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    previous.assert_identity(identity['previous_identity'])
    if digest(previous.PUBLIC/'analysis.json') != identity['previous_analysis_sha256']:
        raise ValueError('Previous outcome changed')
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen transport diagnosis changed: '+path)


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    preg, forecast_reg, data, pid = previous.load()
    for name in ('verification.json', 'checkpoint_replay.json', 'accounting_audit.json'):
        r = json.loads((previous.PUBLIC/name).read_text())
        if not r['all_passed'] or r['analysis_sha256'] != digest(previous.PUBLIC/'analysis.json'):
            raise ValueError('Fully verified starting result required')
    if reg['seeds'] != preg['seeds'] or reg['budget'] != preg['predicted_risk_budget']:
        raise ValueError('Unchanged seeds and risk budget required')
    if any(reg[k] for k in ('new_training','threshold_refit','reserved_roles_opened','deployment_changed','stage5c_executed','smc_enabled')):
        raise ValueError('Source-only fixed-model diagnosis required')
    identity = dict(previous_identity=pid, previous_analysis_sha256=digest(previous.PUBLIC/'analysis.json'),
        bindings={p: digest(ROOT/p) for p in FILES})
    heads = previous.all_heads(preg, data, pid)
    immutable_json(PRIVATE/'identity.json', identity)
    total = sum(2*len(d['held_ids']) for _, c, _, _, d in previous.jobs(preg, data, pid) if c == 'neural')
    immutable_json(PUBLIC/'matrix.json', dict(identity=identity, new_prediction_banks=18,
        prediction_rows=total, views=72, small_vs_full_comparisons=36, new_training=False,
        reserved_roles_opened=False, eight_excluded_sources_per_pair=True))
    return reg, preg, forecast_reg, data, identity, heads


def small_record(fold, half, seed, design, data, identity):
    key = f'fold{fold}_half{half}_seed{seed}'
    r = json.loads((previous.PRIVATE/'producers'/key/'complete.json').read_text())
    binding = r['identity']
    if binding['identity'] != identity['previous_identity']['producer_identity'] or r['fit']['step'] != 4000:
        raise ValueError('Wrong frozen small producer')
    validate_roles(sorted(set(data['sites'][design['train_ids']])), binding['training_sites'],
        sorted(set(data['sites'][design['held_ids']])))
    if digest(ROOT/r['checkpoint']['path']) != r['checkpoint']['sha256']:
        raise ValueError('Producer checkpoint changed')
    return key, r


def model_from(r, forecast_reg):
    state = torch.load(ROOT/r['checkpoint']['path'], map_location='cpu', weights_only=False)
    if state['identity'] != r['identity'] or state['step'] != 4000:
        raise ValueError('Incomplete frozen forecaster')
    model = SourceForecaster(forecast_reg['architecture'], r['identity']['baseline_index'])
    model.load_state_dict(state['model'])
    return model


def bank(reg, preg, forecast_reg, data, identity, pilot=False, verify=False):
    checks = []
    for _, c, fold, seed, design in previous.jobs(preg, data, identity['previous_identity']):
        if c != 'neural':
            continue
        for half in (0, 1):
            key, r = small_record(fold, half, seed, design, data, identity)
            ids = design['held_ids']
            model = model_from(r, forecast_reg)
            path = PRIVATE/'predictions'/(key+'.npz')
            rp = path.with_suffix('.json')
            if pilot:
                start = time.monotonic()
                p = predict(model, data, ids[:reg['replay_rows']], reg['batch_size'])
                seconds = time.monotonic()-start
                immutable_json(PRIVATE/'pilot.json', dict(identity=identity, producer=key,
                    rows=len(p), seconds=seconds, prediction_sha256=array_hash(p),
                    new_training=False, estimated_full_inference_seconds=seconds*3827628/len(p)))
                beat('pilot_complete', rows=len(p), seconds=seconds, new_training=False)
                return
            if rp.exists():
                rec = json.loads(rp.read_text())
                if rec['identity'] != identity or rec['producer_checkpoint'] != r['checkpoint'] or digest(path) != rec['sha256']:
                    raise ValueError('Changed prediction bank')
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(ids, z['ids'])
                    if verify:
                        sampled = ids[:reg['replay_rows']]
                        fresh = predict(model, data, sampled, reg['batch_size'])
                        np.testing.assert_array_equal(fresh, z['prediction'][:len(sampled)])
                checks.append(dict(producer=key, bank=artifact(path), rows=len(ids),
                    replay_rows=min(len(ids),reg['replay_rows']) if verify else 0, exact=verify))
            else:
                if verify:
                    raise ValueError('Cannot replay an absent bank')
                if shutil.disk_usage(PRIVATE).free < 10*1024**3:
                    raise OSError('Below 10GiB; preserve artifacts and stop')
                beat('past_only_inference', producer=key, rows=len(ids))
                start = time.monotonic()
                p = predict(model, data, ids, reg['batch_size'])
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp.npz')
                np.savez(tmp, ids=ids, prediction=p)
                os.replace(tmp, path)
                immutable_json(rp, dict(identity=identity, producer_checkpoint=r['checkpoint'],
                    **artifact(path), ids_sha256=array_hash(ids), prediction_seconds=time.monotonic()-start,
                    inference_input='past_geometry_only', future_inputs=False))
                checks.append(dict(producer=key, bank=artifact(path), rows=len(ids), replay_rows=0, exact=False))
            beat('prediction_bank_checked', producer=key, replay=verify)
    assert_identity(identity)
    immutable_json(PUBLIC/('bank_replay.json' if verify else 'bank_completion.json'),
        dict(identity=identity, checks=checks, new_training=False, predictions=sum(x['rows'] for x in checks)))


def head_scores(r, x, same, task, settings):
    state = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
    if state['identity'] != r['identity'] or state['step'] != 2000:
        raise ValueError('Frozen head identity changed')
    pr = state['preprocess']
    model = build_head(settings['width'], pr, r['identity']['lineage']['seed'])
    model.load_state_dict(state['model'])
    raw = predict_neural(model, x, same if task == 'utility' else np.zeros(len(x), bool), pr)
    return np.maximum(raw,0) if task == 'utility' else nonnegative_moments(raw,same)


def reference_decisions(name, event, fold, ids, bits, identity, previous_receipts):
    checked = 0
    for cal in range(3):
        if cal == fold:
            continue
        path = previous.PRIVATE/'decisions'/(name+'_'+event+f'_cal{cal}_none.json')
        if digest(path) != previous_receipts[str(path.relative_to(ROOT))]:
            raise ValueError('Original decision receipt changed')
        rec = json.loads(path.read_text())
        if rec['identity'] != identity['previous_identity'] or digest(ROOT/rec['path']) != rec['sha256']:
            raise ValueError('Original decision bank changed')
        with np.load(ROOT/rec['path'], allow_pickle=False) as z:
            pos = np.searchsorted(ids, z['ids'])
            np.testing.assert_array_equal(ids[pos], z['ids'])
            np.testing.assert_array_equal(bits[pos], z['switch'])
        checked += 1
    return checked


def evaluate(reg, preg, data, identity, heads, verify):
    old = json.loads((previous.PUBLIC/'analysis.json').read_text())
    previous_receipts = {r['path']:r['sha256'] for r in old['decisions']}
    results, comparisons, lineages = {}, {}, []
    matched = 0
    for name, c, fold, seed, design in previous.jobs(preg, data, identity['previous_identity']):
        if c != 'neural':
            continue
        ids = design['held_ids']
        sites = data['sites'][ids]
        roster = sorted(set(sites))
        b = baseline_numpy(data['history'][ids], 1)-data['origin'][ids,None]
        full = identity['previous_identity']['producer_identity']['frozen_final_producers'][f'single{fold}_seed{seed}']
        validate_roles(sorted(set(data['sites'][design['train_ids']])), full['training_sites'], roster)
        predictions = dict(full4=previous.read_predictions(full['prediction'],ids),
            damping097=baseline_numpy(data['history'][ids],3)-data['origin'][ids,None])
        for half in (0,1):
            key, rec = small_record(fold,half,seed,design,data,identity)
            art = json.loads((PRIVATE/'predictions'/(key+'.json')).read_text())
            if art['identity'] != identity or art['producer_checkpoint'] != rec['checkpoint']:
                raise ValueError('New bank lineage changed')
            predictions[f'half{half}'] = previous.read_predictions(art,ids)
            lineages.append(dict(producer=key, trained=rec['identity']['training_sites'],
                readout=roster, ids_sha256=array_hash(ids), checkpoint=rec['checkpoint'], prediction=artifact(ROOT/art['path'])))
        cv = np.asarray(data['baseline_ade'][ids,1])
        cvf = np.asarray(data['baseline_fde'][ids,1])
        moving = np.linalg.norm(np.diff(data['history'][ids],axis=1),axis=2).sum(1)>0
        masks = dict(all=np.ones(len(ids),bool),easy=(cv>0)&(cv<=design['easy_cut']),hard=cv>=design['hard_cut'])
        def metric(model, reference, mask):
            return paired_scene_metrics(model[mask],reference[mask],sites[mask],expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
                bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
        costs_by_view = {}
        raw = {}
        for variant,p in predictions.items():
            base_name = name if variant != 'damping097' else f'damping097_fold{fold}_seed{seed}'
            x,_ = causal_cost_features(data['geometry'][ids],b,p)
            same = np.all(b==p,axis=(1,2))
            scores = {task:head_scores(heads[base_name+'_'+task],x,same,task,preg['head_training'])
                for task in ('utility',*reg['events'])}
            if variant in ('full4','damping097'):
                for task,s in scores.items():
                    np.testing.assert_array_equal(s,previous.scores(heads[base_name+'_'+task],ids))
            ade,fde = native_errors(p.astype(float)+data['origin'][ids,None],data['target_eval'][ids],
                data['valid'][ids],np.ones(len(ids)))
            raw[variant] = ade
            for event in reg['events']:
                d = score_diagnosis(cv,ade,scores['utility'],scores[event],moving,sites,
                    easy_cut=design['easy_cut'],event=event,budget=reg['budget'],
                    resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
                why = d.pop('reasons'); bits = why==5
                if variant in ('full4','damping097'):
                    matched += reference_decisions(base_name,event,fold,ids,bits,identity,previous_receipts)
                selected, selected_fde = np.where(bits,ade,cv),np.where(bits,fde,cvf)
                costs_by_view[(variant,event)] = selected
                zero = np.isfinite(cv)&(cv==0)
                metrics = {subset:metric(selected,cv,mask) for subset,mask in masks.items()}
                key = f'fold{fold}_seed{seed}_{variant}_{event}'
                results[key] = dict(**d, ADE_vs_CV=metrics, FDE_vs_CV=metric(selected_fde,cvf,masks['all']),
                    raw_ADE_vs_CV=metric(ade,cv,masks['all']),
                    zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((selected[zero]>0).sum())),
                    safety_observed_pass=bool(metrics['easy']['worst_scene_gain_percent'] is not None
                        and metrics['easy']['worst_scene_gain_percent']>=-2 and not (selected[zero]>0).any()),
                    decision_sha256=array_hash(ids,bits), inference_input='past_geometry_and_frozen_scores_only')
            beat('variant_readout',fold=fold,seed=seed,variant=variant,rows=len(ids),verify=verify)
        for half in (0,1):
            for event in reg['events']:
                key = f'fold{fold}_seed{seed}_half{half}_{event}'
                comparisons[key] = dict(raw_small_vs_full=metric(raw[f'half{half}'],raw['full4'],masks['all']),
                    policy_small_vs_full={subset:metric(costs_by_view[(f'half{half}',event)],costs_by_view[('full4',event)],mask)
                        for subset,mask in masks.items()})
    if len(results)!=72 or len(comparisons)!=36 or matched!=72:
        raise ValueError('Incomplete registered transport matrix')
    result = dict(identity=identity,views=results,small_vs_full=comparisons,lineages=lineages,
        result_source='fresh_run_inference_and_diagnosis_cached_verified_models',
        original_decisions_reconstructed=matched,new_training=False,threshold_refit=False,
        reserved_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:
        immutable_json(PUBLIC/'verification.json',dict(analysis_sha256=digest(PUBLIC/'analysis.json'),
            metrics_recomputed=True,all_passed=True,original_decisions_reconstructed=72))
    beat('transport_diagnosis_complete',views=len(results),verify=verify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare','pilot','run','verify'):
        parser.add_argument('--'+flag,action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()) or (args.pilot and (args.run or args.verify)):
        parser.error('Explicit phase required; pilot must be separate')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        reg,preg,forecast_reg,data,identity,heads = load()
        beat('source_verified',rows=len(data['sites']),threads=4,workers=0,architecture=platform.machine())
        if args.pilot:
            bank(reg,preg,forecast_reg,data,identity,pilot=True)
        if args.run or args.verify:
            bank(reg,preg,forecast_reg,data,identity,verify=args.verify)
            evaluate(reg,preg,data,identity,heads,args.verify)
        beat('phase_complete')


if __name__=='__main__':
    main()
