"""Fresh fixed-checkpoint inference, separate from cached metric reproduction."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_source_forecast as forecast
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json,array_hash
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast',action='store_true')
    parser.add_argument('--intervention',action='store_true')
    args = parser.parse_args()
    if not (args.forecast or args.intervention):
        parser.error('Explicit study required')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg,manifest,receipts,identity = forecast.load_source()
    data = forecast.prepare(manifest,receipts,identity)
    designs = forecast.trials(reg,data,identity)
    if args.forecast:
        checks = []
        for key,design,ti in designs:
            complete = forecast.complete(key,ti,reg)
            cp = torch.load(ROOT/complete['checkpoint'],map_location='cpu',weights_only=False)
            if cp['identity']!=ti or cp['step']!=reg['training']['steps']:
                raise ValueError('Checkpoint identity differs')
            ids = design['held_ids'][:128]
            model = SourceForecaster(reg['architecture'],design['baseline_index'])
            model.load_state_dict(cp['model'])
            fresh = predict(model,data,ids,128)
            path = forecast.PRIVATE/'predictions'/(key+'.npz')
            receipt = json.loads(path.with_suffix('.json').read_text())
            if digest(path)!=receipt['sha256']:
                raise ValueError('Saved inference bytes differ')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'][:128],ids)
                saved = z['prediction'][:128].copy()
            np.testing.assert_allclose(fresh,saved,atol=1e-5,rtol=1e-6)
            checks.append(dict(trial=key,rows=len(ids),ids_sha256=array_hash(ids),
                exact=bool(np.array_equal(fresh,saved)),max_absolute_difference=float(np.abs(fresh-saved).max()),
                checkpoint_sha256=complete['checkpoint_sha256']))
        forecast.assert_identity(identity)
        out = dict(result_source='fresh_run_checkpoint_inference_on_cached_verified_inputs',
            verifier_sha256=digest(Path(__file__)),analysis_sha256=digest(forecast.PUBLIC/'analysis.json'),
            checks=checks,all_passed=True,new_training=False,new_checkpoint_inference=True,
            raw_source_reconversion=False,held_out_roles_opened=False)
        immutable_json(forecast.PUBLIC/'checkpoint_replay.json',out)
        print(json.dumps(dict(study='forecast',checks=len(checks),all_passed=True,
                             max_difference=max(c['max_absolute_difference'] for c in checks))),flush=True)
    if args.intervention:
        from scripts import run_m3w_european_source_intervention as intervention
        from src.world_model.m3w_native_gain_harm import build_head,predict_neural,predict_ridge
        ireg,idata,iidentity,idesigns,_ = intervention.load()
        checks = []
        for key,design,ti in idesigns:
            if ti['kind']!='complement':
                continue
            a = intervention.assemble(idata,iidentity,key,design,ti)
            held = design['held_ids'][:4096]
            for arm in ireg['arms']:
                name = key+'_'+arm
                receipt = json.loads((intervention.PRIVATE/'heads'/name/'complete.json').read_text())
                artifact = receipt['artifacts']['checkpoint']
                if digest(ROOT/artifact['path'])!=artifact['sha256']:
                    raise ValueError('Cost checkpoint changed')
                cp = torch.load(ROOT/artifact['path'],map_location='cpu',weights_only=False)
                if cp['identity']!=receipt['identity']:
                    raise ValueError('Cost checkpoint identity differs')
                for k in ('mean','std','constant','weights','known'):
                    np.testing.assert_array_equal(cp['preprocess'][k],a['pr'][k])
                if arm=='ridge':
                    fresh = predict_ridge(cp['head'],a['x'][held],a['same'][held],cp['preprocess'])
                else:
                    model = build_head(ireg['training']['width'],cp['preprocess'],ti['seed'])
                    model.load_state_dict(cp['model'])
                    fresh = predict_neural(model,a['x'][held],a['same'][held],cp['preprocess'])
                fresh = np.maximum(fresh,0)
                score_artifact = receipt['artifacts']['predicted_costs']
                if digest(ROOT/score_artifact['path'])!=score_artifact['sha256']:
                    raise ValueError('Cost scores changed')
                with np.load(ROOT/score_artifact['path'],allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'][:4096],held)
                    saved = z['costs'][:4096].copy()
                np.testing.assert_allclose(fresh,saved,atol=1e-5,rtol=1e-6)
                checks.append(dict(trial=name,rows=len(held),ids_sha256=array_hash(held),
                    exact=bool(np.array_equal(fresh,saved)),max_absolute_difference=float(np.abs(fresh-saved).max()),
                    checkpoint_sha256=artifact['sha256']))
            del a
        intervention.assert_identity(iidentity)
        out = dict(result_source='fresh_run_cost_checkpoint_inference_on_cached_verified_causal_features',
            verifier_sha256=digest(Path(__file__)),analysis_sha256=digest(intervention.PUBLIC/'analysis.json'),
            checks=checks,all_passed=True,new_training=False,new_checkpoint_inference=True,
            raw_source_reconversion=False,held_out_roles_opened=False)
        immutable_json(intervention.PUBLIC/'checkpoint_replay.json',out)
        print(json.dumps(dict(study='intervention',checks=len(checks),all_passed=True,
                             max_difference=max(c['max_absolute_difference'] for c in checks))),flush=True)


if __name__=='__main__':
    main()
