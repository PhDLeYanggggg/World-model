"""Fixed 2x2 capacity-duration study; parent checkpoints remain immutable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 is required before importing Torch')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_tempered_cost as parent
from scripts.run_m3w_bounded_cost import training_data
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_cost_capacity import fit_path
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch

CONFIG = 'configs/m3w_cost_capacity_v1.json'
CODE = ('scripts/run_m3w_cost_capacity.py', 'src/world_model/m3w_cost_capacity.py',
        'src/evaluation/m3w_cost_capacity_eval.py', 'scripts/verify_m3w_cost_capacity.py',
        'tests/test_m3w_cost_capacity.py', 'tests/test_m3w_cost_capacity_protocol.py')
ARMS = ('narrow_short', 'narrow_long', 'wide_short', 'wide_long')


def validate_config(cfg, pcfg):
    if (cfg['widths'] != [64,128] or cfg['budgets'] != [3000,12000] or cfg['loss_exponent'] != 1
            or cfg['sites'] != pcfg['sites'] or cfg['seeds'] != pcfg['seeds']
            or cfg['primary_arm'] != 'wide_long' or cfg['primary_reference'] != pcfg['primary_reference']
            or cfg['matched_count_reference'] != pcfg['matched_count_reference']
            or cfg['bootstrap_resamples'] != 3000 or any(cfg[k] for k in ('threshold_search', 'model_selection',
                'closed_role_readout', 'risk_calibration', 'independent_confirmation', 'deployment',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Only the fixed capacity-duration factorial is allowed')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pcfg, data, views, predictions, controls, pid = parent.load()
    validate_config(cfg, pcfg)
    prior = json.loads((ROOT/cfg['parent_analysis']).read_text())
    assert prior['identity'] == pid and prior['new_fits'] == 12
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed dependency: '+path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], cfg['parent_analysis'], *CODE): bind(path)
    for name in ('replay.json', 'independent_verification.json', 'fit_and_selection_forensics.json'):
        path = str(Path(cfg['parent_analysis']).parent/name); r = json.loads((ROOT/path).read_text())
        if name != 'fit_and_selection_forensics.json':
            assert r['all_checks_passed'] and r['analysis_sha256'] == bindings[cfg['parent_analysis']]
        else:
            assert r['identity']['analysis_sha256'] == bindings[cfg['parent_analysis']]
            for p, sha in r['identity']['source_bindings'].items(): bind(p, sha)
        bind(path)
    for r in prior['training']: bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in prior['archives']: bind(r['path'], r['sha256'])
    identity = dict(config=cfg, source_bindings=bindings, parent_identity=pid,
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0), closed_role_readout=False)
    assert_current(identity)
    return cfg, pcfg, data, views, predictions, controls, prior, identity


def settings(pcfg, width):
    return pcfg['training']|{'width':64 if width == 'narrow' else 128, 'steps':12000}


def train(cfg, pcfg, data, views, prior, identity, args, beat):
    old = {r['view']:r for r in prior['training']}
    for key, meta in views.items():
        if args.view and args.view != key: continue
        ids,x,y,d,pr = training_data(meta,data)
        for width in ('narrow','wide'):
            if args.width and args.width != width: continue
            ti = dict(identity=identity, view=key, width=width, inputs_sha256=array_hash(ids,x,d),
                labels_sha256=array_hash(y), complete_mask_sha256=array_hash(pr['known']),
                prefix_reference_sha256=old[key]['checkpoint_sha256'], training_sites=pr['training_sites'],
                producers=meta['training_producers'])
            folder=ROOT/cfg['output']/'trials'/key/width; receipt=folder/'complete.json'
            if receipt.exists():
                r=json.loads(receipt.read_text()); assert r['identity']==ti and r['fit']['complete']
                assert file_digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
                if width=='wide': assert file_digest(ROOT/r['prefix_checkpoint'])==r['prefix_checkpoint_sha256']
                beat(state='cached_verified_complete',view=key,width=width); continue
            fit=fit_path(x,y,d,data['sites'][ids],pr,width=width,source=ROOT/old[key]['checkpoint'],
                settings=settings(pcfg,width),identity=ti,seed=meta['seed'],directory=folder,
                heartbeat=lambda **v:beat(view=key,width=width,**v),resume=args.resume,stop_at=args.stop_at)
            if not fit['complete']:
                beat(state='pilot_complete_not_matrix',view=key,width=width,step=fit['step']); return
            cp=folder/'checkpoint.pt'
            result=dict(identity=ti,fit=fit,rows=len(ids),supported_rows=int(pr['known'].sum()),
                checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp))
            if width=='wide':
                prefix=folder/'prefix.pt'
                result.update(prefix_checkpoint=str(prefix.relative_to(ROOT)),prefix_checkpoint_sha256=file_digest(prefix))
            assert_current(identity); immutable_json(receipt,result)
    beat(state='training_call_complete')


def load_states(cfg, pcfg, views, prior, identity):
    records, states = {}, {}; old={r['view']:r for r in prior['training']}
    for key,meta in views.items():
        for width in ('narrow','wide'):
            r=json.loads((ROOT/cfg['output']/'trials'/key/width/'complete.json').read_text())
            assert r['identity']['identity']==identity and r['fit']['complete']
            assert file_digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
            assert cp['identity']==r['identity'] and cp['settings']==settings(pcfg,width)
            assert cp['step']==12000 and cp['seed']==meta['seed'] and cp['loss_exponent']==1
            assert cp['forward_arm']=='bounded_native'
            records[key,width]=r; states[key,width+'_long']=cp
        with_old=torch.load(ROOT/old[key]['checkpoint'],map_location='cpu',weights_only=False)
        wide=records[key,'wide']; assert file_digest(ROOT/wide['prefix_checkpoint'])==wide['prefix_checkpoint_sha256']
        short=torch.load(ROOT/wide['prefix_checkpoint'],map_location='cpu',weights_only=False)
        assert short['step']==3000 and short['identity']==wide['identity']
        assert short['settings']==settings(pcfg,'wide') and short['seed']==meta['seed']
        states[key,'narrow_short'],states[key,'wide_short']=with_old,short
        np.testing.assert_array_equal(short['draws'],with_old['draws'])
        np.testing.assert_array_equal(states[key,'narrow_long']['draws'],states[key,'wide_long']['draws'])
        for arm in ARMS:
            cp=states[key,arm]; pr=cp['preprocess']
            for f in ('mean','std','known','weights','constant'):
                np.testing.assert_array_equal(pr[f],with_old['preprocess'][f])
            for f in ('cost_scale','hard_cut','positive_easy_cut'): assert pr[f]==with_old['preprocess'][f]
            assert cp['draws'][~pr['known']].sum()==0 and cp['draws'].sum()==cp['step']*256
    return records,states


def validate_args(args):
    readout=args.audit_only or args.evaluate or args.verify
    if (sum((args.audit_only,args.evaluate,args.verify))>1 or
            (readout and (args.resume or args.view or args.width or args.stop_at is not None)) or
            (args.stop_at is not None and (not args.view or not args.width or
                not (3000 if args.width=='narrow' else 0)<args.stop_at<=12000))):
        raise ValueError('Separate phases; explicit view/width and valid incremental pilot required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only','resume','evaluate','verify'): p.add_argument('--'+name,action='store_true')
    p.add_argument('--view'); p.add_argument('--width',choices=('narrow','wide')); p.add_argument('--stop-at',type=int)
    args=p.parse_args(); validate_args(args)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg=json.loads((ROOT/CONFIG).read_text()); root=ROOT/cfg['output']; root.mkdir(parents=True,exist_ok=True)
    def beat(**v):
        event=dict(pid=os.getpid(),timestamp_unix=time.time(),**v); json_write(root/'heartbeat.json',event)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,pcfg,data,views,predictions,controls,prior,identity=load()
        if args.view and args.view not in views: raise ValueError('Unregistered view')
        immutable_json(root/'identity.json',identity)
        if args.audit_only: beat(state='preflight_pass',bindings=len(identity['source_bindings']),new_paths=12,continued_paths=12)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_cost_capacity_eval import evaluate
            records,states=load_states(cfg,pcfg,views,prior,identity)
            evaluate(cfg,data,views,predictions,controls,prior,identity,records,states,beat,args.verify)
        else: train(cfg,pcfg,data,views,prior,identity,args,beat)


if __name__=='__main__': main()
