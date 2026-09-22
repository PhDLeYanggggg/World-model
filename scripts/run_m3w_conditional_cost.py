"""Fixed decision-region loss weighting with unchanged inference policies."""
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
    raise RuntimeError('Native arm64 required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_cost_budget_matched as parent
from scripts.run_m3w_bounded_cost import training_data
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_conditional_cost_head import fit, region_weights
from src.evaluation.m3w_conditional_cost_audit import strict_bits
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch

CONFIG = 'configs/m3w_conditional_cost_v1.json'
CODE = ('scripts/run_m3w_conditional_cost.py', 'scripts/verify_m3w_conditional_cost.py',
        'src/world_model/m3w_conditional_cost_head.py', 'src/evaluation/m3w_conditional_cost_eval.py',
        'tests/test_m3w_conditional_cost_head.py', 'tests/test_m3w_conditional_cost_protocol.py')


def validate_config(cfg, pcfg):
    if (cfg['sites'] != pcfg['sites'] or cfg['seeds'] != pcfg['seeds']
            or cfg['training'] != pcfg['training'] or cfg['loss_exponent'] != 1
            or cfg['region_multiplier'] != 4. or cfg['region_reference'] != 'tempered_strict_stop'
            or cfg['primary_reference'] != 'tempered_strict_stop' or cfg['policies'] != pcfg['policies']
            or cfg['matched_count_reference'] != pcfg['matched_count_reference']
            or cfg['bootstrap_resamples'] != 3000 or any(cfg[k] for k in (
                'threshold_search', 'model_selection', 'closed_role_readout', 'risk_calibration',
                'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Only fixed fourfold fit-region weighting is allowed')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pcfg, data, views, predictions, _, _, refs, pid = parent.load()
    records, states = parent.load_states(pcfg, views, refs, pid)
    validate_config(cfg, pcfg)
    prior = json.loads((ROOT/cfg['parent_analysis']).read_text())
    assert prior['identity'] == pid and prior['new_updates'] == 288000
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed dependency: '+path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], cfg['parent_analysis'], cfg['diagnostic'], *CODE):
        bind(path)
    diagnostic = json.loads((ROOT/cfg['diagnostic']).read_text())
    assert diagnostic['analysis_sha256'] == bindings[cfg['parent_analysis']]
    for path, sha in diagnostic['identity']['source_bindings'].items():
        bind(path, sha)
    for name in ('replay.json', 'independent_verification.json'):
        path = str(Path(cfg['parent_analysis']).parent/name)
        receipt = json.loads((ROOT/path).read_text())
        assert receipt['all_checks_passed'] and receipt['analysis_sha256'] == bindings[cfg['parent_analysis']]
        bind(path)
    for r in records.values():
        bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in prior['archives']:
        bind(r['path'], r['sha256'])
    identity = dict(source_bindings=bindings, config=cfg, parent_identity=pid,
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        closed_role_readout=False)
    assert_current(identity)
    return cfg, data, views, predictions, prior, states, identity


def weighted_data(meta, data, reference, cfg):
    ids, x, y, d, pr = training_data(meta, data)
    model = build(x.shape[1], cfg['training']['width'], meta['seed'])
    model.load_state_dict(reference['model'])
    score = predict(model, x, d, pr, 'bounded_native')
    bits = strict_bits(score, data['geometry'][ids, :16].reshape(-1, 8, 2), d)
    w = region_weights(bits, pr, cfg['region_multiplier'])
    return ids, x, y, d, pr, bits, w, score


def check_state(cp, reference, w, cfg):
    assert cp['settings'] == reference['settings'] == cfg['training'] and cp['step'] == reference['step'] == 12000
    assert cp['seed'] == reference['seed'] and cp['loss_exponent'] == 1 and cp['forward_arm'] == 'bounded_native'
    np.testing.assert_array_equal(cp['loss_weights'], w)
    np.testing.assert_array_equal(cp['draws'], reference['draws'])
    for k in ('mean', 'std', 'known', 'weights', 'constant'):
        np.testing.assert_array_equal(cp['preprocess'][k], reference['preprocess'][k])
    for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
        assert cp['preprocess'][k] == reference['preprocess'][k]
    assert cp['draws'].sum() == 12000*256 and cp['draws'][~cp['preprocess']['known']].sum() == 0


def train(cfg, data, views, references, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        ids,x,y,d,pr,bits,w,_ = weighted_data(meta,data,references[key,'tempered'],cfg)
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids,x,d),
            labels_sha256=array_hash(y), region_sha256=array_hash(ids,bits,w),
            complete_mask_sha256=array_hash(pr['known']), training_sites=pr['training_sites'],
            producers=meta['training_producers'])
        folder = ROOT/cfg['output']/'trials'/key
        receipt = folder/'complete.json'
        if receipt.exists():
            r = json.loads(receipt.read_text())
            assert r['identity'] == ti and r['fit']['complete']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            beat(state='cached_verified_complete',view=key)
            continue
        _, result = fit(x,y,d,data['sites'][ids],pr,w,seed=meta['seed'],settings=cfg['training'],
            identity=ti,directory=folder,resume=args.resume,stop_at=args.stop_at,
            heartbeat=lambda **v:beat(view=key,**v))
        if not result['complete']:
            beat(state='pilot_complete_not_matrix',view=key,step=result['step'])
            return
        path=folder/'checkpoint.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        check_state(cp,references[key,'tempered'],w,cfg)
        assert_current(identity)
        immutable_json(receipt,dict(identity=ti,fit=result,rows=len(ids),supported_rows=int(pr['known'].sum()),
            emphasized_complete_rows=int((bits & pr['known']).sum()),
            mean_weight_under_sampler=float(np.dot(pr['weights'],w)),
            checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=file_digest(path)))
    beat(state='training_call_complete')


def load_states(cfg,views,identity):
    records,states={},{}
    for key in views:
        r=json.loads((ROOT/cfg['output']/'trials'/key/'complete.json').read_text())
        assert r['identity']['identity'] == identity and r['fit']['complete']
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        cp=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['identity'] == r['identity'] and cp['step'] == 12000
        records[key],states[key]=r,cp
    return records,states


def validate_args(args):
    if (sum((args.audit_only,args.evaluate,args.verify))>1
            or ((args.audit_only or args.evaluate or args.verify) and (args.resume or args.view or args.stop_at is not None))
            or (args.stop_at is not None and (not args.view or not 0<args.stop_at<=12000))):
        raise ValueError('Separate audit/train/readout phases and explicit pilot required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only','evaluate','verify','resume'):
        p.add_argument('--'+name,action='store_true')
    p.add_argument('--view');p.add_argument('--stop-at',type=int)
    args=p.parse_args();validate_args(args)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=json.loads((ROOT/CONFIG).read_text());root=ROOT/cfg['output'];root.mkdir(parents=True,exist_ok=True)
    def beat(**v):
        event=dict(pid=os.getpid(),timestamp_unix=time.time(),**v)
        json_write(root/'heartbeat.json',event)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg,data,views,predictions,prior,refs,identity=load()
        if args.view and args.view not in views:raise ValueError('Unregistered view')
        immutable_json(root/'identity.json',identity)
        if args.audit_only:beat(state='preflight_pass',bindings=len(identity['source_bindings']),new_fits=12)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_conditional_cost_eval import evaluate
            records,states=load_states(cfg,views,identity)
            evaluate(cfg,data,views,predictions,prior,refs,identity,records,states,beat,args.verify)
        else:train(cfg,data,views,refs,identity,args,beat)


if __name__ == '__main__':main()
