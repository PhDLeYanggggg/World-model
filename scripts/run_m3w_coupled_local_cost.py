"""Registered coupled local-cost control, full fitting readout and exact replay."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_completion_cost as parent
from src.world_model import m3w_coupled_local_cost as api
from src.evaluation import m3w_completion_cost_readout as score
from src.evaluation import m3w_multiaction_cost_readout as base

NAME = 'european_coupled_local_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
REG = PUBLIC/'registration.json'; CONFIG = ROOT/'configs/m3w_coupled_local_cost_v1.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_coupled_local_cost.py',
         ROOT/'tests/test_m3w_coupled_local_cost.py', PUBLIC/'protocol.md')
sha, once, write = parent.sha, parent.once, parent.json_write


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, values = parent.admission(); learned = parent.models()
    check = json.loads((parent.PUBLIC/'verification.json').read_text())
    if (cfg['neighbors'] != 32 or cfg['modes'] != list(api.MODES) or cfg['risk_budget'] != .02
            or cfg['held_scored'] or cfg['independent_roles_opened']
            or not check['exact_replay'] or check['summary_sha256'] != sha(parent.PUBLIC/'summary.json')):
        raise ValueError('Fixed local-cost contract and verified previous readout required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_summary_sha256=sha(parent.PUBLIC/'summary.json'), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        contexts=sorted(values[2]['contexts']), required_groups=288, neighbors=32,
        scope='original_fitting_only_other_recording_donors', new_optimizer_updates=0,
        new_retrieval_models_are_deterministic=True, comparator_seeds=cfg['comparison_seeds'],
        whole_pipeline_OOF=False, independent_roles_opened=False, deployment_changed=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest:
                raise ValueError('Registered coupled local control changed')
            parent.parent.parent.committed(ROOT/rel)
        parent.parent.parent.committed(REG)
    return cfg, values, learned


def runtime(cfg):
    parent.torch.set_num_threads(4); parent.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic(); last_print = 0.
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in (PRIVATE/'neighbors').glob('*.npz')) > cfg['neighbor_cache_cap_bytes']:
            raise OSError('Private neighbor checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume the identical full roster')
    def beat(**event):
        nonlocal last_print
        guard(); now = time.monotonic()
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', value)
        if event.get('state') != 'neighbors' or now-last_print > 30:
            with (PRIVATE/'events.jsonl').open('a') as f:
                f.write(json.dumps(value)+'\n')
            print(json.dumps(value), flush=True); last_print = now
    guard(); beat(state='runtime_ready', architecture=platform.machine(), cpu_threads=4, workers=0, resource_probing=False)
    return started, beat, guard


def arrays(packets, preprocess):
    names = sorted(packets); spans = {}; offset = 0
    for name in names:
        spans[name] = offset, offset+len(packets[name]['ids']); offset = spans[name][1]
    fields = ('envelope', 'y', 'scales', 'allowed', 'sites', 'ids')
    bank = {key: np.concatenate([packets[s][key] for s in names]) for key in fields}
    bank['recordings'] = np.concatenate([np.array([json.dumps([s, str(r)]) for r in packets[s]['recordings']]) for s in names])
    bank['z'] = np.concatenate([api.descriptor(packets[s]['x'], packets[s]['scene'], preprocess) for s in names])
    bank['causal'] = bank['allowed'].any(1)
    bank['target'], bank['known'] = api.encode(bank['y'], bank['envelope'], bank['scales'])
    return bank, spans


def retrieve(bank, context, cfg, beat, *, verify=False, resume=False, query_ids=None):
    folder = PRIVATE/'neighbors'; folder.mkdir(exist_ok=True)
    binding = dict(registration_sha256=sha(REG), descriptor_sha256=parent.parent.fingerprint(bank['z']),
        recording_sha256=parent.parent.fingerprint(bank['recordings']), causal_sha256=parent.parent.fingerprint(bank['causal']))
    def path(recording):
        return folder/(context+'_'+hashlib.sha256(recording.encode()).hexdigest()[:24]+'.npz')
    def metadata(ix, donors):
        return dict(binding, query_sha256=parent.parent.fingerprint(ix), donor_sha256=parent.parent.fingerprint(donors))
    def cached(recording, ix, donors):
        p = path(recording)
        if not p.exists() or verify or not resume:
            return None
        with np.load(p, allow_pickle=False) as state:
            if json.loads(str(state['metadata'])) != metadata(ix, donors):
                raise ValueError('Resumed neighbor binding changed')
            return state['neighbors'].copy()
    def store(recording, ix, donors, chosen):
        if query_ids is not None:
            return
        p = path(recording); meta = metadata(ix, donors)
        if p.exists():
            with np.load(p, allow_pickle=False) as old:
                if json.loads(str(old['metadata'])) != meta or not np.array_equal(old['neighbors'], chosen):
                    raise ValueError('Fresh exact neighbor replay differs')
        elif verify:
            raise ValueError('Cannot verify missing neighbor checkpoint')
        else:
            temp = p.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, neighbors=chosen, metadata=json.dumps(meta, sort_keys=True))
            os.replace(temp, p)
    return api.retrieve(bank['z'], bank['recordings'], bank['causal'], query_ids=query_ids,
        k=cfg['neighbors'], chunk=cfg['query_chunk'], heartbeat=beat, cached=cached, store=store)


def cost_predictions(bank, queries, neighbors):
    return api.predictions(bank['target'], bank['known'], bank['allowed'], bank['sites'],
        bank['recordings'], bank['causal'], queries, neighbors, bank['envelope'], bank['scales'])


def summarize(rows, cfg, contexts, parent_cfg):
    expected = {(context, item['packet']['identity']['source'], seed, mode)
        for context, items in contexts.items() for item in items
        for seed in cfg['comparison_seeds'] for mode in api.MODES}
    keys = [(r['context'], r['source'], r['seed'], r['mode']) for r in rows]
    if len(keys) != 288 or len(set(keys)) != 288 or set(keys) != expected:
        raise ValueError('Complete288 fitting control groups required')
    output = {}
    for mode in api.MODES:
        group = [r for r in rows if r['mode'] == mode]
        output[mode] = dict(safety={n: score.safety(group, n) for n in group[0]['result']['policies']},
            accuracy={n: base.interval(group, lambda r, name=n:r['policies'][name]['known_ADE_gain_floor_percent'], parent_cfg)
                for n in group[0]['result']['policies']},
            FDE={n: base.interval(group, lambda r, name=n:r['policies'][name]['known_FDE_gain_floor_percent'], parent_cfg)
                for n in ('physical_only', 'physical_neural')},
            comparisons={key: {m+'_'+b: base.interval(group,
                lambda r,k=key,mode=m,bound=b:r['comparisons'][k][mode][bound], parent_cfg)
                for m in ('full', 'matched') for b in ('known_pp', 'lower_pp', 'upper_pp')}
                for key in ('physical_neural_minus_physical_only',)},
            prior_known_change={m+'_'+b: base.interval(group,
                lambda r,mode=m,bound=b:r['prior_known_change'][mode][bound], parent_cfg)
                for m in ('full', 'matched') for b in ('known_pp', 'lower_pp', 'upper_pp')},
            subsets=score.pooled_subsets(group))
    primary = [r for r in rows if r['mode'] == cfg['primary']]
    local_global = {m+'_'+b: base.interval(primary,
        lambda r,mode=m,bound=b:r['global_to_local'][mode][bound], parent_cfg)
        for m in ('full', 'matched') for b in ('known_pp', 'lower_pp', 'upper_pp')}
    p = output[cfg['primary']]; s = p['safety']['physical_neural']
    positive = lambda c:c['CI95'] is not None and c['CI95'][0] > 0
    gates = dict(full288groups=True,
        complete_mixed_record_risk=s['supported_views']==s['required_views'] and s['supported_record_views']==s['required_record_views'],
        full_prior_change_lower_CI_positive=positive(p['prior_known_change']['full_lower_pp']),
        matched_prior_change_lower_CI_positive=positive(p['prior_known_change']['matched_lower_pp']),
        neural_increment_lower_CI_positive=positive(p['comparisons']['physical_neural_minus_physical_only']['full_lower_pp']),
        local_over_global_lower_CI_positive=positive(local_global['full_lower_pp']))
    return dict(result_source='fresh_fitting_only_coupled_local_cost_readout', modes=output,
        local_over_global=local_global, fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        new_optimizer_updates=0, deterministic_policy_repeated_across_comparator_seeds=True,
        whole_pipeline_OOF=False, held_scored=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def execute(phase, *, resume=False):
    cfg, values, learned = admission(); verify = phase == 'verify'; pilot = phase == 'pilot'
    finish = PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not verify:
        raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real retrieval/replay pilot required')
    started, beat, guard = runtime(cfg); rows = []; refs = []
    _, origin_refs = parent.parent.parent.old.direct.ORIGINS.models(values[2])
    for context, items in sorted(values[2]['contexts'].items())[:1] if pilot else sorted(values[2]['contexts'].items()):
        packets = parent.packets(context, items, values, beat)
        ref = learned[context, 17, 'physical_neural', 'known_only']['checkpoint']
        state = parent.api.base.read_checkpoint(ROOT/ref['path'])
        prepared = parent.api.training_data(packets, parent.parent)
        parent.api.base.exact(prepared['preprocess'], state['preprocess'])
        parent.api.base.exact(prepared['input_hashes'], state['input_hashes']); del prepared
        bank, spans = arrays(packets, state['preprocess'])
        if pilot:
            first = sorted(set(bank['recordings'][bank['causal']]))[0]
            ix = np.flatnonzero(bank['causal'] & (bank['recordings'] == first))[:cfg['pilot_queries']]
            began = time.monotonic(); q, nn, _ = retrieve(bank, context, cfg, beat, query_ids=ix)
            seconds = time.monotonic()-began
            _, repeat, _ = retrieve(bank, context, cfg, beat, query_ids=ix)
            np.testing.assert_array_equal(nn, repeat)
            p, supported = cost_predictions(bank, q, nn)
            once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started,
                queries=len(q), first_lookup_seconds=seconds, exact_neighbor_repeat=True,
                full_context_causal_rows=int(bank['causal'].sum()),
                projected_six_context_lookup_seconds=seconds*int(bank['causal'].sum())/len(q)*6,
                peak_RSS_bytes=parent.rss_bytes(), engineering_pass=True,
                finite_all_modes=all(np.isfinite(a).all() for a in p.values()),
                unsupported_actions={m:int((~a).sum()) for m,a in supported.items()},
                query_outcomes_scored=False, scientific_gate_evaluated=False))
            beat(state='pilot_complete'); return
        q, nn, references = retrieve(bank, context, cfg, beat, verify=verify, resume=resume)
        pred, supported = cost_predictions(bank, q, nn)
        predictions = {m: np.zeros((len(bank['ids']), 7, 5)) for m in api.MODES}
        support = {m: np.zeros((len(bank['ids']), 7), bool) for m in api.MODES}
        for mode in api.MODES:
            predictions[mode][q] = pred[mode]; support[mode][q] = supported[mode]
        freeze = dict(descriptor_sha256=parent.parent.fingerprint(bank['z']),
            neighbor_sha256=parent.parent.fingerprint(nn), query_sha256=parent.parent.fingerprint(q),
            prediction_sha256={m:parent.parent.fingerprint(p) for m,p in predictions.items()},
            recording_references=references, causal_rows=len(q), query_outcomes_used=False,
            no_new_optimizer_updates=True)
        once(PUBLIC/'prediction_freeze'/f'{context}.json', freeze)
        job = next(j for j in values[4] if j['key']==context.split('_controller')[0])
        for seed in cfg['comparison_seeds']:
            states = {(b,m):parent.api.base.read_checkpoint(ROOT/learned[context,seed,b,m]['checkpoint']['path'])
                      for b,m in parent.cells(json.loads(parent.CONFIG.read_text()))}
            origin = parent.parent.parent.old.base.core.read_checkpoint(ROOT/origin_refs[context,seed,'self']['checkpoint']['path'])
            originals = parent.parent.parent.old.direct.origin_predictions(origin, {s:p['args'] for s,p in packets.items()})
            for source, packet in sorted(packets.items()):
                guard(); lo,hi = spans[source]; kw = packet['x'],packet['scene'],packet['envelope'],packet['scales']
                previous = {key:parent.api.predict(s,*kw) for key,s in states.items()}
                oldchoices = {key:parent.api.base.select_actions(p,packet['allowed'],arm=key[0]) for key,p in previous.items()}
                localchoices = {m:{b:parent.api.base.select_actions(predictions[m][lo:hi],packet['allowed'] & support[m][lo:hi],arm=b)
                                  for b in parent.api.base.ARMS} for m in api.MODES}
                op = originals[source]
                take = parent.parent.parent.old.base.actions(op,parent.parent.parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:,6]
                for mode in api.MODES:
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{mode}_fit.json'
                    current = {b:v[0] for b,v in localchoices[mode].items()}
                    rank = {b:v[1] for b,v in localchoices[mode].items()}
                    for (b,m),(c,r) in oldchoices.items():
                        current[m+'_'+b],rank[m+'_'+b] = c,r
                    current['frozen_origin_binary'],rank['frozen_origin_binary'] = np.where(take,6,-1),op[:,0]-op[:,1]
                    current['floor'],rank['floor'] = np.full(len(take),-1,np.int64),np.zeros(len(take))
                    result = base.group_readout(packet,current,rank,{b:predictions[mode][lo:hi] for b in parent.api.BANKS})
                    for m in parent.api.MODES:
                        old = json.loads((parent.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{m}_fit.json').read_text())
                        for b in parent.api.BANKS:
                            if (result['policies'][m+'_'+b] != old['result']['policies'][b]
                                    or parent.parent.fingerprint(current[m+'_'+b]) != old['action_sha256'][b]):
                                raise ValueError('Original paired cost policy changed')
                    result['prior_known_change'] = score.paired(packet,current['known_only_physical_neural'],current['physical_neural'],
                                                               rank['known_only_physical_neural'],rank['physical_neural'])
                    global_choice,global_rank = localchoices['global_bounds']['physical_neural']
                    result['global_to_local'] = score.paired(packet,global_choice,current['physical_neural'],global_rank,rank['physical_neural'])
                    result['proxy_diagnostics'] = {b:score.proxy_diagnostics(packet,predictions[mode][lo:hi],b) for b in parent.api.BANKS}
                    result['subsets'] = score.subsets(packet,current,job['design']['easy_cut'])
                    value = dict(context=context,source=source,seed=seed,mode=mode,scope='fit',result=result,
                        prediction_freeze_sha256=sha(PUBLIC/'prediction_freeze'/f'{context}.json'),
                        action_sha256={n:parent.parent.fingerprint(c) for n,c in current.items()},
                        result_source='fresh_coupled_local_fitting_control',held_scored=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Full coupled local readout replay differs')
                    else:
                        once(path,value)
                    rows.append(value);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            beat(state='fitting_groups',context=context,seed=seed,groups=len(rows),total=288)
        del packets,bank,pred,predictions,support
    summary = summarize(rows,cfg,values[2]['contexts'],values[1])
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or refs != json.loads(finish.read_text())['groups']:
            raise ValueError('Complete coupled local summary replay differs')
        once(PUBLIC/'verification.json',dict(exact_replay=True,groups=288,summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json',summary)
        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-started,peak_RSS_bytes=parent.rss_bytes(),groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'),held_scored=False,independent_roles_opened=False))
    write(PRIVATE/('verification_runtime.json' if verify else 'runtime.json'),
          dict(pid=os.getpid(),seconds=time.monotonic()-started,peak_RSS_bytes=parent.rss_bytes()))
    beat(state='verified' if verify else 'complete',groups=288,fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':
            admission(register=True)
        else:
            execute(args.phase,resume=args.resume)
