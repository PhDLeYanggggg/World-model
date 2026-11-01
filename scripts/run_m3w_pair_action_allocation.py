"""Registered real seven-action pair allocation with unchanged controls/population."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_coupled_benefit_risk as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_pair_action_allocation as api
from src.evaluation import m3w_pair_action_readout as evaluation
from src.evaluation.m3w_metric_bundle import pack

np = api.np; source = previous.source; unit = previous.unit
NAME = 'european_pair_action_allocation_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_pair_action_allocation_v1.json'; REG = PUBLIC/'registration.json'
sha, once, write = previous.sha, previous.once, previous.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_pair_action_allocation.py',
    ROOT/'src/evaluation/m3w_pair_action_readout.py', ROOT/'tests/test_m3w_pair_action_allocation.py', PUBLIC/'protocol.md')


def admission(register=False):
    _, values, store, banks, models, controls = previous.admission()
    cfg = json.loads(CONFIG.read_text()); verified = json.loads((previous.PUBLIC/'verification.json').read_text())
    if (cfg['seeds'] != [17,29,43] or cfg['risk_budget'] != .02 or cfg['pair_weight'] != 1.
            or cfg['past_nearest_neighbors'] != 4 or cfg['new_training_updates']
            or not verified['all72groups_exact'] or verified['summary_sha256'] != sha(previous.PUBLIC/'summary.json')
            or any(cfg[k] for k in ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Frozen original controls,task and risk contract required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        source_verification_sha256=sha(previous.PUBLIC/'verification.json'),
        source_training_freeze_sha256=sha(unit.PUBLIC/'training_freeze.json'),
        contexts=json.loads(unit.REG.read_text())['contexts'], changed_factor='predicted_path_pair_objective_only',
        role='exposed_SOURCE_fitting_resubstitution', inference_labels_used=False, new_training_updates=0,
        own_policy_outcomes_read=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Registered scientific definition changed:'+rel)
            source.frozen.BASE.engine.committed(ROOT/rel)
        source.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models, controls


def runtime(cfg):
    unit.api.parent.torch.set_num_threads(4); unit.api.parent.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume fixed pair decoder')))
    def beat(**event):
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']:
            raise OSError('Private progress cap')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, interop_threads=1, workers=0)
    return start, beat


def infer(context, seed, packets, changed, values, store, models, controls, old, main, beat):
    data, ranges, _ = source.prepare(context, seed, packets, changed, store, models, old, main, beat)
    state = unit.api.parent.read_checkpoint(ROOT/controls[context,seed]['path'])
    return unit.api.parent.predict(state, data['causal']), ranges


def decode(packet, prediction, origins, parent, cfg, *, checkpoint=lambda _: None, heartbeat=lambda **_:None, resume_state=None):
    return api.allocate(prediction, packet['allowed'], packet['recordings'], packet['frames'], packet['ids'],
        packet['rollouts'], packet['floor_rollout'], origins, packet['scales'], parent,
        weight=cfg['pair_weight'], neighbors=cfg['past_nearest_neighbors'], seconds=cfg['solver_seconds_per_query'],
        checkpoint=checkpoint, heartbeat=heartbeat, resume_state=resume_state)


def pilot():
    cfg, values, store, banks, models, controls = admission(); start, beat = runtime(cfg)
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    old, main = aggregate_groups(source.frozen.source.parent.NAME), aggregate_groups(source.frozen.NAME)
    context, items = sorted(values[2]['contexts'].items())[0]
    packets, changed = previous.context_packets(context, items, values, store, banks, beat)
    predictions, ranges = infer(context,17,packets,changed,values,store,models,controls,old,main,beat)
    name = sorted(packets)[0]; packet = packets[name]; p = predictions[ranges[name]]
    key = f'{context}_{name}_seed17_fit.json'
    control = unit.api.parent.read_checkpoint(unit.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['set_joint']
    reference = aggregate_groups(unit.NAME)[key]
    if unit.api.parent.digest(p) != reference['prediction_sha256']['selected_set']: raise ValueError('Old prediction changed')
    groups = api.base.query_groups(packet['recordings'],packet['frames'],packet['ids'])
    select = sorted(set(range(min(cfg['pilot_prefix_queries'],len(groups)))) |
        set(sorted(range(len(groups)),key=lambda i:(-len(groups[i]),i))[:cfg['pilot_largest_queries']]))
    ix = np.sort(np.concatenate([groups[i] for i in select])); small = {k:packet[k][ix] for k in
        ('allowed','recordings','frames','ids','rollouts','floor_rollout','scales')}
    origins = values[3]['origin'][small['ids']]; path = PRIVATE/'progress/pilot.pt.gz'
    class PlannedInterruption(Exception): pass
    def interrupt(state): unit.api.parent.save_checkpoint(path,state); raise PlannedInterruption()
    try: decode(small,p[ix],origins,control[ix],cfg,checkpoint=interrupt)
    except PlannedInterruption: pass
    resumed = decode(small,p[ix],origins,control[ix],cfg,resume_state=unit.api.parent.read_checkpoint(path), heartbeat=lambda **kw:beat(state='pilot_resumed',**kw))
    began = time.monotonic(); direct = decode(small,p[ix],origins,control[ix],cfg, heartbeat=lambda **kw:beat(state='pilot_direct',**kw))
    seconds = time.monotonic()-began; unit.api.parent.exact(resumed,direct)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        direct_decoder_seconds=seconds,queries=len(select),rows=len(ix),maximum_agents=max(len(groups[i]) for i in select),
        rough112596query_projection_seconds=seconds/len(select)*112596,
        projection_is_one_source_and_size_biased_not_completion=True,diagnostics=direct[2],
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        exact_interrupted_recovery=True,old_prediction_control_exact=True,outcomes_scored=False,
        engineering_pass=True,registration_sha256=sha(REG)))
    beat(state='pilot_complete_no_outcome_scoring')


def evaluate(verify=False,resume=False):
    cfg, values, store, banks, models, controls = admission(); start, beat = runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed readout immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real recovery pilot required')
    old, main = aggregate_groups(source.frozen.source.parent.NAME), aggregate_groups(source.frozen.NAME)
    original = aggregate_groups(unit.NAME); previous_rows = aggregate_groups(NAME) if verify else {}
    rows, refs, all_ids, records = [], [], set(), set()
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed = previous.context_packets(context,items,values,store,banks,beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            prediction,ranges = infer(context,seed,packets,changed,values,store,models,controls,old,main,beat)
            for name,packet in sorted(packets.items()):
                key=f'{context}_{name}_seed{seed}_fit.json'; p=prediction[ranges[name]]; path=PUBLIC/'groups'/key
                oldrow=original[key]; cp=PRIVATE/'progress'/(key+'.pt.gz')
                control=unit.api.parent.read_checkpoint(unit.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['set_joint']
                for value,digest in ((p,oldrow['prediction_sha256']['selected_set']),(packet['ids'],oldrow['ids_sha256']),
                        (control,oldrow['action_sha256']['set_joint'])):
                    if unit.api.parent.digest(value) != digest: raise ValueError('Exact original controls required')
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit resume required')
                    row=json.loads(path.read_text())
                    if row['registration_sha256'] != sha(REG): raise ValueError('Partial registration changed')
                else:
                    origins=values[3]['origin'][packet['ids']]
                    choices,ranks,info=decode(packet,p,origins,control,cfg,
                        checkpoint=(lambda _:None) if verify else lambda state:unit.api.parent.save_checkpoint(cp,state),
                        heartbeat=lambda **kw:beat(state='pair_allocation',context=context,source=name,seed=seed,**kw),
                        resume_state=unit.api.parent.read_checkpoint(cp) if resume and cp.exists() and not verify else None)
                    result=evaluation.readout(packet,choices,ranks,info,job['design']['easy_cut'],origins)
                    if result['policies']['unary_joint'] != oldrow['result']['policies']['set_joint']:
                        raise ValueError('Original full-policy score changed')
                    row=dict(context=context,source=name,seed=seed,result=result,old_control_exact=True,
                        prediction_sha256=unit.api.parent.digest(p),ids_sha256=oldrow['ids_sha256'],
                        action_sha256={a:unit.api.parent.digest(c) for a,c in choices.items()},
                        registration_sha256=sha(REG),held_scored=False,independent_roles_opened=False)
                    if verify:
                        if row != previous_rows[key]: raise ValueError('Complete redecoding/readout differs:'+key)
                    else: once(path,row)
                all_ids.update(map(int,packet['ids'])); records.update((name,str(r)) for r in packet['recordings'])
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='pair_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
            del prediction
        del packets,changed
    if len(all_ids)!=149510 or len(records)!=74: raise ValueError('Complete unchanged original cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg)
    summary.update(unique_target_rows=len(all_ids),unique_recordings=len(records))
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary replay differs')
        once(PUBLIC/'verification.json',dict(all72groups_exact=True,all72_old_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary)
        once(PUBLIC/'complete.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=refs,
            registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json')))
        pack(ROOT,PUBLIC)
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),dict(pid=os.getpid(),
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_verified' if verify else 'all72_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('phase',choices=('register','pilot','eval','verify'))
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    (PRIVATE/'progress').mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register': admission(True)
        elif args.phase=='pilot': pilot()
        else: evaluate(args.phase=='verify',args.resume)
