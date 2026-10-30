"""Registered physical6/neural7 reoptimization under a shared causal query count."""
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
    raise RuntimeError('Native arm64 required before Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_frame_action_allocation as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_shared_count_bank as api
from src.world_model.m3w_european_source_forecast import BASELINES
from src.evaluation import m3w_shared_count_readout as evaluation

NAME = 'european_shared_count_bank_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_shared_count_bank_v1.json'
REG = PUBLIC / 'registration.json'
sha, once, write = parent.sha, parent.once, parent.write
FILES = (Path(__file__), CONFIG, ROOT / 'src/world_model/m3w_shared_count_bank.py',
    ROOT / 'src/evaluation/m3w_shared_count_readout.py', ROOT / 'tests/test_m3w_shared_count_bank.py',
    ROOT / 'tests/test_m3w_shared_count_readout.py', PUBLIC / 'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, values, store, banks, models = parent.admission()
    check = json.loads((parent.PUBLIC / 'verification.json').read_text())
    if (cfg['physical_actions'] != list(BASELINES) or cfg['seeds'] != [17, 29, 43]
            or cfg['risk_budget'] != .02 or cfg['groups'] != 72 or cfg['new_training_updates']
            or cfg['count_anchor'] != 'physical_independent_each_current_query'
            or not check['all72groups_exact'] or check['summary_sha256'] != sha(parent.PUBLIC / 'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified fixed bank and closed fitting roles required')
    for rel, digest in json.loads(parent.REG.read_text())['bindings'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Frozen parent allocation changed: '+rel)
    diagnosis = parent.PUBLIC.parent / 'european_frame_allocation_attribution_v1'
    dcheck = json.loads((diagnosis / 'verification.json').read_text())
    if not dcheck['all72groups_exact'] or dcheck['summary_sha256'] != sha(diagnosis / 'summary.json'):
        raise ValueError('Complete verified prior attribution required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_complete_sha256=sha(parent.PUBLIC / 'complete.json'),
        parent_summary_sha256=sha(parent.PUBLIC / 'summary.json'), parent_verification_sha256=sha(parent.PUBLIC / 'verification.json'),
        attribution_summary_sha256=sha(diagnosis / 'summary.json'), attribution_verification_sha256=sha(diagnosis / 'verification.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'], cached_verified_cost_heads=18,
        primary='neural_joint_minus_physical_joint_at_common_physical_query_count',
        new_training_updates=0, common_count='physical_independent_each_query',
        old_joint_choices_cached_verified_not_resolved_again=True, risk_budget=.02,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered shared-count comparison changed: '+rel)
            parent.BASE.engine.committed(ROOT / rel)
        parent.BASE.engine.committed(REG)
    return cfg, values, store, banks, models


def runtime(cfg):
    parent.BASE.torch.set_num_threads(4); parent.BASE.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume fixed shared-count checkpoints')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume original roster')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_progress_cap_bytes']:
            raise OSError('Private progress cap exceeded')
    def beat(**event):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', value)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), threads=4, workers=0, device='cpu')
    return start, beat, guard


def pilot():
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'pilot.json').exists():
        raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, packets, _, _ = parent.BASE.prepared_context(context, items, values, store, banks, beat)
    name = sorted(packets)[0]; packet = packets[name]
    reference = aggregate_groups(parent.source.parent.NAME)[f'{context}_{name}_seed17_fit.json']
    state = parent.BASE.api.completion.base.read_checkpoint(ROOT / models[context, 17, 'admitted_score']['checkpoint']['path'])
    p, _, _ = parent.infer(packet, state, store, reference)
    groups = api.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    selected = sorted(set(range(min(cfg['pilot_prefix_queries'], len(groups)))) |
        set(sorted(range(len(groups)), key=lambda i: (-len(groups[i]), i))[:cfg['pilot_largest_queries']]))
    at = np.sort(np.concatenate([groups[i] for i in selected]))
    args = (p[at], packet['allowed'][at], packet['recordings'][at], packet['frames'][at], packet['ids'][at])
    cp = PRIVATE / 'pilot_progress.pt.gz'
    class PlannedInterruption(Exception):
        pass
    def stop(value):
        parent.BASE.api.completion.base.save_checkpoint(cp, value)
        raise PlannedInterruption()
    try:
        api.allocate(*args, seconds=cfg['solver_seconds_per_query'], checkpoint=stop)
    except PlannedInterruption:
        pass
    resumed = api.allocate(*args, seconds=cfg['solver_seconds_per_query'],
        resume_state=parent.BASE.api.completion.base.read_checkpoint(cp), heartbeat=lambda **kw: beat(state='pilot_resumed', **kw))
    began = time.monotonic()
    direct = api.allocate(*args, seconds=cfg['solver_seconds_per_query'], heartbeat=lambda **kw: beat(state='pilot_direct', **kw))
    direct_seconds = time.monotonic()-began
    parent.BASE.api.completion.base.exact(resumed, direct)
    once(PUBLIC / 'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        context=context, source=name, seed=17, causal_queries=len(selected), rows=len(at),
        direct_seconds=direct_seconds, exact_interrupted_recovery=True, source_queries=len(groups),
        rough72group_projection_seconds=direct_seconds/len(selected)*len(groups)*72,
        projection_based_on_one_source_large_queries_not_full_roster=True,
        diagnostics=direct[2], prediction_control_exact=True, outcome_scoring=False,
        new_training_updates=0, registration_sha256=sha(REG), engineering_pass=True))
    beat(state='real_shared_count_pilot_complete_no_outcome_scoring')


def execute(verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'complete.json').exists() and not verify:
        raise ValueError('Completed comparison immutable; use verify')
    if not json.loads((PUBLIC / 'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real interrupted recovery pilot required')
    start, beat, guard = runtime(cfg)
    main = aggregate_groups(parent.NAME); old = aggregate_groups(parent.source.parent.NAME)
    prior = aggregate_groups(NAME) if verify else {}
    prior_refs = {Path(r['path']).name: r for r in json.loads((PUBLIC / 'complete.json').read_text())['groups']} if verify else {}
    refs, rows = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, _ = parent.BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            model_ref = models[context, seed, 'admitted_score']['checkpoint']
            if sha(ROOT / model_ref['path']) != model_ref['sha256']:
                raise ValueError('Frozen head changed')
            state = parent.BASE.api.completion.base.read_checkpoint(ROOT / model_ref['path'])
            for name, packet in sorted(packets.items()):
                key = f'{context}_{name}_seed{seed}_fit.json'; path = PUBLIC / 'groups' / key
                if path.exists() and not verify:
                    if not resume:
                        raise ValueError('Explicit partial readout resume required')
                    row = json.loads(path.read_text())
                    if row['registration_sha256'] != sha(REG):
                        raise ValueError('Partial comparison registration differs')
                else:
                    p, legacy_independent, _ = parent.infer(packet, state, store, old[key])
                    if parent.BASE.engine.fingerprint(p) != main[key]['prediction_sha256']:
                        raise ValueError('Original current-query model predictions differ')
                    groups = api.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
                    previous = parent.BASE.api.completion.base.read_checkpoint(parent.PRIVATE / 'progress' / (key+'.pt.gz'))
                    legacy_joint = previous['choices']['joint_matched']
                    if (previous['completed_queries'] != len(groups)
                            or parent.BASE.engine.fingerprint(legacy_joint) != main[key]['action_sha256']['joint_matched']):
                        raise ValueError('Old completed joint choices differ')
                    for a, c in (('independent', legacy_independent), ('joint_matched', legacy_joint)):
                        if evaluation.bank.policy_metrics(packet, c) != main[key]['result']['policies'][a]:
                            raise ValueError('Original complete legacy policy scores differ')
                    progress = PRIVATE / 'progress' / (key+'.pt.gz'); progress.parent.mkdir(parents=True, exist_ok=True)
                    if progress.exists() and not (resume or verify):
                        raise ValueError('Explicit action-checkpoint resume required')
                    choices, ranks, info = api.allocate(p, packet['allowed'], packet['recordings'],
                        packet['frames'], packet['ids'], seconds=cfg['solver_seconds_per_query'],
                        heartbeat=lambda **kw: beat(state='shared_count_allocating', context=context, source=name, seed=seed, **kw),
                        checkpoint=(lambda _: None) if verify else lambda value: parent.BASE.api.completion.base.save_checkpoint(progress, value),
                        resume_state=parent.BASE.api.completion.base.read_checkpoint(progress) if resume and progress.exists() and not verify else None)
                    result = evaluation.readout(packet, choices, ranks, info, legacy_independent, legacy_joint,
                        job['design']['easy_cut'], values[3]['origin'][packet['ids']])
                    row = dict(context=context, source=name, seed=seed, result=result,
                        prediction_sha256=parent.BASE.engine.fingerprint(p),
                        action_sha256={a: parent.BASE.engine.fingerprint(c) for a, c in choices.items()},
                        ids_sha256=main[key]['ids_sha256'], legacy_controls_exact=True,
                        legacy_joint_action_source='cached_verified_frozen_parent_not_resolved_again',
                        registration_sha256=sha(REG), model_checkpoint_sha256=model_ref['sha256'],
                        result_source='fresh_common_count_bank_inference_and_reoptimization_on_cached_verified_SOURCE_fitting_assets',
                        held_scored=False, independent_roles_opened=False)
                    if verify:
                        if row != prior[key]:
                            raise ValueError('Separate full shared-count comparison replay differs')
                    else:
                        once(path, row)
                rows.append(row); refs.append(prior_refs[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='shared_count_group_complete', groups=len(rows), total=72, context=context, source=name, seed=seed)
        del packets, state
    summary = evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    if verify:
        if (summary != json.loads((PUBLIC / 'summary.json').read_text())
                or refs != json.loads((PUBLIC / 'complete.json').read_text())['groups']):
            raise ValueError('Complete comparison summary differs')
        once(PUBLIC / 'verification.json', dict(all72groups_exact=True, all144legacy_full_policy_controls_exact=True,
            summary_sha256=sha(PUBLIC / 'summary.json'), complete_sha256=sha(PUBLIC / 'complete.json')))
    else:
        once(PUBLIC / 'summary.json', summary)
        once(PUBLIC / 'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            groups=refs, summary_sha256=sha(PUBLIC / 'summary.json'), new_training_updates=0,
            independent_roles_opened=False, deployment_changed=False))
    write(PRIVATE / ('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72shared_count_groups_verified' if verify else 'all72shared_count_groups_complete',
        fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if a.phase == 'register' else pilot() if a.phase == 'pilot' else execute(a.phase == 'verify', a.resume)
