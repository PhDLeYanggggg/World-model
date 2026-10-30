"""Fixed-critic test of per-action versus joint positive utility."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_query_unit_critic as control
from scripts import run_m3w_current_policy_critic as current
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_signed_utility_allocation as api
from src.evaluation.m3w_multiaction_support import solve_action_bank
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_metric_bundle import pack

parent = control.parent; bank = parent.evaluation.bank; legacy = bank.legacy; np = control.np
NAME = 'european_signed_utility_allocation_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_signed_utility_allocation_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_signed_utility_allocation.py',
    ROOT/'tests/test_m3w_signed_utility_allocation.py', PUBLIC/'protocol.md')
sha, once, write = control.sha, control.once, control.write


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, values, store, banks, models, _ = control.admission()
    for run in (control, current):
        v = json.loads((run.PUBLIC/'verification.json').read_text())
        if not v['all72groups_exact'] or v['summary_sha256'] != sha(run.PUBLIC/'summary.json'):
            raise ValueError('Complete prior readout/replay required')
    if (cfg['risk_budget'] != .02 or cfg['solver_seconds'] != 2. or cfg['seeds'] != [17, 29, 43]
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Unchanged risk and closed roles required')
    heads = current.roster(control.PUBLIC/'training_freeze.json')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        control_registration_sha256=sha(control.REG), control_training_sha256=sha(control.PUBLIC/'training_freeze.json'),
        control_complete_sha256=sha(control.PUBLIC/'complete.json'), control_summary_sha256=sha(control.PUBLIC/'summary.json'),
        current_negative_summary_sha256=sha(current.PUBLIC/'summary.json'),
        current_negative_verification_sha256=sha(current.PUBLIC/'verification.json'),
        contexts=json.loads(control.REG.read_text())['contexts'], cached_verified_selected_set_heads=18,
        fresh_training_updates=0, changed_factor='per_action_positive_utility_admissibility_only',
        primary='signed_matched_minus_positive_matched_full_completion_lower',
        role='exposed_SOURCE_fitting_diagnosis_only', own_policy_outcomes_read_at_registration=False,
        risk_budget=.02, independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, h in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != h: raise ValueError('Registered experiment changed:'+rel)
            parent.frozen.BASE.engine.committed(ROOT/rel)
        parent.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models, heads


def runtime(cfg):
    parent.api.torch.set_num_threads(4); parent.api.torch.set_num_interop_threads(1); started = time.monotonic()
    def beat(**event):
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h limit')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row); print(json.dumps(row), flush=True)
    beat(state='runtime_ready', device='cpu', architecture=platform.machine(), threads=4, workers=0)
    return started, beat


def prepare(context, seed, packets, changed, values, store, models, heads, old, main, beat):
    data, ranges, _ = parent.prepare(context, seed, packets, changed, store, models, old, main, beat)
    cp = heads[context, seed, 'selected_set']; state = parent.api.read_checkpoint(ROOT/cp['path'])
    parent.api.exact(state['input_hashes'], data['input_hashes'])
    prediction = parent.api.predict(state, data['causal'])
    del data
    return prediction, ranges


def source_anchor(context, seed, source, prediction, packet, controls):
    key = f'{context}_{source}_seed{seed}_fit.json'; row = controls[key]
    c = parent.api.read_checkpoint(control.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['set_joint']
    if (parent.api.digest(prediction) != row['prediction_sha256']['selected_set']
            or parent.api.digest(c) != row['action_sha256']['set_joint']
            or bank.policy_metrics(packet, c) != row['result']['policies']['set_joint']):
        raise ValueError('Original complete prediction/action/policy control differs')
    return key, c


def pilot():
    cfg, values, store, banks, models, heads = admission(); start, beat = runtime(cfg)
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    context, items = next(iter(sorted(values[2]['contexts'].items())))
    _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    p, ranges = prepare(context, 17, packets, changed, values, store, models, heads, old, main, beat)
    source = sorted(packets)[0]; packet = packets[source]; p = p[ranges[source]]
    _, anchor = source_anchor(context, 17, source, p, packet, aggregate_groups(control.NAME))
    groups = api.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    at = np.sort(np.concatenate(groups[:cfg['pilot_queries']]))
    args = (p[at], packet['allowed'][at], packet['recordings'][at], packet['frames'][at], packet['ids'][at], anchor[at])
    cp = PRIVATE/'pilot.pt.gz'
    class Interrupted(Exception): pass
    def interrupt(v):
        parent.api.save_checkpoint(cp, v); raise Interrupted()
    try: api.allocate(*args, seconds=cfg['solver_seconds'], checkpoint=interrupt)
    except Interrupted: pass
    resumed = api.allocate(*args, seconds=cfg['solver_seconds'], resume_state=parent.api.read_checkpoint(cp), heartbeat=beat)
    began = time.monotonic(); direct = api.allocate(*args, seconds=cfg['solver_seconds'], heartbeat=beat)
    measured = time.monotonic()-began; parent.api.exact(resumed, direct)
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rows=len(at), queries=min(len(groups), cfg['pilot_queries']),
        direct_seconds=measured, one_source_query_projection_seconds=measured/min(len(groups), cfg['pilot_queries'])*len(groups)*72,
        projection_not_measured_full_roster=True, exact_real_interrupted_recovery=True,
        outcome_scoring=False, engineering_pass=True, source=source, context=context))


def readout(packet, choices, info, easy_cut):
    ps = {a: bank.policy_metrics(packet, c) for a, c in choices.items()}
    records = {str(rec): {a: legacy.compact_score(packet['y'][at], packet['envelope'][at], c[at]) for a, c in choices.items()}
        for rec in sorted(set(packet['recordings'])) for at in (packet['recordings'] == rec,)}
    diag = {}
    known = np.isfinite(packet['y']).all((1, 2))
    for rec in sorted(set(packet['recordings'])):
        at = packet['recordings'] == rec; y, env, ok = packet['y'][at], packet['envelope'][at], packet['allowed'][at]
        full, _ = solve_action_bank(y, env, ok)
        restricted, _ = solve_action_bank(y, env, ok & known[at, None] & (y[..., 0] > y[..., 1]))
        diag[str(rec)] = dict(full=full, positive_action_only=restricted)
    comparisons = {mode: legacy.paired_bounds(packet, choices['positive_'+mode], choices['signed_'+mode]) for mode in ('matched', 'free')}
    floor = np.full(len(packet['ids']), -1, np.int64)
    for a, c in choices.items(): comparisons[a+'_minus_floor'] = legacy.paired_bounds(packet, floor, c)
    return dict(policies=ps, recordings=records, comparisons=comparisons, allocation=info,
        subsets=subsets(packet, choices, easy_cut), oracle_admissibility=diag,
        inference_uses_labels=False, new_training_updates=0)


def summarize(rows, cfg):
    contexts = json.loads(REG.read_text())['contexts']
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != expected:
        raise ValueError('Complete72-group roster required')
    interval = lambda get: legacy.interval(rows, get, cfg)
    effects = {mode: {key: interval(lambda r, mode=mode, key=key: r['comparisons'][mode][key])
        for key in ('known_pp', 'lower_pp', 'upper_pp')} for mode in ('matched', 'free')}
    safety = {a: bank.safety(rows, a) for a in api.NAMES}
    unique = [r for r in rows if r['seed'] == 17]; oracle = Counter()
    for r in unique:
        for other in (x for x in rows if x['context'] == r['context'] and x['source'] == r['source']):
            if other['result']['oracle_admissibility'] != r['result']['oracle_admissibility']:
                raise ValueError('Oracle must not depend on head seed')
        for record in r['result']['oracle_admissibility'].values():
            oracle['required_record_contexts'] += 1
            for k, v in record.items():
                oracle[k+'_positive_feasible'] += int(v['positive_feasible'])
                oracle[k+'_supported'] += int(v['budget_feasible'])
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    floor = interval(lambda r: r['comparisons']['signed_matched_minus_floor']['lower_pp'])
    risk = safety['signed_matched']
    gates = dict(complete72=len(rows) == 72, all72_original_controls_exact=all(r['original_control_exact'] for r in rows),
        oracle_control_exact=all(r['original_oracle_exact'] for r in rows), matched_count_exact=all(r['result']['allocation']['matched_count_every_query'] for r in rows),
        primary_lower_CI_positive=positive(effects['matched']['lower_pp']), floor_lower_CI_positive=positive(floor),
        complete_original_risk=risk['supported_record_views'] == risk['required_record_views'] and risk['supported_views'] == 72,
        primary_optima_certified=not any(r['result']['allocation']['nonoptimal'].get('signed_matched', 0) for r in rows))
    return dict(groups=72, accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in api.NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in api.NAMES},
        safety=safety, effects=effects, oracle_admissibility=dict(oracle), subsets=pooled_subsets(rows),
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()), new_training_updates=0,
        result_source='fresh_causal_allocation_cached_verified_fixed_critic_exposed_SOURCE_fitting',
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def execute(verify=False, resume=False):
    cfg, values, store, banks, models, heads = admission(); start, beat = runtime(cfg)
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real pilot required')
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Complete experiment immutable')
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    controls = aggregate_groups(control.NAME); prior = aggregate_groups(NAME) if verify else {}; rows = []; refs = []
    oracle_controls = aggregate_groups(parent.frozen.BASE.NAME)
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            p, ranges = prepare(context, seed, packets, changed, values, store, models, heads, old, main, beat)
            for source, packet in sorted(packets.items()):
                pred = p[ranges[source]]; key, anchor = source_anchor(context, seed, source, pred, packet, controls)
                path = PUBLIC/'groups'/key
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit incomplete-roster resume required')
                    row = json.loads(path.read_text())
                    if row['registration_sha256'] != sha(REG): raise ValueError('Partial registration changed')
                else:
                    cp = PRIVATE/'progress'/(key+'.pt.gz'); cp.parent.mkdir(exist_ok=True)
                    if cp.exists() and not (verify or resume): raise ValueError('Explicit progress resume required')
                    choices, info = api.allocate(pred, packet['allowed'], packet['recordings'], packet['frames'], packet['ids'], anchor,
                        seconds=cfg['solver_seconds'], checkpoint=(lambda _: None) if verify else lambda v: parent.api.save_checkpoint(cp, v),
                        heartbeat=lambda **kw: beat(state='allocation', context=context, source=source, seed=seed, **kw),
                        resume_state=parent.api.read_checkpoint(cp) if resume and cp.exists() and not verify else None)
                    result = readout(packet, choices, info, job['design']['easy_cut'])
                    for rec, o in result['oracle_admissibility'].items():
                        if o['full'] != oracle_controls[key]['result']['bank_opportunity_diagnostic']['new_neural'][rec]:
                            raise ValueError('Original record-bank oracle control changed')
                    row = dict(context=context, source=source, seed=seed, result=result,
                        registration_sha256=sha(REG), prediction_sha256=parent.api.digest(pred),
                        action_sha256={a: parent.api.digest(c) for a, c in choices.items()},
                        original_control_exact=True, original_oracle_exact=True)
                    if verify:
                        if row != prior[key]: raise ValueError('Separate complete replay differs')
                    else: once(path, row)
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='group_complete', groups=len(rows), total=72)
            del p
    summary = summarize(rows, cfg)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, all72_original_controls_exact=True,
            all148_oracles_exact=True, summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs, summary_sha256=sha(PUBLIC/'summary.json')))
        pack(ROOT, PUBLIC)
    once(PUBLIC/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_verified' if verify else 'all72_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.phase == 'register': admission(True)
        elif a.phase == 'pilot': pilot()
        else: execute(a.phase == 'verify', a.resume)
