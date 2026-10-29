"""Exact label-aware feasibility on fitting recordings, with frozen controls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import time

from scripts import run_m3w_cross_record_policy as prior
from src.evaluation import m3w_train_feasibility as api

ROOT, sha, once = prior.ROOT, prior.sha, prior.once
base = prior.api.base
whole, direct = prior.previous, prior.previous.previous
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_train_feasibility_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_train_feasibility_v1'
REG = PUBLIC/'registration.json'
ARMS = ('origin', 'exact_gradient', 'same_record_meta', 'cross_record_meta')


def fit_packet(args, partition):
    records = args[5].astype(str)
    source = args[-1]['training_site']
    if partition != base.shared.recording_partition(source, records):
        raise ValueError('Keep the frozen whole-recording partition')
    mask = api.np.isin(records, partition['fit_recordings'])
    arrays = tuple(a[mask] for a in args[:-1])
    x, env, y, _, sites, recs, frames = arrays
    pr = base.shared.old.core.preprocess(x, env, y, sites, recs, frames, training_site=source)
    return (*arrays, pr)


def fit_packets(items, ancestor, beat):
    result = {}
    for item in items:
        source = item['packet']['identity']['source']
        beat(state='loading_source_TRAIN_packet', source=source)
        raw = direct.LOADER.census.heads.previous.execution.load_packet(item['packet'], ancestor)
        result[source] = fit_packet(raw, item['partition'])
        del raw
    return result


def register():
    reg, _, ancestor = prior.admission()
    paths = [Path(__file__).resolve(), ROOT/'src/evaluation/m3w_train_feasibility.py',
             ROOT/'tests/test_m3w_train_feasibility.py', PUBLIC/'protocol.md', prior.REG,
             prior.PUBLIC/'training_freeze.json', prior.PUBLIC/'complete.json']
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], policies=list(ARMS),
        fitting_packet_views=24, frozen_policy_seed_views=72, risk_budget=.02,
        label_aware_diagnostic_only=True, new_training=False, inner_held_scored=False,
        physical_packets_contain_original_TRAIN_inner_held_rows=True,
        independent_roles_read=False, deployable=False, stage5c_executed=False, smc_enabled=False))
    return reg, ancestor


def summarize(rows):
    if len(rows) != 72 or len({(r['context'], r['source'], r['seed']) for r in rows}) != 72:
        raise ValueError('Complete fitting roster required')
    unique = {}
    for row in rows:
        key = row['context'], row['source']
        oracle = {r:d['oracle'] for r,d in row['result']['recordings'].items()}
        if key in unique and unique[key] != oracle:
            raise ValueError('Label oracle cannot depend on head seed')
        unique[key] = oracle
    records = [(context, source, rec, d) for (context, source), rr in unique.items() for rec,d in rr.items()]
    policies = {}
    for arm in ARMS:
        views = [r['result']['policies'][arm] for r in rows]
        rviews = [d['policies'][arm] for r in rows for d in r['result']['recordings'].values()]
        normalized = lambda key: statistics.fmean(100*r['result']['policies'][arm][key]
            /r['result']['full_known_reference_mass'] for r in rows)
        all_positive = statistics.fmean(100*r['result']['oracle']['positive_benefit_upper_mass']
            /r['result']['full_known_reference_mass'] for r in rows)
        policies[arm] = dict(
            observed_budget_views=sum(api.budget_status(v['observed'])['supported'] for v in views),
            observed_budget_record_views=sum(api.budget_status(v['observed'])['supported'] for v in rviews),
            deletion_oracle_budget_record_views=sum(v['deletion_oracle']['budget_feasible'] for v in rviews),
            deletion_oracle_positive_record_views=sum(v['deletion_oracle']['safe_positive_selection_exists'] for v in rviews),
            missed_positive_benefit_pp=normalized('missed_positive_benefit_mass'),
            retained_positive_benefit_pp=normalized('retained_positive_benefit_mass'),
            retained_fraction_of_eligible_positive_benefit=normalized('retained_positive_benefit_mass')/all_positive,
            observed_known_harm_pp=statistics.fmean(100*v['result']['policies'][arm]['observed']['selected_known_harm_mass']
                /v['result']['full_known_reference_mass'] for v in rows),
            observed_unknown_envelope_pp=statistics.fmean(100*v['result']['policies'][arm]['observed']['selected_unknown_envelope_mass']
                /v['result']['full_known_reference_mass'] for v in rows),
            required_record_views=len(rviews), required_views=len(views))
    reference_rows = [r for r in rows if r['seed'] == 17]
    return dict(packet_views=len(unique), record_views_without_head_seed_repeats=len(records),
        unique_source_recordings=len({(s,r) for _,s,r,_ in records}),
        fitting_rows_without_head_seed_repeats=sum(r['result']['rows'] for r in reference_rows),
        eligible_rows_without_head_seed_repeats=sum(r['result']['eligible_rows'] for r in reference_rows),
        oracle_budget_record_views=sum(d['budget_feasible'] for *_,d in records),
        oracle_positive_record_views=sum(d['safe_positive_selection_exists'] for *_,d in records),
        infeasibility_reasons=dict(Counter(d['infeasibility_reason'] for *_,d in records if not d['budget_feasible'])),
        unsupported_records=[dict(context=c,source=s,recording=r,reason=d['infeasibility_reason'],
            allowed_rows=d['allowed_rows'],positive_upper_mass=d['positive_benefit_upper_mass'],
            allowed_known_easy_rows=d['allowed_known_easy_rows']) for c,s,r,d in records if not d['budget_feasible']],
        positive_benefit_upper_pp=statistics.fmean(100*r['result']['oracle']['positive_benefit_upper_mass']
            /r['result']['full_known_reference_mass'] for r in reference_rows),
        policies=policies, scope='fitting_recordings_only', inner_held_scored=False,
        future_labels_used_by_oracle=True, deployable=False, new_training=False,
        independent_confirmation=False, population_safety_proven=False)


def run(check=False, resume=False):
    reg, ancestor = register()
    for rel in [*json.loads(REG.read_text())['bindings'], str(REG.relative_to(ROOT))]:
        direct.previous.pooled.previous.previous.committed(ROOT/rel)
    if (PUBLIC/'complete.json').exists() and not check:
        raise ValueError('Completed diagnostic immutable; use check')
    base.torch.set_num_threads(4); base.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    def beat(**kw):
        d = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(d)+'\n'); os.replace(tmp, PRIVATE/'heartbeat.json')
        print(json.dumps(d), flush=True)
        if time.monotonic()-began > 43200: raise TimeoutError('12h boundary; resume same roster')
        if shutil.disk_usage(ROOT).free < 10*2**30: raise OSError('Keep 10GiB reserve')
    learned, parent = prior.models(), whole.models()
    _, origins = direct.ORIGINS.models(reg)
    finish = json.loads((prior.PUBLIC/'complete.json').read_text())
    prior_refs = {Path(r['path']).name:r for r in finish['groups'] if r['path'].endswith('_fit.json')}
    rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            packets = fit_packets(items, ancestor, beat)
            for seed in reg['seeds']:
                origin = base.core.read_checkpoint(ROOT/origins[context, seed, 'self']['checkpoint']['path'])
                predictions = direct.origin_predictions(origin, packets)
                states = {a:base.core.read_checkpoint(ROOT/learned[context,seed,a]['checkpoint']['path']) for a in prior.api.ARMS}
                states['exact_gradient'] = base.core.read_checkpoint(ROOT/parent[context,seed,'exact_gradient']['checkpoint']['path'])
                for source, args in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_fit.json'
                    if path.exists() and not (check or resume): raise ValueError('Explicit resume required')
                    eligible = base.shared.eligible(args)
                    actions = dict(origin=base.actions(predictions[source], eligible))
                    for arm, state in states.items():
                        base.core.exact(state['source_preprocess'][source], args[-1])
                        for k,a in zip(('x','envelope','labels','series','sites','recordings','frames'),args[:-1]):
                            if state['input_hashes'][source][k] != base.fingerprint(a): raise ValueError('Frozen fitting input mismatch')
                        _, actions[arm] = base.predict(state,source,args[0],args[1],predictions[source],eligible)
                    result = api.evaluate(args, eligible, actions)
                    oldpath = ROOT/prior_refs[path.name]['path']
                    if sha(oldpath) != prior_refs[path.name]['sha256']: raise ValueError('Frozen fit readout changed')
                    old = json.loads(oldpath.read_text())
                    for arm in ARMS:
                        if (base.fingerprint(actions[arm]) != old['action_sha256'][arm]
                                or result['policies'][arm]['observed'] != old['result']['policies'][arm]):
                            raise ValueError('Frozen fit actions or scores differ')
                    value = dict(context=context,source=source,seed=seed,result=result,
                        registration_sha256=sha(REG), prior_fit_group_sha256=sha(oldpath),
                        result_source='fresh_run_TRAIN_label_aware_oracle_frozen_controls_cached_verified',
                        inner_held_scored=False, independent_roles_read=False)
                    if check:
                        if value != json.loads(path.read_text()): raise ValueError('Exact diagnosis replay differs')
                    else: once(path,value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='fit_views_complete',views=len(rows),total=72)
        summary = summarize(rows)
        if check:
            done = json.loads((PUBLIC/'complete.json').read_text())
            if summary != json.loads((PUBLIC/'summary.json').read_text()) or done['groups'] != refs:
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(PUBLIC/'complete.json',dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),
                registration_sha256=sha(REG),seconds=time.monotonic()-began,pid=os.getpid(),
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                new_training=False,inner_held_scored=False,independent_roles_read=False))
        beat(state='exact_replay_complete' if check else 'diagnosis_complete',
            oracle_feasible=summary['oracle_budget_record_views'],records=summary['record_views_without_head_seed_repeats'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action',choices=('register','run','check'))
    p.add_argument('--resume',action='store_true'); a = p.parse_args()
    if a.action == 'register': register()
    else: run(a.action == 'check',a.resume)
