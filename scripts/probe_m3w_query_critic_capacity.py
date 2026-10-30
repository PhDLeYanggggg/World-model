"""Read-only training-label scale/capacity diagnostic, no new policy readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_selected_query_critic as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_query_critic_capacity as api

np = parent.np
NAME = 'european_query_critic_capacity_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_query_critic_capacity.py',
    ROOT/'tests/test_m3w_query_critic_capacity.py', PUBLIC/'protocol.md')
sha, once, write = parent.sha, parent.once, parent.write


def admission(register=False):
    cfg, values, store, banks, models = parent.admission()
    freeze = json.loads((parent.PUBLIC/'training_freeze.json').read_text())
    trace = json.loads((parent.PUBLIC/'training_diagnostics.json').read_text())
    if len(freeze['fits']) != 36 or trace['primary_optimizer_updates'] != 72000:
        raise ValueError('Full frozen matched training and actual telemetry required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        training_diagnostics_sha256=sha(parent.PUBLIC/'training_diagnostics.json'),
        parent_policy_readout_complete_at_registration=(parent.PUBLIC/'complete.json').exists(),
        parent_new_policy_outcomes_read=False, label_role='original_SOURCE_fitting_supervision_only',
        groups=72, context_heads=18, risk_budget=.02, numerical_count_tolerance=1e-10,
        target='training_query_units_and_protected_moment_label_fitting_geometry',
        new_training_updates=0, new_policy_scored=False, inference_features_changed=False,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Diagnostic definition changed: '+rel)
            parent.frozen.BASE.engine.committed(ROOT/rel)
        parent.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models


def summarize(rows):
    if len(rows) != 72 or len({(r['context'], r['seed'], r['source']) for r in rows}) != 72:
        raise ValueError('All72 fitting diagnostic groups required')
    heads = []
    for context, seed in sorted({(r['context'], r['seed']) for r in rows}):
        subset = [r['result'] for r in rows if (r['context'], r['seed']) == (context, seed)]
        if len(subset) != 4: raise ValueError('All four original fitting sources required')
        source_means = [np.asarray(r['query_target_squared_sum'])/r['complete_selected_queries']
            for r in subset if r['complete_selected_queries']]
        target = np.sqrt(np.mean(source_means, 0)) if source_means else None
        current = np.asarray(subset[0]['current_query_score_RMS'])
        if any(not np.array_equal(r['current_query_score_RMS'], current) for r in subset):
            raise ValueError('The common frozen fitting RMS must agree')
        heads.append(dict(context=context, seed=seed,
            sources_with_complete_selected_support=len(source_means),
            current_query_score_RMS=current.tolist(),
            label_query_score_RMS_source_balanced=target.tolist() if target is not None else None,
            current_over_label_RMS_after_1e6_floor=(current/np.maximum(target, 1e-6)).tolist() if target is not None else None))
    keys = ('queries', 'complete_selected_queries', 'unknown_selected_queries', 'empty_selected_queries',
        'known_legal_actions', 'point_harm_above_protected_cap_actions', 'point_easy_harm_above_protected_cap_actions',
        'query_target_below_zero_harm', 'query_target_below_zero_easy_harm', 'query_target_above_all_harm_cap',
        'query_target_above_easy_harm_cap', 'query_target_easy_exceeds_all_harm')
    return dict(heads=heads, counts={k: sum(r['result'][k] for r in rows) for k in keys},
        source_weights='equal_source_conditional_complete_selected_query_second_moments',
        zero_support_sources_reported_not_imputed=True, geometry_oracle_not_deployable=True,
        conditional_mean_misspecification_proved=False, current_model_changed=False,
        new_training_updates=0, parent_new_policy_outcomes_read=False,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False)


def execute(verify=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed diagnostic immutable')
    parent.api.torch.set_num_threads(4); parent.api.torch.set_num_interop_threads(1)
    start = time.monotonic()
    def beat(**event):
        if time.monotonic()-start > 43200: raise TimeoutError('12h diagnostic limit')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row)
        print(json.dumps(row), flush=True)
    beat(state='read_only_training_label_diagnostic_started', threads=4, workers=0)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    previous = aggregate_groups(NAME) if verify else {}
    prior_refs = {Path(r['path']).name: r for r in json.loads((PUBLIC/'complete.json').read_text())['groups']} if verify else {}
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            data, ranges, _ = parent.prepare(context, seed, packets, changed, store, models, old, main, beat)
            c = data['causal']
            for name, packet in sorted(packets.items()):
                sl = ranges[name]; initial = c['initial'][sl]
                groups = parent.decoder.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
                result = api.diagnose(c['prediction'][sl], packet['envelope'], packet['y'], packet['scales'],
                    packet['allowed'], initial, groups, data['rms'])
                row = dict(context=context, seed=seed, source=name, result=result,
                    original_prediction_sha256=parent.frozen.BASE.engine.fingerprint(c['prediction'][sl]),
                    original_selected_set_sha256=parent.frozen.BASE.engine.fingerprint(initial),
                    training_labels_sha256=parent.frozen.BASE.engine.fingerprint(packet['y']),
                    registration_sha256=sha(REG), new_policy_outcomes_read=False)
                key = f'{context}_{name}_seed{seed}.json'; path = PUBLIC/'groups'/key
                if verify:
                    if row != previous[key]: raise ValueError('Full training-label diagnostic replay differs')
                else: once(path, row)
                rows.append(row); refs.append(prior_refs[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            del data
        del changed, packets
    summary = summarize(rows)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Diagnostic summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, summary_sha256=sha(PUBLIC/'summary.json'),
            complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            groups=refs, summary_sha256=sha(PUBLIC/'summary.json'), parent_new_policy_outcomes_read=False))
    beat(state='all72_capacity_diagnostics_verified' if verify else 'all72_capacity_diagnostics_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'run', 'verify'))
    args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if args.phase == 'register' else execute(args.phase == 'verify')
