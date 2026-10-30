"""Fresh frozen-head inference and exact selected-action attribution."""
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
from scripts import run_m3w_budget_score_cost as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_selected_action_attribution as api
from src.evaluation.m3w_active_bank_cost_readout import policy_metrics

NAME = 'european_selected_action_attribution_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), ROOT / 'src/evaluation/m3w_selected_action_attribution.py',
         ROOT / 'tests/test_m3w_selected_action_attribution.py', PUBLIC / 'protocol.md')


def admission(register=False):
    cfg, values, _, store, banks = parent.admission()
    models = parent.models()
    check = json.loads((parent.PUBLIC / 'verification.json').read_text())
    if not check['all216groups_exact'] or check['summary_sha256'] != parent.sha(parent.PUBLIC / 'summary.json'):
        raise ValueError('Verified complete parent required')
    parent.once(REG, dict(bindings={str(p.relative_to(ROOT)): parent.sha(p) for p in FILES},
        parent_complete_sha256=parent.sha(parent.PUBLIC / 'complete.json'),
        parent_verification_sha256=parent.sha(parent.PUBLIC / 'verification.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'], groups=216,
        new_optimizer_updates=0, new_policies=0, evaluation_labels_only=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if parent.sha(ROOT / rel) != digest:
                raise ValueError('Registered attribution changed')
            parent.frozen.engine.committed(ROOT / rel)
        parent.frozen.engine.committed(REG)
    return cfg, values, store, banks, models


def pool(values):
    out = {k: sum(v[k] for v in values) for k in ('rows', 'known_rows', 'unknown_rows',
        'selected_known_rows', 'selected_unknown_rows', 'selected_unknown_envelope_mass',
        'full_known_reference_mass', 'selected_net_benefit_mass',
        'predicted_selected_net_benefit_mass', 'selected_nonpositive_benefit_rows')}
    out['easy_support'] = {k: sum(v['easy_support'][k] for v in values) for k in values[0]['easy_support']}
    out['channels'] = {}
    for ch in ('all', 'easy'):
        channel = {}
        for k, value in values[0]['channels'][ch].items():
            channel[k] = ({a: sum(v['channels'][ch][k][a] for v in values) for a in value}
                          if isinstance(value, dict) else sum(v['channels'][ch][k] for v in values))
        out['channels'][ch] = channel
    return out


def summarize(rows, cfg, contexts, unique):
    expected = {(c, s, seed, obj) for c, sources in contexts.items() for s in sources
        for seed in cfg['seeds'] for obj in cfg['objectives']}
    keys = [(r['context'], r['source'], r['seed'], r['objective']) for r in rows]
    if len(rows) != 216 or len(set(keys)) != 216 or set(keys) != expected:
        raise ValueError('Complete fixed216group roster required')
    objects = {}
    for obj in cfg['objectives']:
        values = [r['attribution'] for r in rows if r['objective'] == obj]
        records = {}
        for r in rows:
            if r['objective'] == obj:
                for rec, value in r['attribution']['records'].items():
                    records.setdefault(rec, []).append(value)
        objects[obj] = dict(pooled_repeated_views=pool(values),
            by_action={str(a): pool([v['actions'][str(a)] for v in values]) for a in range(7)},
            by_record={k: pool(v) for k, v in sorted(records.items())})
    return dict(groups=len(rows), unique_queries=len(unique), objectives=objects,
        all_frozen_predictions_actions_full_metrics_exact=True,
        result_source='fresh_frozen_Torch_inference_evaluation_only_arithmetic_attribution',
        repeated_views_not_independent=True, selection_rule_unchanged=True,
        unknown_outcomes_imputed=False, causal_root_cause_claim=False,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, metric_claim=False, seconds_claim=False,
        stage5c_executed=False, smc_enabled=False)


def execute(verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'complete.json').exists() and not verify:
        raise ValueError('Completed attribution immutable; use verify')
    parent.torch.set_num_threads(4)
    parent.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume immutable receipts')))
    def beat(**kw):
        if time.monotonic() - start > 43200 or shutil.disk_usage(ROOT).free < 10 * 2**30:
            raise RuntimeError('12h or10GiB guard; resume same roster')
        v = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
        parent.write(PRIVATE / 'heartbeat.json', v)
        print(json.dumps(v), flush=True)
    beat(state='ready', threads=4, workers=0, new_optimizer_updates=0)
    previous = aggregate_groups(parent.NAME)
    rows, refs, unique = [], [], set()
    for context, items in sorted(values[2]['contexts'].items()):
        op, packets, original, changed = parent.frozen.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            for obj in cfg['objectives']:
                model = models[context, seed, 'new_neural', obj]
                state = parent.frozen.api.completion.base.read_checkpoint(ROOT / model['checkpoint']['path'])
                for source, packet in sorted(packets.items()):
                    extra = store.read(packet['ids'], 'history')
                    p = parent.frozen.api.detector.predict(state, packet['x'], packet['scene'], extra,
                        packet['envelope'], packet['scales'])
                    c, _ = parent.frozen.api.completion.base.select_actions(p, packet['allowed'], arm='physical_neural')
                    ref = previous[f'{context}_{source}_seed{seed}_{obj}_fit.json']
                    if (parent.frozen.engine.fingerprint(p) != ref['prediction_sha256']['new_neural']
                            or parent.frozen.engine.fingerprint(c) != ref['action_sha256']['new_neural']
                            or policy_metrics(packet, c) != ref['result']['policies']['new_neural']
                            or parent.frozen.engine.fingerprint(packet['ids']) != ref['ids_sha256']):
                        raise ValueError('Frozen predictions/actions/full metrics or row alignment changed')
                    value = dict(context=context, source=source, seed=seed, objective=obj,
                        prediction_sha256=ref['prediction_sha256']['new_neural'],
                        action_sha256=ref['action_sha256']['new_neural'], ids_sha256=ref['ids_sha256'],
                        prior_full_policy_metrics_exact=True,
                        attribution=api.analyze(packet, p, c, easy_cut=job['design']['easy_cut']))
                    path = PUBLIC / 'groups' / f'{context}_{source}_seed{seed}_{obj}.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit partial resume required')
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact attribution replay differs')
                    else:
                        parent.once(path, value)
                    unique.update(map(int, packet['ids']))
                    rows.append(value)
                    refs.append(dict(path=str(path.relative_to(ROOT)), sha256=parent.sha(path)))
            beat(state='attribution_progress', context=context, seed=seed, groups=len(rows), total=216)
        del op, packets, original, changed
    summary = summarize(rows, cfg, json.loads(REG.read_text())['contexts'], unique)
    if verify:
        if summary != json.loads((PUBLIC / 'summary.json').read_text()):
            raise ValueError('Summary replay differs')
        parent.once(PUBLIC / 'verification.json', dict(all216groups_exact=True,
            all_frozen_predictions_actions_metrics_exact=True, summary_sha256=parent.sha(PUBLIC / 'summary.json')))
    else:
        parent.once(PUBLIC / 'summary.json', summary)
        parent.once(PUBLIC / 'complete.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            groups=refs, summary_sha256=parent.sha(PUBLIC / 'summary.json'),
            new_optimizer_updates=0, held_scored=False, independent_roles_opened=False))
    parent.write(PRIVATE / ('verification_runtime.json' if verify else 'runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic() - start,
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all216_exact_verified' if verify else 'all216_complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if args.phase == 'register' else execute(args.phase == 'verify', args.resume)
