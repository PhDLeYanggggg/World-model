"""Recompute fixed TRAIN-monitor aggregation; no fitting or held-out labels."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts import read_m3w_scaled_risk_heads as heads

training = heads.training
ROOT, base = heads.ROOT, training.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_risk_aggregation_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_risk_aggregation_v1'
REG = PUBLIC/'registration.json'


def registration():
    files = [Path(__file__).resolve(), ROOT/'src/world_model/m3w_risk_aggregation_audit.py',
             ROOT/'tests/test_m3w_risk_aggregation_audit.py', PUBLIC/'protocol.md']
    result = dict(bindings={str(p.relative_to(ROOT)):base.sha(p) for p in files},
        parent_freeze_sha256=base.sha(training.PUBLIC/'training_freeze.json'),
        role='posthoc_TRAIN_monitor_mechanism_diagnostic', views=144,
        monitor_queries=128, new_training=False, validation_read=False,
        independent_roles_read=False, deployment_changed=False)
    base.once(REG, result)
    return result


def execute(resume):
    reg = registration()
    for rel in [*reg['bindings'], str(REG.relative_to(ROOT))]:
        output = subprocess.run(['git','show','HEAD:'+rel], cwd=ROOT, capture_output=True)
        if output.returncode or output.stdout != (ROOT/rel).read_bytes():
            raise ValueError('Commit registered diagnostic first')
    if (PUBLIC/'complete.json').exists(): raise ValueError('Complete audit is immutable')
    from src.world_model import m3w_risk_aggregation_audit as api
    from src.world_model.m3w_false_safe_diagnostic import moving_from_features
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    documents, inventory = heads.admission()
    train_reg, _ = heads.previous.execution.check_registration()
    manifest = json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rows, refs, previous = [], [], None
    started = time.monotonic()
    def beat(**kwargs):
        value = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kwargs)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(value)+'\n')
        os.replace(tmp, PRIVATE/'heartbeat.json'); print(json.dumps(value), flush=True)
    with (PRIVATE/'process.lock').open('a') as lock, heads.Reader(inventory) as reader:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for ref in manifest['heads']:
            group, identity = ref['group'], ref['identity']
            if group != previous:
                beat(state='streaming_TRAIN', group=group)
                args = heads.previous.execution.load_packet(ref, train_reg)
                _, support = api.old.parent.forest.causal_inputs(args[0], args[1], args[-1])
                eligible = moving_from_features(args[0]) & support
                previous = group
            for mode in ('unit_control', 'budget_scaled'):
                doc = documents[group, identity['source'], identity['seed'], mode]
                path = PUBLIC/'groups'/(group+'_head'+str(identity['seed'])+'_'+mode+'.json')
                binding = dict(registration_sha256=base.sha(REG), checkpoint_sha256=doc['checkpoint']['sha256'], mode=mode)
                if path.exists():
                    if not resume: raise ValueError('Explicit resume required')
                    row = json.loads(path.read_text())
                    if any(row[k] != v for k,v in binding.items()): raise ValueError('Changed audit')
                else:
                    state = reader.load(ROOT/doc['checkpoint']['path'])
                    row = dict(**api.frozen_monitor(state, args, eligible), **binding)
                    base.once(path, row)
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path)))
            beat(state='diagnosed', views=len(rows), total=144)
        summary = {}
        for mode in ('unit_control', 'budget_scaled'):
            subset = [r for r in rows if r['mode'] == mode]
            summary[mode] = dict(views=len(subset))
            for gate in ('soft', 'hard'):
                summary[mode][gate] = {}
                for constraint in ('all', 'easy'):
                    values = [r['aggregations'][gate]['constraints'][constraint] for r in subset]
                    summary[mode][gate][constraint] = dict(
                        query_excess_views=sum(v['query_mean_excess'] > 0 for v in values),
                        mass_excess_views=sum(v['agent_mass_mean_excess'] > 0 for v in values),
                        query_pass_mass_fail_views=sum(v['query_pass_mass_fail'] for v in values),
                        mass_pass_query_fail_views=sum(v['mass_pass_query_fail'] for v in values),
                        unsupported_reference_views=sum(not v['reference_support'] for v in values))
        base.once(PUBLIC/'summary.json', summary)
        base.once(PUBLIC/'complete.json', dict(groups=refs, summary_sha256=base.sha(PUBLIC/'summary.json'),
            registration_sha256=base.sha(REG), seconds=time.monotonic()-started,
            result_source='fresh_run_fixed_TRAIN_monitor_aggregation', checkpoint_source='cached_verified',
            optimizer_steps=0, validation_read=False, independent_roles_read=False, deployment_changed=False))
        beat(state='complete', views=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register','run']); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'register': registration(); print('Registered TRAIN-only aggregation diagnostic')
    else: execute(args.resume)


if __name__ == '__main__': main()
