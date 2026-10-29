"""Same-region fitting diagnosis of the frozen paired supervision endpoints."""
import argparse
import fcntl
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_completion_cost as run
from scripts.diagnose_m3w_multiaction_forest_costs import conditional_summary
from src.evaluation.m3w_policy_region_shift import compare_regions

REG = run.PUBLIC/'mechanism_registration.json'
FILES = (Path(__file__), run.PUBLIC/'mechanism_protocol.md',
    ROOT/'tests/test_m3w_completion_regions.py', ROOT/'src/evaluation/m3w_policy_region_shift.py',
    ROOT/'src/evaluation/m3w_selected_cost_diagnosis.py',
    ROOT/'scripts/diagnose_m3w_multiaction_forest_costs.py')


def check_pair(left, right):
    for a, b in (('selected', 'frozen_teacher_region'), ('frozen_teacher_region', 'selected'), ('all_legal', 'all_legal')):
        for key in ('observed_known_mass', 'action_views', 'known_action_views', 'unknown_action_views'):
            if left[a][key] != right[b][key]:
                raise ValueError('Both estimators must use the identical observed/unknown evaluation region')


def admission(register=False):
    cfg, values = run.admission(); learned = run.models()
    verification = json.loads((run.PUBLIC/'verification.json').read_text())
    if not verification['exact_replay'] or verification['summary_sha256'] != run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Completed primary exact replay required')
    run.once(REG, dict(bindings={str(p.relative_to(ROOT)): run.sha(p) for p in FILES},
        primary_summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        primary_verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        required_paired_record_views=888, scope='fitting_only_fixed_cross_policy_regions',
        new_optimizer_updates=0, held_scored=False, independent_roles_opened=False, deployment_changed=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT/rel) != digest:
                raise ValueError('Registered same-region diagnosis changed')
            run.parent.parent.committed(ROOT/rel)
        run.parent.parent.committed(REG)
    return cfg, values, learned


def execute(verify=False):
    cfg, values, learned = admission(); started, beat, guard = run.runtime(cfg)
    records = []
    for context, items in sorted(values[2]['contexts'].items()):
        packets = run.packets(context, items, values, beat)
        for seed in cfg['seeds']:
            for bank in cfg['banks']:
                states = {mode: run.api.base.read_checkpoint(ROOT/learned[context, seed, bank, mode]['checkpoint']['path']) for mode in cfg['modes']}
                for source, packet in sorted(packets.items()):
                    guard(); kw = packet['x'], packet['scene'], packet['envelope'], packet['scales']
                    prediction = {mode: run.api.predict(state, *kw) for mode, state in states.items()}
                    choices = {}
                    for mode, p in prediction.items():
                        choices[mode], _ = run.api.base.select_actions(p, packet['allowed'], arm=bank)
                        old = run.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{mode}_fit.json'
                        if run.parent.fingerprint(choices[mode]) != json.loads(old.read_text())['action_sha256'][bank]:
                            raise ValueError('Primary frozen fitting action changed')
                    for recording in sorted(set(packet['recordings'])):
                        at = packet['recordings'] == recording
                        args = packet['y'][at], packet['allowed'][at]
                        p, q = [prediction[m][at] for m in cfg['modes']]
                        a, b = [choices[m][at] for m in cfg['modes']]
                        left = compare_regions(p, *args, a, b, packet['scales'][at], run.api.base.ARMS[bank])
                        right = compare_regions(q, *args, b, a, packet['scales'][at], run.api.base.ARMS[bank])
                        check_pair(left, right)
                        records.append(dict(context=context, source=source, seed=seed, bank=bank, recording=str(recording),
                            diagnosis=dict(known_only=left, completion_bounds=right)))
                beat(state='completion_region_progress', context=context, seed=seed, bank=bank,
                     paired_record_views=len(records), total=888)
        del packets
    identities = {(r['context'], r['source'], r['seed'], r['bank'], r['recording']) for r in records}
    if len(records) != 888 or len(identities) != 888:
        raise ValueError('All 888 distinct paired fitting record/head/bank views required')
    mappings = dict(all_legal=dict(known_only='all_legal', completion_bounds='all_legal'),
        fixed_known_only_selection=dict(known_only='selected', completion_bounds='frozen_teacher_region'),
        fixed_completion_selection=dict(known_only='frozen_teacher_region', completion_bounds='selected'))
    summary = {bank: {region: {mode: conditional_summary([r for r in records if r['bank'] == bank], mode, pool)
        for mode, pool in mapping.items()} for region, mapping in mappings.items()} for bank in cfg['banks']}
    value = dict(result_source='fresh_fitting_only_fixed_region_diagnosis_cached_verified_new_endpoints',
        records=records, summary=summary, diagnostic_bindings=json.loads(REG.read_text())['bindings'],
        primary_summary_sha256=run.sha(run.PUBLIC/'summary.json'), new_optimizer_updates=0,
        future_labels_used_for_choices=False, held_scored=False, independent_roles_opened=False,
        learned_proxy_moments_claimed_calibrated=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    content = json.dumps(value, indent=2).encode()+b'\n'; path = run.PUBLIC/'fitting_cost_diagnosis.json.gz'
    if verify:
        if gzip.decompress(path.read_bytes()) != content:
            raise ValueError('Exact paired region diagnosis differs')
        run.once(run.PUBLIC/'fitting_cost_verification.json', dict(exact_replay=True, paired_record_views=888,
            archive_sha256=run.sha(path), summary_sha256=run.sha(run.PUBLIC/'fitting_cost_summary.json')))
    elif path.exists():
        raise ValueError('Completed fitting diagnosis immutable')
    else:
        path.write_bytes(gzip.compress(content, mtime=0))
        run.once(run.PUBLIC/'fitting_cost_summary.json', summary)
    run.json_write(run.PRIVATE/('cost_mechanism_verify_runtime.json' if verify else 'cost_mechanism_runtime.json'),
        dict(pid=run.os.getpid(), seconds=time.monotonic()-started, peak_RSS_bytes=run.rss_bytes()))
    beat(state='completion_region_verified' if verify else 'completion_region_complete', paired_record_views=888)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify')); args = parser.parse_args()
    run.PRIVATE.mkdir(parents=True, exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        else:
            execute(args.phase == 'verify')
