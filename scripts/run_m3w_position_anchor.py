"""Registered past-only position-anchor comparison on the full fitting cohort."""
import argparse
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
    raise RuntimeError('Native arm64 required before importing Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_cost as base
from scripts import run_m3w_signed_utility_allocation as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_position_anchor as model
from src.evaluation import m3w_position_anchor as evaluation
from src.evaluation.m3w_metric_bundle import pack

np = base.np
NAME = 'european_position_anchor_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_position_anchor_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_position_anchor.py',
    ROOT/'src/evaluation/m3w_position_anchor.py', ROOT/'tests/test_m3w_position_anchor.py',
    ROOT/'tests/test_m3w_position_anchor_readout.py', ROOT/'tests/test_m3w_position_anchor_runner.py', PUBLIC/'protocol.md')
sha, once, write = base.sha, base.once, base.json_write


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); values = base.admission()
    check = json.loads((previous.PUBLIC/'verification.json').read_text())
    if (cfg['windows'] != [4, 8] or cfg['anchors'] != list(model.ANCHORS) or cfg['risk_budget'] != .02
            or cfg['groups'] != 24 or not check['all72groups_exact']
            or check['summary_sha256'] != sha(previous.PUBLIC/'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified preceding failure,unchanged cohort/budget and closed roles required')
    contexts = {c: [r['packet']['identity']['source'] for r in items] for c, items in values[1]['contexts'].items()}
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        original_source_registration_sha256=sha(base.REG), preceding_summary_sha256=sha(previous.PUBLIC/'summary.json'),
        preceding_verification_sha256=sha(previous.PUBLIC/'verification.json'), contexts=contexts,
        changed_factor='position_anchor_only_same_OLS_velocity', primary='gated_OLS8_fitted_minus_raw_completion_lower',
        own_outcomes_read_at_registration=False, groups=24, critic_seeds_not_repeated=True,
        optimizer_updates=0, result_role='exposed_SOURCE_fitting_only', risk_budget=.02,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Registered position-anchor experiment changed:'+rel)
            base.committed(ROOT/rel)
        base.committed(REG)
    return cfg, values


def runtime(cfg):
    base.torch.set_num_threads(4); base.torch.set_num_interop_threads(1); started = time.monotonic()
    def beat(**event):
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row); print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, workers=0,
        new_neural_training=False, resource_probing=False)
    return started, beat


def prepared(context, items, values, beat):
    source, jobs, identity = values[3:6]
    scopes = base.prepare_context(context, items, source, jobs, identity, beat, include_held=False)
    if set(scopes) != {'fit'}: raise ValueError('Fitting-only packets required')
    return scopes['fit']


def pilot():
    cfg, values = admission(); start, beat = runtime(cfg)
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Complete pilot immutable')
    context, items = next(iter(sorted(values[1]['contexts'].items())))
    packets = prepared(context, items, values, beat); source = sorted(packets)[0]; packet = packets[source]
    ids = packet['ids'][:cfg['pilot_rows']]; history = values[3]['history'][ids]
    origin = values[3]['origin'][ids]; cut = len(ids)//2; paths = []
    for k in cfg['windows']:
        raw = model.predict(history, window=k, anchor='raw_last')-origin[:, None]
        np.testing.assert_array_equal(raw, packet['rollouts'][:len(ids), 4 if k == 4 else 5])
        paths.append(model.predict(history, window=k, anchor='causal_fitted'))
    checkpoint = PRIVATE/'pilot_chunks.npz'
    with checkpoint.open('wb') as stream: np.savez(stream, **{str(k): model.predict(history[:cut], window=k, anchor='causal_fitted') for k in cfg['windows']})
    with np.load(checkpoint, allow_pickle=False) as saved:
        for i, k in enumerate(cfg['windows']):
            reconstructed = np.concatenate((saved[str(k)], model.predict(history[cut:], window=k, anchor='causal_fitted')))
            np.testing.assert_array_equal(paths[i], reconstructed)
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rows=len(ids), context=context, source=source,
        raw_prediction_controls_exact=True, checkpoint_chunk_reconstruction_exact=True,
        own_future_outcomes_scored=False, old_packet_labels_checked=True,
        full_training_or_interrupted_process_claim=False, engineering_pass=True))
    beat(state='real_past_only_pilot_complete')


def execute(verify=False, resume=False):
    cfg, values = admission(); start, beat = runtime(cfg)
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real pilot required')
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Complete experiment immutable')
    source, jobs = values[3:5]; controls = aggregate_groups('european_query_unit_critic_v1')
    prior = aggregate_groups(NAME) if verify else {}; rows = []; refs = []; all_ids = set(); records = set()
    for context, items in sorted(values[1]['contexts'].items()):
        packets = prepared(context, items, values, beat)
        job = next(j for j in jobs if j['key'] == context.split('_controller')[0])
        for name, packet in sorted(packets.items()):
            key = context+'_'+name+'.json'; path = PUBLIC/'groups'/key; ids = packet['ids']
            control = controls[f'{context}_{name}_seed17_fit.json']
            if (base.fingerprint(ids) != control['ids_sha256'] or len(ids) != control['result']['rows']
                    or int(np.isfinite(packet['floor_ade']).sum()) != control['result']['known_rows']):
                raise ValueError('Original row identity or future-label support changed')
            if path.exists() and not verify:
                if not resume: raise ValueError('Explicit incomplete-roster resume required')
                row = json.loads(path.read_text())
                if row['registration_sha256'] != sha(REG) or row['ids_sha256'] != base.fingerprint(ids):
                    raise ValueError('Partial group changed')
            else:
                eligible = base.old.base.shared.eligible(packet['args'])
                result = evaluation.evaluate(source['history'][ids], source['origin'][ids], source['target_eval'][ids],
                    source['valid'][ids], packet, eligible, job['design']['easy_cut'])
                row = dict(context=context, source=name, seed=43, result=result, registration_sha256=sha(REG),
                    ids_sha256=base.fingerprint(ids), input_history_sha256=base.fingerprint(source['history'][ids]),
                    identity_control_exact=True, held_scored=False, independent_roles_opened=False)
                if verify:
                    if row != prior[key]: raise ValueError('Separate full position-anchor replay differs')
                else: once(path, row)
            all_ids.update(map(int, ids)); records.update((name, str(r)) for r in packet['recordings'])
            rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='group_complete', groups=len(rows), total=24, context=context, source=name)
    summary = evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    if len(all_ids) != 149510 or len(records) != 74: raise ValueError('Full149510-row74-record cohort must remain')
    summary.update(unique_target_rows=len(all_ids), unique_recordings=len(records), record_context_views=sum(len(r['result']['recordings']) for r in rows))
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Canonical summary differs')
        once(PUBLIC/'verification.json', dict(all24groups_exact=True, all48_raw_controls_exact=True,
            all24_identity_controls_exact=True, summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs, summary_sha256=sha(PUBLIC/'summary.json')))
        pack(ROOT, PUBLIC)
    once(PUBLIC/('verification_runtime.json' if verify else 'readout_runtime.json'), dict(pid=os.getpid(),
        seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all24_verified' if verify else 'all24_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.phase == 'register': admission(True)
        elif a.phase == 'pilot': pilot()
        else: execute(a.phase == 'verify', a.resume)
