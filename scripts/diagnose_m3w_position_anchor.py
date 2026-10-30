"""Post-hoc,full-cohort squared-error accounting with no model changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_position_anchor as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_position_anchor_mechanism import decompose
from src.evaluation.m3w_metric_bundle import pack

np = parent.np; NAME = parent.NAME+'/mechanism'
PUBLIC = parent.PUBLIC/'mechanism'; PRIVATE = parent.PRIVATE/'mechanism'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_position_anchor_mechanism.py',
    ROOT/'tests/test_m3w_position_anchor_mechanism.py', PUBLIC/'protocol.md')
sha, once, write = parent.sha, parent.once, parent.write


def admission(register=False):
    cfg, values = parent.admission(); check = json.loads((parent.PUBLIC/'verification.json').read_text())
    if not check['all24groups_exact'] or check['summary_sha256'] != sha(parent.PUBLIC/'summary.json'):
        raise ValueError('Complete parent negative readout/replay required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_complete_sha256=sha(parent.PUBLIC/'complete.json'),
        parent_summary_sha256=sha(parent.PUBLIC/'summary.json'), parent_verification_sha256=sha(parent.PUBLIC/'verification.json'),
        role='posthoc_exposed_SOURCE_label_side_arithmetic_only', own_arithmetic_values_not_yet_read=True,
        models_policies_thresholds_changed=False, new_optimizer_updates=0, groups=24,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Arithmetic definition changed')
            parent.base.committed(ROOT/rel)
        parent.base.committed(REG)
    return cfg, values


def summarize(rows):
    contexts = json.loads(parent.REG.read_text())['contexts']
    if len(rows) != 24 or {(r['context'], r['source']) for r in rows} != {(c, s) for c, ss in contexts.items() for s in ss}:
        raise ValueError('All24 fitting groups required')
    windows = {}
    for k in ('4', '8'):
        moments = {}
        for field in ('direct', 'cross', 'squared_shift'):
            sources = {}
            for r in rows:
                v = r['windows'][k]['terms'][field]
                if v is not None: sources.setdefault(r['source'], []).append(v)
            means = {s: float(np.mean(v)) for s, v in sorted(sources.items())}
            moments[field] = dict(mean=float(np.mean(list(means.values()))) if means else None,
                by_source=means, supported_sources=len(means), required_sources=12,
                supported_views=sum(r['windows'][k]['terms'][field] is not None for r in rows), required_views=24)
        windows[k] = dict(moments=moments,
            known_row_views=sum(r['windows'][k]['known_rows'] for r in rows),
            unknown_row_views=sum(r['windows'][k]['unknown_rows'] for r in rows),
            squared_error_increased_row_views=sum(r['windows'][k]['squared_error_increased_rows'] for r in rows),
            active_past_acceleration_row_views=sum(r['windows'][k]['past_acceleration_alignment']['active_rows'] for r in rows),
            opposite_past_acceleration_row_views=sum(r['windows'][k]['past_acceleration_alignment']['opposite_direction_rows'] for r in rows),
            max_abs_identity_error=max((r['windows'][k]['max_abs_identity_error'] or 0) for r in rows))
    return dict(groups=24, all24_parent_readouts_exact=all(r['parent_readout_exact'] for r in rows), windows=windows,
        result_source='fresh_posthoc_error_identity_on_frozen_exposed_SOURCE_predictions',
        new_training_updates=0, not_primary_ADE_or_noise_identification=True, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def execute(verify=False, resume=False):
    cfg, values = admission(); parent.base.torch.set_num_threads(4); parent.base.torch.set_num_interop_threads(1)
    started = time.monotonic()
    def beat(**event):
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row); print(json.dumps(row), flush=True)
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Complete accounting immutable')
    source, jobs = values[3:5]; previous = aggregate_groups(parent.NAME)
    old = aggregate_groups(NAME) if verify else {}; rows = []; refs = []
    for context, items in sorted(values[1]['contexts'].items()):
        packets = parent.prepared(context, items, values, beat)
        job = next(j for j in jobs if j['key'] == context.split('_controller')[0])
        for name, packet in sorted(packets.items()):
            key = context+'_'+name+'.json'; path = PUBLIC/'groups'/key; ids = packet['ids']
            if path.exists() and not verify:
                if not resume: raise ValueError('Explicit incomplete accounting resume required')
                row = json.loads(path.read_text())
                if row['registration_sha256'] != sha(REG): raise ValueError('Partial registration changed')
            else:
                h, origin = source['history'][ids], source['origin'][ids]
                eligible = parent.base.old.base.shared.eligible(packet['args'])
                result = parent.evaluation.evaluate(h, origin, source['target_eval'][ids], source['valid'][ids], packet, eligible, job['design']['easy_cut'])
                if result != previous[key]['result'] or parent.base.fingerprint(h) != previous[key]['input_history_sha256']:
                    raise ValueError('Frozen full parent result/history changed')
                windows = {}
                for k in (4, 8):
                    paths = [np.where(eligible[:, None, None], parent.model.predict(h, window=k, anchor=a)-origin[:, None],
                        packet['floor_rollout'])+origin[:, None] for a in parent.model.ANCHORS]
                    windows[str(k)] = decompose(*paths, source['target_eval'][ids], source['valid'][ids], h, packet['scales'])
                row = dict(context=context, source=name, windows=windows, registration_sha256=sha(REG), parent_readout_exact=True)
                if verify:
                    if row != old[key]: raise ValueError('Full diagnostic replay differs')
                else: once(path, row)
            rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='group_complete', groups=len(rows), total=24)
    summary = summarize(rows)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary differs')
        once(PUBLIC/'verification.json', dict(all24groups_exact=True, all24_parent_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs, summary_sha256=sha(PUBLIC/'summary.json')))
        pack(ROOT, PUBLIC)
    once(PUBLIC/('verification_runtime.json' if verify else 'readout_runtime.json'), dict(pid=os.getpid(),
        seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all24_verified' if verify else 'all24_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(True) if a.phase == 'register' else execute(a.phase == 'verify', a.resume)
