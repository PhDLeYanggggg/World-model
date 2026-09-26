"""Registered fitting-only cap accounting with per-view resume and exact replay."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 Python before importing numerical runtimes')
from scripts import run_m3w_european_risk_conditioned_residual as parent
from src.evaluation.m3w_fixed_cap_diagnostic import diagnose_bank
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_fixed_cap_diagnostic_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_fixed_cap_diagnostic_v1'
CONFIG = 'configs/m3w_european_fixed_cap_diagnostic_v1.json'
FILES = [CONFIG, 'src/evaluation/m3w_fixed_cap_diagnostic.py',
         'tests/test_m3w_fixed_cap_diagnostic.py',
         'scripts/run_m3w_european_fixed_cap_diagnostic.py',
         str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    _, pid = parent.registration()
    v = json.loads((parent.PUBLIC/'verification.json').read_text())
    assert v['all_passed']
    for p, h in v['artifacts'].items():
        assert digest(parent.PUBLIC/p) == h
    for p, h in v['source_bindings'].items():
        assert digest(ROOT/p) == h
    assert cfg['views'] == 144
    assert cfg['banks'] == ['original_in_sample', 'oof', 'in_sample_next', 'in_sample_prev']
    assert cfg['weightings'] == ['uniform_positive_envelope', 'registered_risk_weights']
    assert cfg['caps'] == ['frozen_harm_cap', 'causal_envelope_cap']
    assert not any(cfg[k] for k in ('held_labels_evaluated', 'new_model_fit', 'threshold_refit',
        'selection_access', 'reserved_calibration_access', 'confirmation_access',
        'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create:
        immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        parent.base.previous.require_committed(path)
    return cfg, identity


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def measure_view(v):
    bank, y, w = parent.banks(v)
    assert v['outer'] not in set(v['sites']) and len(set(v['sites'])) == 3
    out = parent.PRIVATE/'support'/v['tag']
    receipt = json.loads((out/'complete.json').read_text())
    for a in receipt['artifacts'].values():
        assert artifact(ROOT/a['path']) == a
    with np.load(out/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], v['ids'])
        original = z['scores'].copy()
    pmap = {'original_in_sample': original}
    for variant in ('oof', 'in_sample_next', 'in_sample_prev'):
        p, _, _, _ = parent.nested.method.assemble(bank, v['raw'], v['cv'], v['sites'], v['outer'], variant)
        pmap[variant] = p
    rows = {key: diagnose_bank(p, y, v['env'], w) for key, p in pmap.items()}
    support_rows = json.loads((out/'diagnostic.json').read_text())['rows']
    for row in support_rows:
        assert row['target_sha256'] == array_hash(y)
        assert row['fitting_ids_sha256'] == array_hash(v['ids'])
        assert row['inner_score_sha256'] == array_hash(pmap[row['variant']])
        assert row['original_fitting_score_sha256'] == array_hash(original)
    return dict(tag=v['tag'], pair=v['pair'], fitting_localities=sorted(set(v['sites'])),
        excluded_outer=v['outer'], fitting_ids_sha256=array_hash(v['ids']),
        target_sha256=array_hash(y), envelope_sha256=array_hash(v['env']),
        prediction_sha256={k: array_hash(p) for k, p in pmap.items()}, rows=rows,
        result_source='fresh_run_fitting_only_accounting_cached_verified_scores',
        held_labels_evaluated=False, new_model_fit=False, conditional_bias_identified=False)


def run(cfg, identity, *, pilot=False, verify=False):
    parent.check_sources(identity['parent'])
    parent.check_freeze()
    start = time.monotonic()
    refs = []
    for v in parent.nested.views(identity['parent']['parent']['parent']):
        if shutil.disk_usage(PRIVATE).free < 10*2**30:
            raise OSError('Preserve 10 GiB; per-view resume available')
        path = PRIVATE/'views'/(v['tag']+'.json')
        receipt = PRIVATE/'receipts'/(v['tag']+'.json')
        if path.exists() and receipt.exists() and not verify and not pilot:
            saved = json.loads(receipt.read_text())
            assert saved['diagnostic'] == artifact(path)
            assert saved['registration'] == artifact(PUBLIC/'registration_lock.json')
            row = json.loads(path.read_text())
            assert row['registration'] == artifact(PUBLIC/'registration_lock.json')
            assert row['diagnostic']['fitting_ids_sha256'] == array_hash(v['ids'])
        else:
            row = dict(registration=artifact(PUBLIC/'registration_lock.json'), diagnostic=measure_view(v))
            if verify:
                assert json.loads(path.read_text()) == row
                assert json.loads(receipt.read_text()) == dict(
                    registration=artifact(PUBLIC/'registration_lock.json'), diagnostic=artifact(path))
            elif not pilot:
                immutable_json(path, row)
                immutable_json(receipt, dict(registration=artifact(PUBLIC/'registration_lock.json'),
                                             diagnostic=artifact(path)))
        if pilot:
            doc = dict(seconds_including_source_loading=time.monotonic()-start,
                views=1, banks=4, identities=16, target_role='fitting_only',
                new_model_fit=False, result=row)
            immutable_json(PRIVATE/'pilot.json', doc)
            beat('pilot_complete', seconds=doc['seconds_including_source_loading'])
            return
        refs.append(artifact(path))
        beat('view_replayed' if verify else 'view_complete', views=len(refs), tag=v['tag'])
    assert len(refs) == cfg['views']
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), views=refs,
               exact_identities=2304, held_labels_evaluated=False, new_model_fit=False,
               no_new_inference_features=True, selection_access=False,
               reserved_calibration_access=False, confirmation_access=False,
               independent_confirmation=False, deployment_changed=False)
    immutable_json(PUBLIC/('replay.json' if verify else 'completion.json'), doc)
    beat('replay_complete' if verify else 'diagnostic_complete', views=len(refs),
         seconds_including_source_loading=time.monotonic()-start)


def report(cfg):
    complete = json.loads((PUBLIC/'completion.json').read_text())
    records = []
    cells = []
    for ref in complete['views']:
        assert artifact(ROOT/ref['path']) == ref
        d = json.loads((ROOT/ref['path']).read_text())['diagnostic']
        for bank, values in d['rows'].items():
            for weighting, r in values.items():
                h, e = r['frozen_harm_cap'], r['causal_envelope_cap']
                records.append(dict(tag=d['tag'], pair=d['pair'], bank=bank, weighting=weighting,
                    rows=h['rows'], mse=h['mse'], floor=h['projection_floor'],
                    floor_share_percent=h['floor_share_percent'], distance=h['distance_to_projection'],
                    cross=h['boundary_cross_term'], above_cap_fraction=h['fraction_above_cap'],
                    mean_target_minus_cap=h['mean_target_minus_cap'],
                    envelope_floor=e['projection_floor'], identity_error=h['max_identity_error']))
                for b in r['risk_bins']:
                    cells.append(dict(tag=d['tag'], pair=d['pair'], bank=bank, weighting=weighting, **b))
    assert len(records) == 1152 and len(cells) == 3456
    summaries = []
    for pair in cfg['pairs']:
        for bank in cfg['banks']:
            for weighting in cfg['weightings']:
                rows = [r for r in records if (r['pair'], r['bank'], r['weighting']) == (pair, bank, weighting)]
                rr = [r['floor_share_percent'] for r in rows if r['floor_share_percent'] is not None]
                cc = [c for c in cells if (c['pair'], c['bank'], c['weighting']) == (pair, bank, weighting)]
                assert len(rows) == 72 and len(cc) == 216
                summaries.append(dict(pair=pair, bank=bank, weighting=weighting, dependent_views=72,
                    median_floor_share_percent=float(np.median(rr)) if rr else None,
                    min_floor_share_percent=min(rr) if rr else None,
                    max_floor_share_percent=max(rr) if rr else None,
                    max_envelope_floor=max(r['envelope_floor'] for r in rows),
                    median_above_cap_fraction=float(np.median([r['above_cap_fraction'] for r in rows])),
                    positive_whole_view_cap_gap=sum(r['mean_target_minus_cap'] > 0 for r in rows),
                    nonempty_risk_bins=sum(c['rows'] > 0 for c in cc),
                    positive_risk_bin_cap_gap=sum(c['mean_target_minus_cap'] is not None and c['mean_target_minus_cap'] > 0 for c in cc)))
    doc = dict(result_source='fresh_run_fitting_only_diagnostic_cached_verified_inputs',
        summaries=summaries, all_views=144, exact_identities=2304,
        maximum_identity_error=max(r['identity_error'] for r in records),
        conditional_bias_identified=False, independent_confirmation=False,
        deployment_changed=False, new_model_fit=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC/'summary.json', doc)
    for name, rows in (('view_metrics.csv', records), ('risk_bin_metrics.csv', cells)):
        with (PUBLIC/name).open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['# Fixed-Cap Fitting Diagnostic', '', '## Material Passport', '',
        'Fresh algebraic diagnosis, cached verified inputs, no new fitting or held readout.',
        '144 dependent source-development fitting views; unknown labels excluded.', '',
        '| Inputs / bank / weights | Median floor share (%) | Range (%) | Mean-gap positive views /72 | Positive score bins / nonempty |',
        '|---|---:|---|---:|---|']
    for r in summaries:
        fmt = lambda value: f'{value:.4f}' if value is not None else 'undefined'
        lines.append(f"| {r['pair']} / {r['bank']} / {r['weighting']} | {fmt(r['median_floor_share_percent'])} | [{fmt(r['min_floor_share_percent'])}, {fmt(r['max_floor_share_percent'])}] | {r['positive_whole_view_cap_gap']} | {r['positive_risk_bin_cap_gap']}/{r['nonempty_risk_bins']} |")
    lines += ['', 'A floor uses realized labels and is not attainable predictive improvement.',
        'Ranges and medians summarize dependent views, not confidence intervals or independent studies.',
        'Positive fitting-bin mean gaps are descriptive resubstitution moments, not conditional-bias tests.',
        'No trajectory/policy metric changed; no independent calibration, confirmation or promotion.',
        'Obs8/pred12 annotation steps and detector pixels only; Stage5C and SMC off.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(doc, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['register', 'pilot', 'run', 'verify', 'report'])
    args = ap.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, identity = registration(args.phase == 'register')
        if args.phase in ('pilot', 'run', 'verify'):
            run(cfg, identity, pilot=args.phase == 'pilot', verify=args.phase == 'verify')
        elif args.phase == 'report':
            report(cfg)


if __name__ == '__main__':
    main()
