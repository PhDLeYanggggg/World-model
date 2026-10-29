"""Fixed fitting-only output-amplitude controls, not another cost-head fit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import diagnose_m3w_cost_bank_factorization as previous
from src.evaluation import m3w_forecast_amplitude as api
from src.evaluation import m3w_multiaction_cost_readout as readout
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets

parent = previous.parent
base = parent.parent.parent
NAME = 'european_forecast_amplitude_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_forecast_amplitude_v1.json'; REG = PUBLIC/'registration.json'
sha, once, json_write = previous.sha, previous.once, previous.json_write
FILES = (CONFIG, Path(__file__), ROOT/'src/evaluation/m3w_forecast_amplitude.py',
    ROOT/'tests/test_m3w_forecast_amplitude.py', PUBLIC/'protocol.md', PUBLIC/'prior_controls.md')


def load():
    cfg = json.loads(CONFIG.read_text()); _, values = previous.admission()
    if (cfg['amplitudes'] != [1., .5, .25, .125, .0625] or cfg['risk_budget'] != .02
            or cfg['seeds'] != [17, 29, 43] or cfg['families'] != ['forest', 'neural']
            or cfg['held_scored'] or cfg['independent_roles_opened']):
        raise ValueError('Fixed causal fitting controls and unchanged roles required')
    verified = json.loads((previous.PUBLIC/'verification.json').read_text())
    if not verified['exact_replay'] or verified['summary_sha256'] != sha(previous.PUBLIC/'summary.json'):
        raise ValueError('Completed bank-factorization replay required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(previous.REG), parent_verification_sha256=sha(previous.PUBLIC/'verification.json'),
        parent_summary_sha256=sha(previous.PUBLIC/'summary.json'), groups=144, amplitude_views=720,
        contexts=sorted(values[2]['contexts']), seeds=cfg['seeds'], families=cfg['families'],
        amplitudes=cfg['amplitudes'], scope='original_fitting_only',
        transformed_costs='convex_surrogates_not_refitted_expected_moments_or_calibrated_risk',
        all_banks_same_amplitude_grid=True, held_scored=False, new_optimizer_updates=0, new_trees=0,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, values


def admission():
    cfg, values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered amplitude asset changed: ' + rel)
        base.committed(ROOT/rel)
    base.committed(REG)
    return cfg, values


def evaluation_packet(packet, paths, allowed, source, design):
    ids = packet['ids']; ade, fde, labels, envelope = [], [], [], []
    for candidate in paths.swapaxes(0, 1):
        a, f = base.native_errors(candidate + source['origin'][ids, None], source['target_eval'][ids],
                                  source['valid'][ids], np.ones(len(ids)))
        ade.append(a); fde.append(f)
        labels.append(base.targets(packet['cv_ade'], packet['floor_ade'], a, design['easy_cut']))
        envelope.append(base.causal_bank_envelope(packet['rollouts'][:, 1], packet['floor_rollout'], candidate))
    return dict(packet, rollouts=paths, allowed=allowed, ade=np.stack(ade, 1), fde=np.stack(fde, 1),
                y=np.stack(labels, 1), envelope=np.stack(envelope, 1))


def paired_outputs(left_packet, left, right_packet, right):
    a = readout.chosen_values(left_packet['ade'], left_packet['floor_ade'], left)
    b = readout.chosen_values(right_packet['ade'], right_packet['floor_ade'], right)
    known = np.isfinite(a); np.testing.assert_array_equal(known, np.isfinite(b))
    pa = readout.chosen_values(left_packet['rollouts'], left_packet['floor_rollout'], left)
    pb = readout.chosen_values(right_packet['rollouts'], right_packet['floor_rollout'], right)
    envelope = np.linalg.norm(pa - pb, axis=-1).max(1)
    delta = float((a[known] - b[known]).sum()); mass = float(envelope[~known].sum())
    reference = float(left_packet['floor_ade'][known].sum())
    return dict(known_pp=100*delta/reference if reference > 0 else None,
        lower_pp=100*(delta-mass)/reference if reference > 0 else None,
        upper_pp=100*(delta+mass)/reference if reference > 0 else None,
        known_delta_mass=delta, unknown_disagreement_mass=mass,
        full_reference_mass_counted_once=reference)


def paired_control(original, before, before_rank, fractional, after, after_rank):
    a, b = readout.matched_choices(before, after, before_rank, after_rank,
        original['recordings'], original['frames'], original['ids'])
    return dict(full=paired_outputs(original, before, fractional, after),
        matched=paired_outputs(original, a, fractional, b), matched_count=int((a >= 0).sum()),
        matched_role='utility_only_not_safety')


def safety(rows, name):
    ps = [r['result']['policies'][name]['policy'] for r in rows]
    records = [rec[name] for r in rows for rec in r['result']['recordings'].values()]
    return dict(required_views=len(ps), supported_views=sum(p['budget_status']['supported'] for p in ps),
        required_record_views=len(records), supported_record_views=sum(p['budget_status']['supported'] for p in records),
        undefined_easy_record_views=sum(p['easy_selected_risk_upper'] is None for p in records),
        known_all_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk'] > .02+1e-12 for p in records),
        known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in records),
        selected=sum(p['selected_count'] for p in ps), selected_unknown=sum(p['selected_unknown'] for p in ps))


def summarize(rows, cfg, pc):
    results = {}
    expected = {(r['context'], r['source'], r['seed']) for r in rows if r['family'] == 'forest'}
    if len(rows) != 144 or len(expected) != 72:
        raise ValueError('Full original fitting roster required')
    for family in cfg['families']:
        selected = [r for r in rows if r['family'] == family]
        if len(selected) != 72 or {(r['context'], r['source'], r['seed']) for r in selected} != expected:
            raise ValueError('Both complete families required; do not drop unsafe views')
        results[family] = {}
        for alpha in cfg['amplitudes']:
            key = str(alpha)
            group = [dict(context=r['context'], source=r['source'], seed=r['seed'], result=r['amplitudes'][key]) for r in selected]
            names = list(group[0]['result']['policies'])
            results[family][key] = dict(groups=72,
                safety={name: safety(group, name) for name in names},
                accuracy={name: readout.interval(group, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent'], pc) for name in names},
                FDE={name: readout.interval(group, lambda r, n=name: r['policies'][n]['known_FDE_gain_floor_percent'], pc) for name in names},
                contrasts={name: {mode+'_'+bound: readout.interval(group,
                    lambda r, n=name, m=mode, b=bound: r['amplitude_contrasts'][n][m][b], pc)
                    for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
                    for name in group[0]['result']['amplitude_contrasts']},
                floor_contrasts={name: readout.interval(group,
                    lambda r, n=name: r['floor_contrasts'][n]['lower_pp'], pc) for name in names},
                subsets=pooled_subsets(group))
    return dict(result_source='fresh_fixed_fitting_output_controls_cached_verified_estimators', families=results,
        groups=144, amplitude_views=720, contexts=6, head_seeds=cfg['seeds'], amplitudes=cfg['amplitudes'],
        scope='fitting_resubstitution_only', transformed_costs_are_calibrated=False,
        held_scored=False, optimizer_updates=0, new_trees=0, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def execute(verify=False, resume=False):
    cfg, values = admission(); finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed readout immutable; use verify')
    # Share the tested runtime guard, but keep this experiment's lock and heartbeat private.
    previous.PRIVATE = PRIVATE
    started, beat, guard = previous.runtime(cfg)
    learned, neural = parent.models(), parent.parent.models()
    _, origin_refs = base.old.direct.ORIGINS.models(values[2])
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        scopes = parent.context_data(context, items, values, beat, held=False)
        if set(scopes) != {'fit'}:
            raise ValueError('No held data permitted')
        packets = scopes['fit']
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            states = dict(forest={a: parent.joblib.load(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in api.ARMS},
                neural={a: base.api.read_checkpoint(ROOT/neural[context, seed, a]['checkpoint']['path']) for a in api.ARMS})
            pr = states['forest']['physical_only']['preprocess']
            for bank in states.values():
                for state in bank.values():
                    base.api.exact(pr, state['preprocess'])
            origin = base.old.base.core.read_checkpoint(ROOT/origin_refs[context, seed, 'self']['checkpoint']['path'])
            original = base.old.direct.origin_predictions(origin, {s: p['args'] for s, p in packets.items()})
            for source, packet in sorted(packets.items()):
                for family, bank in states.items():
                    guard()
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{family}_fit.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit group resume required')
                    oldpath = previous.PUBLIC/'groups'/path.name
                    old = json.loads(oldpath.read_text())
                    predict = parent.api.predict if family == 'forest' else parent.parent.api.predict
                    kw = packet['x'], packet['scene'], packet['envelope'], packet['scales']
                    prediction = {a: predict(state, *kw) for a, state in bank.items()}
                    for name, p in prediction.items():
                        if parent.api.fingerprint(p) != old['prediction_sha256'][name]:
                            raise ValueError('Frozen endpoint cost predictions changed')
                    origin_take = base.old.base.actions(original[source], base.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    endpoint = api.decisions(prediction, packet['allowed'], packet['floor_rollout'], packet['rollouts'], 1.)
                    _, _, endpoint_choices, endpoint_ranks = endpoint
                    results, checks, hashes = {}, {}, {}
                    for alpha in cfg['amplitudes']:
                        key = str(alpha)
                        paths, legal, choices, ranks = api.decisions(prediction, packet['allowed'], packet['floor_rollout'], packet['rollouts'], alpha)
                        score_packet = packet if alpha == 1 else evaluation_packet(packet, paths, legal, values[3], job['design'])
                        checks[key] = api.check_convex_labels(packet['y'], score_packet['y'], alpha)
                        choices['frozen_origin_binary'] = np.where(origin_take & legal[:, 6], 6, -1)
                        ranks['frozen_origin_binary'] = np.where(choices['frozen_origin_binary'] >= 0, original[source][:, 0]-original[source][:, 1], 0.)
                        result = readout.group_readout(score_packet, choices, ranks, {})
                        if alpha == 1:
                            for name in (*api.ARMS, 'factorized', 'floor', 'frozen_origin_binary'):
                                if result['policies'][name] != old['result']['policies'][name]:
                                    raise ValueError('Endpoint neutral policy metrics changed: ' + name)
                                if parent.api.fingerprint(choices[name]) != old['action_sha256'][name]:
                                    raise ValueError('Endpoint neutral choices changed')
                        result['amplitude_contrasts'] = {name: paired_control(packet, endpoint_choices[name], endpoint_ranks[name],
                            score_packet, choices[name], ranks[name]) for name in endpoint_choices if name != 'floor'}
                        result['floor_contrasts'] = {name: paired_outputs(packet, endpoint_choices['floor'], score_packet, choice)
                                                    for name, choice in choices.items()}
                        result['subsets'] = subsets(score_packet, choices, job['design']['easy_cut'])
                        results[key] = result
                        hashes[key] = dict(rollouts=parent.api.fingerprint(paths),
                            allowed=parent.api.fingerprint(legal), choices={n: parent.api.fingerprint(c) for n, c in choices.items()})
                    value = dict(context=context, source=source, seed=seed, family=family, scope='fit',
                        amplitudes=results, numerical_convexity=checks, causal_output_hashes=hashes,
                        registration_sha256=sha(REG), parent_group_sha256=sha(oldpath),
                        result_source='fresh_fitting_output_controls_no_new_training', held_scored=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact output-control replay differs')
                    else:
                        once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='amplitude_control_progress', context=context, seed=seed, groups=len(rows), total=144)
        del scopes, packets
    summary = summarize(rows, cfg, values[1])
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or refs != json.loads(finish.read_text())['groups']:
            raise ValueError('Exact output-control summary differs')
        once(PUBLIC/'verification.json', dict(exact_replay=True, groups=144, amplitude_views=720,
                                             summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, groups=refs,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, held_scored=False,
            new_optimizer_updates=0, new_trees=0, independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='amplitude_control_verified' if verify else 'amplitude_control_complete', groups=144)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            load()
        else:
            execute(args.phase == 'verify', args.resume)
