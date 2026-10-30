"""Matched original/symmetric/upper budget-score cost learning on frozen banks."""
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
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_active_bank_cost as frozen
from src.world_model import m3w_budget_score_cost as api

NAME = 'european_budget_score_cost_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_budget_score_cost_v1.json'
REG = PUBLIC / 'registration.json'
sha, once, write = frozen.sha, frozen.once, frozen.write
FILES = (Path(__file__), CONFIG, ROOT / 'src/world_model/m3w_budget_score_cost.py',
         ROOT / 'tests/test_m3w_budget_score_cost.py', ROOT / 'tests/test_m3w_budget_score_runner.py',
         PUBLIC / 'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    pc, values, _, store, _ = frozen.admission()
    models, banks = frozen.models(), frozen.cache_refs()
    check = json.loads((frozen.PUBLIC / 'verification.json').read_text())
    if (not check['all72groups_exact'] or not check['all_old_predictions_actions_record_scores_exact']
            or check['summary_sha256'] != sha(frozen.PUBLIC / 'summary.json')
            or check['complete_sha256'] != sha(frozen.PUBLIC / 'complete.json')):
        raise ValueError('Complete exact previous54head/72group experiment required')
    if (cfg['training'] != pc['training'] or cfg['variants'] != pc['variants'] or cfg['seeds'] != pc['seeds']
            or cfg['objectives'] != list(api.OBJECTIVES) or cfg['risk_budget'] != .02
            or cfg['budget_score_weight'] != 1.
            or cfg['score_underprediction_weights'] != dict(symmetric_score=1., upper_score=4.)
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed objectives, complete parent design, original2% rule and closed roles required')
    parent_reg = json.loads(frozen.REG.read_text())
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_training_freeze_sha256=sha(frozen.PUBLIC / 'training_freeze.json'),
        parent_complete_sha256=sha(frozen.PUBLIC / 'complete.json'),
        parent_verification_sha256=sha(frozen.PUBLIC / 'verification.json'),
        frozen_bank_sha256=sha(frozen.PUBLIC / 'bank_complete.json'),
        contexts=parent_reg['contexts'], fits=162, exact_original_controls=54,
        primary_optimizer_updates=324000, extra_full_refit_updates=2000, pilot_updates=420,
        parameters=30405, causal_fields=462, representation='history',
        objective_factor='continuous_five_moments_plus_original_budget_score_residual',
        score_target='positive_harm_minus_0.02_reference_and_easy_positive_harm_minus_0.02_easy_reference',
        score_normalization='unchanged_fitting_physical_RMS_quadrature_not_a_confidence_bound',
        risk_budget=.02, prediction_decoder_unchanged=True, new_forecasters=False,
        fitting_roles_only=True, held_scored=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered budget-score implementation changed: ' + rel)
            frozen.engine.committed(ROOT / rel)
        frozen.engine.committed(REG)
    return cfg, values, models, store, banks


def runtime(cfg):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume the same checkpoints')))
    def guard():
        if time.monotonic() - start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; retain full162fit roster and resume')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private checkpoint cap exceeded')
    def beat(**event):
        guard()
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', value)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, workers=0,
         resource_probing=False)
    return start, beat, guard


def checkpoint(context, seed, variant, objective_arm, phase='train'):
    return PRIVATE / phase / f'{context}_seed{seed}_{variant}_{objective_arm}.pt.gz'


def fit(data, cfg, context, seed, variant, objective_arm, path, beat, guard, *, resume=False, stop_at=None):
    identity = dict(registration_sha256=sha(REG), context=context, variant=variant,
        objective_arm=objective_arm, frozen_bank_sha256=sha(frozen.PUBLIC / 'bank_complete.json'))
    return api.fit(data, objective_arm=objective_arm, variant=variant, settings=cfg['training'], seed=seed,
        identity=identity, path=path, heartbeat=lambda **kw: beat(context=context, variant=variant,
            objective_arm=objective_arm, **kw), guard=guard, resume=resume, stop_at=stop_at)


def pilot(resume=False):
    cfg, values, _, store, banks = admission()
    if (PUBLIC / 'pilot.json').exists():
        raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, _, original, changed = frozen.prepared_context(context, items, values, store, banks, beat)
    states, cells = [], []
    for objective_arm in cfg['objectives']:
        for variant in cfg['variants']:
            data = changed if variant == 'new_neural' else original
            path = checkpoint(context, 17, variant, objective_arm, 'pilot')
            if not path.exists():
                fit(data, cfg, context, 17, variant, objective_arm, path, beat, guard, stop_at=10)
            state = fit(data, cfg, context, 17, variant, objective_arm, path, beat, guard, resume=True, stop_at=20)
            direct = fit(data, cfg, context, 17, variant, objective_arm,
                checkpoint(context, 17, variant, objective_arm, 'pilot_direct'), beat, guard, resume=resume, stop_at=20)
            frozen.api.assert_scientific_equal(state, direct)
            if objective_arm == 'original':
                previous = frozen.api.fit(data, variant=variant, settings=cfg['training'], seed=17,
                    identity=dict(pilot_control=True), path=checkpoint(context, 17, variant, objective_arm, 'pilot_control'),
                    heartbeat=beat, guard=guard, resume=resume, stop_at=20)
                frozen.api.assert_scientific_equal(state, previous)
            if (state['parameters'] != 30405 or state['unknown_rows_with_loss'] or not state['unknown_rows_drawn']
                    or not any(not torch.equal(v, state['initial_model'][k]) for k, v in state['model'].items())):
                raise ValueError('Real complete-size update and original unknown sampling required')
            states.append(state)
            cells.append(dict(variant=variant, objective_arm=objective_arm, seconds=state['seconds'],
                exact_recovery=True, parameters=state['parameters'], updates=20,
                unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=state['unknown_rows_with_loss']))
    for other in states[1:]:
        frozen.api.assert_common_design(states[0], other)
    once(PUBLIC / 'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cells=cells,
        engineering_pass=True, all9recoveries_exact=True, all3original_controls_exact=True,
        real_optimizer_updates=420, projected_primary_optimizer_seconds=float(np.median([r['seconds'] for r in cells])) * 100 * 162,
        projection_excludes_packet_preparation_full_refit_and_readout=True))
    beat(state='real_nine_cell_pilot_complete', updates=420)


def train(resume=False):
    cfg, values, legacy, store, banks = admission()
    if (PUBLIC / 'training_freeze.json').exists():
        raise ValueError('Completed162fit freeze immutable')
    if not json.loads((PUBLIC / 'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real nine-cell full-context pilot required')
    start, beat, guard = runtime(cfg)
    refs, neutral = [], 0
    first_context = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        op, npackets, original, changed = frozen.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            states = []
            for objective_arm in cfg['objectives']:
                for variant in cfg['variants']:
                    path = checkpoint(context, seed, variant, objective_arm)
                    receipt = PUBLIC / 'fits' / f'{context}_seed{seed}_{variant}_{objective_arm}.json'
                    if receipt.exists():
                        if not resume:
                            raise ValueError('Explicit --resume required for completed receipts')
                        row = json.loads(receipt.read_text())
                        if (row['registration_sha256'] != sha(REG) or row['objective_arm'] != objective_arm
                                or row['variant'] != variant or row['step'] != 2000
                                or row['checkpoint']['path'] != str(path.relative_to(ROOT))
                                or sha(path) != row['checkpoint']['sha256']):
                            raise ValueError('Completed score-objective receipt changed')
                        state = frozen.api.completion.base.read_checkpoint(path)
                    else:
                        data = changed if variant == 'new_neural' else original
                        state = fit(data, cfg, context, seed, variant, objective_arm, path, beat, guard,
                            resume=path.exists() and resume)
                        if objective_arm == 'original':
                            prior = frozen.api.completion.base.read_checkpoint(ROOT / legacy[context, seed, variant]['checkpoint']['path'])
                            frozen.api.assert_scientific_equal(state, prior)
                        if context == first_context and seed == 17 and objective_arm == 'upper_score' and variant == 'new_neural':
                            rp = checkpoint(context, seed, variant, objective_arm, 'full_refit')
                            refit = fit(changed, cfg, context, seed, variant, objective_arm, rp, beat, guard,
                                resume=rp.exists() and resume)
                            frozen.api.assert_scientific_equal(state, refit)
                            once(PUBLIC / 'first_full_refit.json', dict(scientific_state_exact=True, updates=2000,
                                context=context, seed=seed, variant=variant, objective_arm=objective_arm,
                                checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                        row = dict(context=context, seed=seed, variant=variant, objective_arm=objective_arm,
                            step=state['step'], parameters=state['parameters'], seconds=state['seconds'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            original_scientific_state_exact=objective_arm == 'original', draw_hash=state['draw_hash'],
                            unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=state['unknown_rows_with_loss'],
                            first_trace=state['trace'][0], final_trace=state['trace'][-1], input_hashes=state['input_hashes'],
                            actual_objective=state['objective'], registration_sha256=sha(REG),
                            result_source='fresh_native_arm64_budget_score_Torch_training',
                            held_scored=False, independent_roles_opened=False)
                        once(receipt, row)
                    if state['step'] != 2000 or state['parameters'] != 30405 or state['unknown_rows_with_loss']:
                        raise ValueError('Complete2k budget, exact head size and known-only loss required')
                    if objective_arm == 'original':
                        if not row['original_scientific_state_exact']:
                            raise ValueError('Original full-state control not verified')
                        neutral += 1
                    if states:
                        frozen.api.assert_common_design(states[0], state)
                    states.append(state)
                    refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                    beat(state='score_endpoint_complete', context=context, seed=seed, variant=variant,
                        objective_arm=objective_arm, fits=len(refs), total=162, original_exact=neutral)
        del op, npackets, original, changed, states
    refit = json.loads((PUBLIC / 'first_full_refit.json').read_text())
    original_path = checkpoint(first_context, 17, 'new_neural', 'upper_score')
    refit_path = checkpoint(first_context, 17, 'new_neural', 'upper_score', 'full_refit')
    if (len(refs) != 162 or neutral != 54 or not refit['scientific_state_exact']
            or refit['checkpoint_sha256'] != sha(original_path) or refit['refit_checkpoint_sha256'] != sha(refit_path)):
        raise ValueError('Complete162fits/54exact controls/full new refit required')
    once(PUBLIC / 'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs, count=162,
        primary_optimizer_updates=324000, extra_full_refit_updates=2000, pilot_updates=420,
        exact_original_controls=54, registration_sha256=sha(REG),
        parent_training_freeze_sha256=sha(frozen.PUBLIC / 'training_freeze.json'),
        first_full_refit_sha256=sha(PUBLIC / 'first_full_refit.json'),
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    beat(state='all162budget_score_heads_complete', primary_optimizer_updates=324000)


def models():
    path = PUBLIC / 'training_freeze.json'
    frozen.engine.committed(path)
    value = json.loads(path.read_text())
    if (value['count'] != 162 or value['exact_original_controls'] != 54
            or value['registration_sha256'] != sha(REG)
            or value['parent_training_freeze_sha256'] != sha(frozen.PUBLIC / 'training_freeze.json')
            or value['first_full_refit_sha256'] != sha(PUBLIC / 'first_full_refit.json')):
        raise ValueError('Complete committed162head freeze required')
    output = {}
    for ref in value['fits']:
        path = ROOT / ref['path']
        frozen.engine.committed(path)
        row = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT / row['checkpoint']['path']) != row['checkpoint']['sha256'] or row['step'] != 2000:
            raise ValueError('Frozen budget-score endpoint changed')
        key = row['context'], row['seed'], row['variant'], row['objective_arm']
        if key in output:
            raise ValueError('Repeated score endpoint')
        output[key] = row
    reg = json.loads(REG.read_text())
    expected = {(c, s, v, o) for c in reg['contexts'] for s in (17, 29, 43)
                for v in api.active.VARIANTS for o in api.OBJECTIVES}
    if set(output) != expected:
        raise ValueError('All162registered endpoints required')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        elif args.phase == 'pilot':
            pilot(args.resume)
        else:
            train(args.resume)
