"""Frozen-source opportunity attribution; no fitting or deployable oracle."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_protected_motion as prior
import numpy as np
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from scripts.report_m3w_european_cv_reference import require_verification
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_opportunity_diagnosis import causal_reasons, opportunity_ledger, risk_bands
from src.world_model.m3w_european_conditional_risk import fitting_support

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_opportunity_diagnosis_v1'
FILES = ('scripts/run_m3w_european_opportunity_diagnosis.py',
         'src/evaluation/m3w_opportunity_diagnosis.py', 'tests/test_m3w_opportunity_diagnosis.py',
         'outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1/diagnosis_plan.md')
INTERVENTION = prior.previous.prior.previous


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load():
    reg, data, oldid, designs, _ = prior.load()
    old = json.loads((prior.PUBLIC/'analysis.json').read_text())
    require_verification(prior.PUBLIC, old)
    accounting = json.loads((prior.PUBLIC/'accounting_audit.json').read_text())
    if not accounting['all_passed'] or accounting['analysis_sha256'] != digest(prior.PUBLIC/'analysis.json'):
        raise ValueError('Verified protected-motion accounting required')
    identity = dict(bindings={p: digest(ROOT/p) for p in FILES},
        previous_identity=oldid, previous_analysis_sha256=digest(prior.PUBLIC/'analysis.json'),
        previous_accounting_sha256=digest(prior.PUBLIC/'accounting_audit.json'),
        source_rows=len(data['sites']), scope='opened_source_training_localities_only',
        inference_policy=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'matrix.json', dict(identity=identity, seeds=reg['seeds'],
        policy_views=48, pointwise_targets=len(data['sites']), producer_comparisons=6,
        new_fitting=False, reserved_roles_opened=False))
    return reg, data, designs, old, identity


def assert_identity(identity):
    for name, sha in identity['bindings'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Frozen diagnosis changed: '+name)
    prior.assert_identity(identity['previous_identity'])
    if digest(prior.PUBLIC/'analysis.json') != identity['previous_analysis_sha256']:
        raise ValueError('Prior scientific readout changed')


def checked_prediction(key, ids, data, expected_training_sites):
    path = INTERVENTION.parent.PRIVATE/'predictions'/(key+'.json')
    receipt = json.loads(path.read_text())
    training = set(receipt['identity']['fit_sites'])
    if training != set(expected_training_sites) or training & set(data['sites'][ids]):
        raise ValueError('Producer exposed held source locality')
    prediction = INTERVENTION.prediction(key, ids)
    error, _ = native_errors(prediction.astype(float)+data['origin'][ids, None],
                            data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
    return error, dict(producer=key, prediction_sha256=receipt['sha256'],
        training_sites=sorted(training), evaluated_sites=sorted(set(data['sites'][ids])),
        row_ids_sha256=array_hash(ids), rows=len(ids))


def one_seed(seed, reg, data, designs, old, identity):
    n, sites = len(data['sites']), data['sites']
    roster = sorted(identity['previous_identity']['folds'])
    cv = np.asarray(data['baseline_ade'][:, 1])
    motion = np.linalg.norm(np.diff(data['history'], axis=1), axis=2).sum(1) > 0
    relevant = [(k, d, t) for k, d, t in designs if t['kind'] == 'complement' and t['seed'] == seed]
    seen = np.zeros(n, bool)
    errors = {name: np.full(n, np.nan) for name in ('neural', 'single_lower_fold', 'single_upper_fold')}
    errors['damping097'] = np.asarray(data['baseline_ade'][:, 3])
    easy, hard = np.zeros(n, bool), np.zeros(n, bool)
    lineage = []
    for key, design, ti in relevant:
        held = design['held_ids']
        if seen[held].any(): raise ValueError('Repeated source target')
        seen[held] = True
        easy[held] = (cv[held] > 0) & (cv[held] <= design['easy_cut'])
        hard[held] = cv[held] >= design['hard_cut']
        errors['neural'][held], receipt = checked_prediction(key, held, data, ti['fit_sites'])
        lineage.append(receipt)
        others = sorted(set(identity['previous_identity']['folds'].values())-{ti['fold']})
        for label, trained in zip(('single_lower_fold', 'single_upper_fold'), others):
            expected = [s for s in roster if identity['previous_identity']['folds'][s] == trained]
            errors[label][held], receipt = checked_prediction(f'single{trained}_seed{seed}', held, data, expected)
            lineage.append(receipt)
    if not seen.all(): raise ValueError('Missing source targets')
    def metric(model, reference):
        return paired_scene_metrics(model, reference, sites, expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
    candidates = {name: dict(raw_vs_CV=metric(e, cv),
                            hindsight_oracle_vs_CV=metric(np.minimum(e, cv), cv))
                  for name, e in errors.items()}
    producer_comparisons = {label: metric(errors['neural'], errors[label])
                            for label in ('single_lower_fold', 'single_upper_fold')}
    policies = {}
    artifact_map = {a['path']: a['sha256'] for a in old['controls']}
    for candidate in reg['candidates']:
        utility = np.zeros(n)
        for key, design, _ in relevant:
            r = prior.checked_head(key, 'utility', 'neural_underharm4', candidate, identity['previous_identity'])
            score = prior.scores(r, 'utility', design['held_ids'])
            utility[design['held_ids']] = (score[:, 0]-score[:, 1])/r['identity']['lineage']['cost_scale']
        for event in reg['events']:
            for arm in reg['arms']:
                moments = np.zeros((n, 2))
                for key, design, _ in relevant:
                    r = prior.checked_head(key, event, arm, candidate, identity['previous_identity'])
                    moments[design['held_ids']] = prior.scores(r, event, design['held_ids'])
                for guard in reg['support_controls']:
                    support, stored = np.zeros(n, bool), np.zeros(n, bool)
                    receipts = []
                    for key, design, _ in relevant:
                        held, fit = design['held_ids'], design['train_ids']
                        available = fitting_support(cv[fit], sites[fit])
                        name = f'{candidate}_{key}_{event}_{arm}_{guard}'
                        path = prior.PRIVATE/'decisions'/(name+'.json')
                        if digest(path) != artifact_map[str(path.relative_to(ROOT))]:
                            raise ValueError('Decision receipt changed')
                        r = json.loads(path.read_text())
                        if (r['identity']['identity'] != identity['previous_identity']
                                or r['identity']['source_support'] != available
                                or digest(ROOT/r['path']) != r['sha256'] or r['used_future_inputs']):
                            raise ValueError('Unverified decision or source support')
                        with np.load(ROOT/r['path'], allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['pointwise_ids'], held)
                            stored[held] = z['pointwise']
                        support[held] = guard == 'no_guard' or available['gate_available']
                        receipts.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))
                    why = causal_reasons(utility, moments, motion, support, budget=reg['predicted_risk_budget'])
                    np.testing.assert_array_equal(why == 5, stored)
                    name = f'{seed}_{candidate}_{event}_{arm}_{guard}'
                    ledgers = {name: opportunity_ledger(cv, errors[candidate], why, sites,
                        expected_scenes=roster, subset=mask, resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
                        for name, mask in (('all', None), ('easy', easy), ('hard', hard))}
                    original = old['policies'][name]['full']['ADE_vs_CV']
                    np.testing.assert_allclose(ledgers['all']['summary']['net_gain']['equal_locality'],
                        original['equal_scene_gain_percent'], rtol=1e-10, atol=1e-10)
                    for site in roster:
                        np.testing.assert_allclose(ledgers['all']['by_scene'][site]['contributions_percent']['net_gain'],
                            original['by_scene'][site]['gain_percent'], rtol=1e-10, atol=1e-10)
                    policies[name] = dict(ledgers=ledgers, risk_bands=risk_bands(cv, errors[candidate], utility, moments, why),
                        decision_receipts=receipts, causal_reasons_match_frozen_decisions=True)
                    beat('policy_attribution', trial=name)
    assert_identity(identity)
    return dict(identity=identity, seed=seed, producer_lineage=lineage,
                candidates=candidates, producer_comparisons=producer_comparisons, policies=policies)


def main():
    parser = argparse.ArgumentParser()
    for flag in ('prepare', 'run', 'verify'): parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()): parser.error('Explicit phase required')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg, data, designs, old, identity = load()
        beat('inputs_verified', rows=len(data['sites']), new_training=False)
        if args.run or args.verify:
            output = {}
            for seed in reg['seeds']:
                path = PRIVATE/f'seed{seed}.json'
                receipt = path.with_suffix('.receipt.json')
                if path.exists() and not args.verify:
                    r = json.loads(receipt.read_text())
                    if r['identity'] != identity or digest(path) != r['sha256']:
                        raise ValueError('Unverified resume artifact')
                    output[str(seed)] = json.loads(path.read_text())
                    beat('verified_seed_resume', seed=seed)
                else:
                    output[str(seed)] = one_seed(seed, reg, data, designs, old, identity)
                    immutable_json(path, output[str(seed)])
                    immutable_json(receipt, dict(identity=identity, sha256=digest(path)))
            result = dict(result_source='fresh_arithmetic_on_cached_verified_source_models', identity=identity,
                seeds=output, new_training=False, independent_reserved_readout=False,
                oracle_is_diagnostic_only=True, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
            assert_identity(identity)
            immutable_json(PUBLIC/'analysis.json', result)
            if args.verify:
                immutable_json(PUBLIC/'verification.json', dict(result_source='fresh_complete_diagnostic_recomputation',
                    analysis_sha256=digest(PUBLIC/'analysis.json'), exact_replay=True, new_training=False))
            beat('diagnosis_complete', seeds=len(output), policy_views=sum(len(x['policies']) for x in output.values()), verified=args.verify)
        beat('phase_complete')


if __name__ == '__main__': main()
