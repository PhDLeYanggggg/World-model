"""Reconcile the reserved DroneCrowd role and exact Moving A predictor lineage."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_quarter_allocation as study
from scripts.run_m3w_dronecrowd_blind_preflight import role_sets
from src.data_unification.m3w_moving_forecast_data import array_hash, prepare
from src.world_model.m3w_european_source_forecast import fit_design

run, np, h = study.run, study.np, study.h
HOME = run.old.HOME
PUBLIC = HOME/'dronecrowd_exact_bank_v2'
REGISTRY = ROOT/'configs/m3w_external_source_reservations_v1.json'
GROUPS = HOME/'dronecrowd_grouping_v2/exclusion_constraints.json'
PROPOSAL = HOME/'dronecrowd_external_role_proposal_v1/alternatives.json'
FORECASTERS = HOME/'dronecrowd_three_seed_bank_v1/analysis.json'
SCREEN = HOME/'dronecrowd_combined_forecaster_coverage_v1/analysis.json'
SELF = HOME/'dronecrowd_blind_preflight_v1/kinematic_coverage.json'
CONTROLLER = HOME/'dronecrowd_controller_coverage_v1/analysis.json'
PRODUCER = HOME/'moving_source_forecast_supported_v1'
CUTS = HOME/'moving_role_separated_cost_v1/registration_v3.json'


def role_compatibility(registry, groups, proposal):
    reserved = {k.removeprefix('dronecrowd_'): r for k, r in registry['records'].items()
                if r['dataset'] == 'DroneCrowd'}
    if (set(reserved) != set(groups) or any(r['exclusion_group'] != groups[k] or
            r['reserved_role'] != 'confirmation' or r['predictive_admission']
            for k, r in reserved.items())):
        raise ValueError('Existing confirmation reservation or group identity changed')
    if any(proposal[k] for k in ('scientific_roles_assigned', 'outcome_readout_allowed',
                                 'predictive_admission', 'outcome_or_support_counts_used_for_partition')):
        raise ValueError('Only unassigned label-blind alternatives are covered')
    alternatives = role_sets(proposal['alternatives'], groups)
    return {name: dict(calibration_recordings=len(roles['calibration']),
                       confirmation_recordings=len(roles['confirmation']),
                       conflicts_with_existing_reservation=bool(roles['calibration']))
            for name, roles in alternatives.items()}


def reconstruct_cuts():
    import scripts.run_m3w_moving_source_forecast as training

    config = run.read(training.CONFIG)
    d, receipt, _ = prepare(ROOT, config['admission'])
    population = PRODUCER/'population.json'
    if run.read(population) != run.stable(receipt):
        raise ValueError('Raw-to-array reconstruction differs from the original producer population')
    cut_jobs = [x['job'] for x in run.read(CUTS)['jobs']]
    original = run.read(run.old.PARENTS[4])['cuts']
    producer_jobs = run.read(PRODUCER/'job_roster.json')['jobs']
    out = {}
    for f in (0, 1):
        registered = [r for r in cut_jobs if r['producer_fold'] == f]
        sites, = {tuple(r['producer_sites']) for r in registered}
        design = fit_design(d, sites, d['baseline_ade'])
        ids = design['train_ids']
        indexed = array_hash(ids)
        same = [r['identity'] for r in producer_jobs if r['identity']['fold'] == f]
        if not same or any(r['train_ids_sha256'] != indexed or r['fit_groups'] != list(sites)
                           or r['population_sha256'] != run.sha(population) for r in same):
            raise ValueError('Cut training rows do not match the actual producer fit')
        expected = original[str(f)]
        value = dict(easy=design['easy_cut'], hard=design['hard_cut'])
        exact = value == expected and all(dict(easy=r['easy_cut'], hard=r['hard_cut']) == expected
                                          for r in registered)
        out[str(f)] = dict(fit_groups=len(sites), fit_rows=len(ids), train_rows_sha256=indexed,
            raw_to_array_population_exact=True, recorded=expected, reconstructed=value,
            exact_original_cut=exact)
    native = [r['identity'] for r in run.read(run.parent.REG)['forecasts']]
    declared, = {tuple(r['fit_groups']) for r in native}
    sites, = {tuple(sorted(r['normalizers'])) for r in native}
    if set(sites) != set(declared) & set(d['sites']):
        raise ValueError('Contributing sources disagree with producer fit normalizers')
    design = fit_design(d, sites, d['baseline_ade'])
    indexed = array_hash(design['train_ids'])
    if len(native) != 6 or any(r['fold'] != 2 or r['train_ids_sha256'] != indexed
                              or r['population_sha256'] != run.sha(population) for r in native):
        raise ValueError('Transport cut must use only the exact comparator fit')
    return out, dict(fit_groups=len(sites), declared_fit_groups=len(declared),
        declared_without_fit_rows=sorted(set(declared)-set(sites)),
        fit_rows=len(design['train_ids']), train_rows_sha256=indexed,
        easy=design['easy_cut'], hard=design['hard_cut'], candidate_only=True,
        criterion='positive_observed_CV_ADE_q25_and_observed_CV_ADE_q75')


def verify_bank():
    self_refs, native_refs = run.register()
    trials = {t['key']: t for t in run.read(FORECASTERS)['trials']}
    screened = {t['key']: t for t in run.read(SCREEN)['trials']}
    seeds = tuple(run.kin.prior.SEEDS)
    checkpoints = []
    for r in native_refs:
        ident = r['identity']
        name = f'{ident["family"]}_fold{ident["fold"]}_seed{ident["seed"]}'
        prior = trials[name]
        if (ident['fold'] != 2 or ident['baseline_index'] != 4
                or prior['prediction_sha256'] != r['sha256']
                or prior['checkpoint']['sha256'] != r['checkpoint_sha256']
                or screened[name]['still_uncovered']):
            raise ValueError('Unverified or mismatched exact-bank producer')
        run.checked(r)
        run.checked(prior['checkpoint'])
        checkpoints.append(dict(family=ident['family'], seed=ident['seed'],
                                checkpoint_sha256=r['checkpoint_sha256']))
    coverage = {t['seed']: t for t in run.read(SELF)['checkpoint_receipts']}
    for r in self_refs:
        if r['arm'] != 'self':
            continue
        if coverage[r['seed']]['fitted'] != r['checkpoint']['sha256']:
            raise ValueError('Self producer not bound to the source screen')
        run.checked(r['checkpoint'])
        run.checked(r['forecast'])
        checkpoints.append(dict(family='self', seed=r['seed'], checkpoint_sha256=r['checkpoint']['sha256']))
    if sorted((x['family'], x['seed']) for x in checkpoints) != sorted(
            (family, seed) for family in ('pct', 'eqmotion', 'self') for seed in seeds):
        raise ValueError('The original bank requires three exact producers per seed')
    return checkpoints


def audit():
    registry, grouping, proposal = map(run.read, (REGISTRY, GROUPS, PROPOSAL))
    for name, digest in registry['bindings'].items():
        if run.sha(ROOT/name) != digest:
            raise ValueError('Changed binding in the whole-source registry')
    if proposal['exclusion_constraints_sha256'] != run.sha(GROUPS):
        raise ValueError('Unassigned proposal uses a different group definition')
    groups = grouping['recording_group']
    roles = role_compatibility(registry, groups, proposal)
    coverage = run.read(CONTROLLER)
    if (len(groups), len(set(groups.values())), coverage['possible_fit_recording_union']) != (112, 45, 34):
        raise ValueError('Archived external group or development fit roster changed')
    print(dict(state='role_reconciled', alternatives=roles), flush=True)
    bank = verify_bank()
    print(dict(state='exact_bank_verified', neural_checkpoints=len(bank)), flush=True)
    cut, candidate = reconstruct_cuts()
    print(dict(state='cuts_reconstructed', cuts=cut, candidate=candidate), flush=True)
    sources = (REGISTRY, GROUPS, PROPOSAL, FORECASTERS, SCREEN, SELF, CONTROLLER, CUTS,
               PRODUCER/'population.json', PRODUCER/'job_roster.json', run.old.PARENTS[4],
               run.parent.REG)
    return dict(bindings={str(p.relative_to(ROOT)): run.sha(p) for p in sources},
        candidate_bank=list(run.NAMES), neural_checkpoints=bank, reference_cv=run.parent.CV,
        fitting_selected_comparator=run.parent.OLS, training_cut_reconstruction=cut,
        exact_comparator_fit_cut_candidate=candidate,
        role_compatibility=roles, reserved_confirmation_recordings=len(groups), exclusion_groups=len(set(groups.values())),
        controller_possible_fit_recordings=coverage['possible_fit_recording_union'],
        external_calibration_role_changed=False, predictive_admission=False, external_controller_frozen=False,
        external_cut_selected=False, external_inputs_opened=False, external_outcomes_opened=False,
        independent_physical_sites_verified=False, independent_confirmation=False,
        original_rules_unchanged=True, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('audit', 'replay'))
    phase = p.parse_args().phase
    start = time.monotonic()
    out = PUBLIC/'analysis.json'
    if phase == 'replay' and not out.exists():
        raise ValueError('Run the original audit first')
    run.once(out, run.stable(audit()))
    info = dict(phase=phase, seconds=time.monotonic()-start, analysis_sha256=run.sha(out))
    dest = PUBLIC/f'{phase}_execution.json'
    if dest.exists():
        if run.read(dest)['analysis_sha256'] != info['analysis_sha256']:
            raise ValueError('Prior audit bound another artifact')
    else:
        run.once(dest, info)
    print(info, flush=True)


if __name__ == '__main__':
    main()
