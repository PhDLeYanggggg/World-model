"""Bind the three self-kinematic checkpoints to the archived source screen."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_record_joint_actions as run
from scripts.run_m3w_dronecrowd_blind_preflight import HOME, DATA, PUBLIC, CONSTRAINT, PROPOSAL, read, digest, once

BROAD = HOME/'movingdronecrowd_background_v2'


def main():
    kin = run.kin
    path = kin.PUBLIC/'training_freeze.json'
    fit = read(path)
    jobs = {j['seed']: j for j in read(kin.prior.REG)['jobs']}
    pop_path = kin.prior.PUBLIC/'fit_population.json'
    population = read(pop_path)
    records = set(population['recordings'])
    br, ba, proof = (BROAD/n for n in ('registration.json', 'analysis.json', 'verification.json'))
    reg, screen, proof_data = map(read, (br, ba, proof))
    if (screen['registration_sha256'] != digest(br) or proof_data['analysis_sha256'] != digest(ba)
            or not proof_data['all_pair_receipts_and_graph_exact'] or screen['moving_nodes_linked_to_protected']
            or screen['insufficient_background_moving_nodes']):
        raise ValueError('Require complete replay-verified background image screen')
    screened = {r['node'] for r in reg['view_selection']}
    if not records <= screened or len(records) != 24:
        raise ValueError('Self-kinematic training roster not fully screened')
    groups = read(CONSTRAINT)['recording_group']
    proposal = read(PROPOSAL)
    if (proposal['scientific_roles_assigned'] or proposal['outcome_readout_allowed']
            or proposal['predictive_admission'] or proposal['exclusion_constraints_sha256'] != digest(CONSTRAINT)):
        raise ValueError('Unassigned external roster required')
    parent = HOME/'movingdronecrowd_background_v1/registration.json'
    if read(parent)['protected_constraints_sha256'] != digest(CONSTRAINT):
        raise ValueError('Broad source screen used a different external grouping')
    if not fit['before_A_inference_and_labels'] or set(jobs) != {17, 29, 43}:
        raise ValueError('Require three frozen original-seed fits')
    checkpoints = []
    for f in fit['fits']:
        if f['arm'] != 'self':
            continue
        seed = f['seed']
        warm = jobs[seed]
        if (f['identity']['registration_sha256'] != digest(kin.REG)
                or f['identity']['global_fit_rows_sha256'] != population['global_fit_ids_sha256']
                or warm['identity']['train_ids_sha256'] != population['global_fit_ids_sha256']
                or set(warm['lineage']['recordings']) != records or warm['lineage']['rows'] != population['rows']
                or f['step'] != kin.SETTINGS['steps'] or not f['complete'] or f['held_rows_sampled']):
            raise ValueError('Checkpoint lineage differs from the screened roster')
        run.checked(f['checkpoint'])
        run.checked(warm['checkpoint'])
        checkpoints.append(dict(seed=seed, fitted=f['checkpoint']['sha256'], warm=warm['checkpoint']['sha256']))
    if {f['seed'] for f in checkpoints} != set(jobs) or len(checkpoints) != 3:
        raise ValueError('Missing matched self checkpoints')
    h = hashlib.sha256()
    for s in sorted(records):
        for clip in sorted(groups):
            name = f'{s}___drone_{clip}.json'
            packet = DATA/'movingdronecrowd_background_v2/pairs'/name
            v = read(packet)
            if (v['registration_sha256'] != digest(br)
                    or (v['left'], v['right'], v['kind'], v['status']) != (s, 'drone_'+clip, 'cross_source', 'unsupported')
                    or v['support']):
                raise ValueError('Pair lineage changed or a cross-archive candidate was flagged')
            h.update(bytes.fromhex(digest(packet)))
    inputs = (kin.REG, kin.prior.REG, pop_path, path, br, ba, proof, parent, CONSTRAINT, PROPOSAL)
    doc = dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        checkpoint_receipts=checkpoints, fitting_recordings=len(records), fitting_rows=population['rows'],
        external_recordings=len(groups), external_exclusion_groups=len(set(groups.values())),
        verified_cached_image_pairs=len(records)*len(groups), image_pair_chain_sha256=h.hexdigest(),
        physical_independence_certified=False, scientific_roles_assigned=False,
        predictive_admission=False, forecast_or_outcome_readout=False,
        controller_and_cut_transfer_frozen=False, submission_ready=False)
    once(PUBLIC/'kinematic_coverage.json', doc)
    print(dict(checkpoints=len(checkpoints), cached_image_pairs=doc['verified_cached_image_pairs'],
        analysis_sha256=digest(PUBLIC/'kinematic_coverage.json')), flush=True)


if __name__ == '__main__':
    main()
