"""Join the latest fitted cost-head training roster to the existing image screen."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_directional_utility as study

run, np = study.run, study.np
HOME = ROOT/'outputs/publication_readiness_2026_09'
SCREEN = HOME/'dronecrowd_cross_archive_v1'
PUBLIC = HOME/'dronecrowd_controller_coverage_v1'


def main():
    reg, summary, proof = (SCREEN/p for p in
        ('registration.json', 'analysis.json', 'execution_verification.json'))
    roster, result, verified = (run.read(p) for p in (reg, summary, proof))
    if (result['registration_sha256'] != run.sha(reg) or verified['analysis_sha256'] != run.sha(summary)
            or not verified['all_48_receipts_and_aggregate_exact'] or result['supported_pairs']):
        raise ValueError('Require the complete verified cross-archive screen with no flagged pairs')
    registration = HOME/'moving_past_background_v1/registration.json'
    old = {int(r['recording_index']):r for r in run.read(registration)['roster']['records']}
    screened = {r['node'] for r in roster['exposed'] if 'A' in r['roles']}
    freeze = study.PUBLIC/'freeze.json'
    fit_receipt = study.PUBLIC/'fit_execution.json'
    if run.read(fit_receipt)['artifact']['sha256'] != run.sha(freeze):
        raise ValueError('Require the verified completed directional fit')
    parent_population = run.old.prior.PARENT/'population.json'
    files = run.read(parent_population)['files']
    d = {k:np.load(run.checked(files[k]), mmap_mode='r', allow_pickle=False)
         for k in ('sites', 'recordings', 'global_ids')}
    if set(old) != set(map(int,d['recordings'])) or len(old) != 34 or len(d['sites']) != 125540:
        raise ValueError('Require the original 34-recording controller population')
    nodes = {}
    for i,r in old.items():
        at = d['recordings'] == i
        if int(at.sum()) != r['rows'] or set(d['sites'][at]) != {r['source']}:
            raise ValueError('Recording index, fit source and exact row count must match')
        nodes.setdefault(r['source'], set()).add(r['node'])
    if screened != set().union(*nodes.values()) or len(nodes) != 14:
        raise ValueError('Image screen does not cover the controller training source roster')
    parent = study.size.PUBLIC/'freeze.json'
    old_folds = {(c['seed'], f['identity']['held']):f for c in run.read(parent)['cells']
                 for f in c['selection']}
    details = []
    for c in run.read(freeze)['cells']:
        for f in c['folds']:
            i = f['identity']
            previous = old_folds[c['seed'], i['held']]
            if i['fit'] != previous['identity']['fit'] or i['held'] in i['fit'] or len(i['fit']) != 7:
                raise ValueError('Cost and gain head fitting roles differ')
            for cached in (f, previous):
                with np.load(run.checked(cached['cache']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],d['global_ids'][d['sites']==i['held']])
            used = set().union(*(nodes[s] for s in i['fit']))
            if used - screened:
                raise ValueError('Some possible fitting recordings were not screened')
            details.append(dict(seed=c['seed'], held_source=i['held'], possible_fit_recordings=len(used)))
    roles = HOME/'dronecrowd_external_role_proposal_v1/alternatives.json'
    scope = run.read(roles)
    if scope['scientific_roles_assigned'] or scope['outcome_readout_allowed'] or scope['predictive_admission']:
        raise ValueError('This is only a preparation receipt')
    if len(details) != 42:
        raise ValueError('Require all source-seed fits')
    inputs = (reg, summary, proof, registration, parent_population, freeze, fit_receipt, parent, roles)
    doc = dict(result_source='fresh_fit_roster_join_cached_verified_image_screen',
        sources={str(p.relative_to(ROOT)):run.sha(p) for p in inputs},
        fit_cells=len(details), possible_fit_recording_union=len(screened),
        external_recordings=len(roster['external']), per_cell=details,
        scope='direct_cost_and_directional_gain_head_recording_coverage_only',
        physically_distinct_sites_proven=False, external_policy_frozen=False,
        full_candidate_bank_and_cut_lineage_admitted=False,
        scientific_roles_assigned=False, external_inference_run=False, external_outcomes_opened=False)
    run.once(PUBLIC/'analysis.json',run.stable(doc))
    print({k:doc[k] for k in ('fit_cells','possible_fit_recording_union','external_recordings',
                              'external_inference_run','external_outcomes_opened')})


if __name__ == '__main__':
    main()
