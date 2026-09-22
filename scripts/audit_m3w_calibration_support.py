"""Audit current calibration reuse without loading targets or assigning roles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_calibration_support import (
    audit_view, best_case_cluster_support, require_calibration_exclusion, required_producers,
)

LATEST = 'outputs/publication_readiness_2026_09/conditional_cost_v1/analysis.json'
MANIFEST = 'data/stage_cvpr2027_experiments/eqmotion_nested_v1/cost_views.json'
CODE = ('scripts/audit_m3w_calibration_support.py', 'src/evaluation/m3w_calibration_support.py',
        'tests/test_m3w_calibration_support.py', 'src/world_model/m3w_joint_intervention.py',
        'src/world_model/m3w_native_nested.py')
OUTPUT = 'outputs/publication_readiness_2026_09/calibration_support_v1/analysis.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run():
    report = json.loads((ROOT/LATEST).read_text())
    bindings = dict(report['identity']['source_bindings'])
    for path, expected in bindings.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Changed source dependency: '+path)
    for path in (LATEST, *CODE):
        bindings[path] = digest(ROOT/path)
    for name in ('replay.json', 'independent_verification.json'):
        path = str(Path(LATEST).parent/name)
        receipt = json.loads((ROOT/path).read_text())
        if not receipt['all_checks_passed'] or receipt['analysis_sha256'] != bindings[LATEST]:
            raise ValueError('Unverified parent readout')
        bindings[path] = digest(ROOT/path)
    if MANIFEST not in bindings:
        raise ValueError('Current analysis must bind its producer manifest')
    manifest = json.loads((ROOT/MANIFEST).read_text())
    cfg = report['identity']['config']
    sites, seeds = cfg['sites'], cfg['seeds']
    if len(manifest['views']) != len(sites)*len(seeds):
        raise ValueError('Incomplete source matrix')
    records, available, checked = [], {}, set()
    for view in manifest['views']:
        key = f'{view["outer_site"]}_seed{view["seed"]}'
        if key in checked:
            raise ValueError('Duplicate view')
        checked.add(key)
        path = f'{cfg["output"]}/trials/{key}/complete.json'
        receipt = json.loads((ROOT/path).read_text()); head = receipt['identity']
        training = next(r for r in report['training'] if r['view'] == key)
        if (head['identity'] != report['identity'] or not receipt['fit']['complete']
                or receipt['checkpoint_sha256'] != training['checkpoint_sha256']
                or digest(ROOT/receipt['checkpoint']) != receipt['checkpoint_sha256']
                or head['producers'] != [g['producer'] for g in view['groups']]):
            raise ValueError('Current head does not match checked producer graph')
        bindings[path] = digest(ROOT/path)
        bindings[receipt['checkpoint']] = receipt['checkpoint_sha256']
        records.extend(audit_view(view, head['training_sites'], sites))
        for g in view['groups']:
            producer = g['producer']
            old = available.setdefault(producer['id'], producer)
            if old != producer:
                raise ValueError('Producer identity collision')
    requirements = required_producers(sites, seeds, list(available.values()))
    # Check the hypothetical corrected graph through the same refusal guard;
    # this validates feasibility only, without pretending those fits exist.
    hypothetical_passes = 0
    for r in records:
        template = next(v for v in available.values() if v['seed'] == r['seed'])
        def proposed(excluded):
            fit = sorted(set(sites)-set(excluded))
            return dict(template, id='not_run_'+ '_'.join(sorted(excluded))+str(r['seed']),
                excluded_sites=sorted(excluded), training_sites=fit, preprocessing_fit_sites=fit)
        require_calibration_exclusion(calibration_sites=[r['proposed_calibration_site']],
            head_fit_sites=r['remaining_head_fit_sites'], preprocessing_fit_sites=r['remaining_head_fit_sites'],
            scoring_producer=available[r['replacement_scoring_producer']],
            target_producers=[proposed(g['excluded_sites']) for g in r['required_target_exclusions']])
        hypothetical_passes += 1
    summary = dict(current_head_views=len(checked), proposed_inner_views=len(records),
        existing_head_reuse_rejected=sum(not r['attempts']['reuse_current_head']['accepted'] for r in records),
        row_only_refit_rejected=sum(not r['attempts']['drop_site_rows_and_refit_head_only']['accepted'] for r in records),
        scoring_replacement_still_rejected=sum(not r['attempts']['drop_rows_and_replace_scoring_predictor']['accepted'] for r in records),
        contaminated_cost_target_incidences=sum(len(r['exposed_cost_target_producers']) for r in records),
        unique_existing_pair_producers=len(available),
        missing_triple_excluded_producers=sum(not r['available'] for r in requirements),
        hypothetical_fixed_views_pass_fitting_exclusion=hypothetical_passes,
        calibration_sites_per_hypothetical_view=1)
    result = dict(result_source='fresh_run_metadata_lineage_audit_cached_verified_parents',
        source_bindings=bindings, summary=summary, proposed_views=records,
        triple_excluded_requirements=requirements, synthetic_bound_illustration=best_case_cluster_support(),
        causal_or_future_arrays_deserialized=False, new_training=False, roles_assigned=False,
        original_closed_roles_opened=False, risk_tolerance_changed=False,
        calibration_executed=False, independent_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)
    for path, expected in bindings.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Dependency changed during audit: '+path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = run(); path = ROOT/OUTPUT
    if args.verify:
        if not path.exists() or json.loads(path.read_text()) != result:
            raise ValueError('Recorded audit differs or is missing')
    elif path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Refusing to overwrite a changed audit')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(summary=result['summary'], verified=args.verify,
                         analysis_sha256=digest(path)), indent=2))


if __name__ == '__main__':
    main()
