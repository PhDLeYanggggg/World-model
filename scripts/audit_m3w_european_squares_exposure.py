"""Bounded pre-intake source-exposure search; absence is not universal proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_european_squares import digest, save

OUT = ROOT / 'outputs/publication_readiness_2026_09/european_squares_exposure_v1'
BEFORE = '5155154d'
PATHS = ['README.md', 'README_RESULTS.md', 'research_state.json', 'configs',
         'src', 'scripts', 'outputs/publication_readiness_2026_09']


def main():
    groups_path = ROOT / 'outputs/publication_readiness_2026_09/european_squares_site_groups_v1/site_groups.json'
    groups = json.loads(groups_path.read_text())
    terms = ['EuropeanSquares', 'European Squares', 'European_Squares',
             's41597-026-06686', '18267205', 'kaktusracing',
             'pedestrian_trajectories', '01.Trajectories_raw']
    terms += list(groups['camera_reference_identities'].values())
    terms += [s.split(':', 1)[1] for s in groups['camera_reference_identities'].values()
              if s.startswith('youtube:')]
    terms += [Path(r['source_member']).name for r in groups['recordings']]
    terms = sorted(set(terms))
    commit = subprocess.check_output(['git', 'rev-parse', BEFORE], cwd=ROOT, text=True).strip()
    command = ['git', 'grep', '-n', '-I', '-i', '-F', '-f', '-', commit, '--', *PATHS]
    found = subprocess.run(command, cwd=ROOT, input='\n'.join(terms)+'\n',
                           text=True, capture_output=True, timeout=300)
    if found.returncode not in (0, 1):
        raise RuntimeError(found.stderr)
    inventory = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', *PATHS],
                                        cwd=ROOT, text=True).splitlines()
    private = ROOT / 'data/stage_cvpr2027_experiments'
    receipts = sorted(set(private.glob('*/identity.json')) | set(private.glob('*/run_identity.json')))
    receipts = [p for p in receipts if not p.parent.name.startswith('european_squares')]
    receipt_results = []
    for path in receipts:
        if path.is_symlink() or path.stat().st_size > 10_000_000:
            raise ValueError('Unsafe or oversized identity receipt')
        value = path.read_text()
        json.loads(value)
        matches = [term for term in terms if term.casefold() in value.casefold()]
        receipt_results.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path), matches=matches))
    result = dict(result_source='fresh_run', pre_intake_commit=commit,
        grouping_sha256=digest(groups_path), script_sha256=digest(Path(__file__)),
        terms=terms, git_search_paths=PATHS, git_tracked_paths=len(inventory),
        git_inventory_sha256=hashlib.sha256(('\n'.join(inventory)+'\n').encode()).hexdigest(),
        git_text_search_returncode=found.returncode, git_text_matches=found.stdout.splitlines(),
        private_identity_receipts=receipt_results,
        private_receipt_scope='one-level experiment identity/run_identity JSON only; no model weights or raw predictions',
        no_match_means='no listed aliases in bounded text and producer-receipt scope; not proof of universal non-exposure',
        remote_assets=dict(status='not_run', reason='exact M3W remote directory unknown',
            coordination_task='simulation model', coordination_thread='019f18cf-90b0-7640-b113-5ca73a0bd2ba',
            response_date='2026-09-24', no_remote_home_or_simulation_scan=True),
        arbitrary_binary_or_untracked_source_exposure_excluded=False,
        source_pretraining_exposure_of_publisher_detector_unknown=True,
        predictive_outcomes_opened=False, roles_assigned=False,
        stage5c_executed=False, smc_enabled=False)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/'analysis.json', result)
    print(json.dumps(dict(git_matches=len(result['git_text_matches']),
        receipt_files=len(receipt_results), receipt_files_with_matches=sum(bool(r['matches']) for r in receipt_results),
        terms=len(terms), git_paths=len(inventory))))


if __name__ == '__main__':
    main()
