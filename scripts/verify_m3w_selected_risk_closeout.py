"""Close the audit-only source update without repeating unchanged model replays."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selected_risk_learning as run


def main():
    path = run.PUBLIC/'verification.json'; original = json.loads(path.read_text())
    assert original['all_passed'] and original['tests'] == 431
    audit_script = 'scripts/audit_m3w_selected_query_budget.py'
    # The only source change adds independent mass checks to audit.main.
    for name, sha in original['source_bindings'].items():
        if name != audit_script: assert run.digest(ROOT/name) == sha
    for name, sha in original['artifacts'].items(): assert run.digest(run.PUBLIC/name) == sha
    q = json.loads((run.PUBLIC/'query_budget_verification.json').read_text())
    assert q['all_passed'] and q['independent_locality_event_mass_checks'] == 576
    assert q['source_sha256'] == run.digest(ROOT/audit_script)
    assert q['audit'] == run.artifact(run.PUBLIC/'query_budget_audit.json')
    xml = run.PRIVATE/'closeout_tests.xml'
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_m3w_selected_risk_reporting.py',
        '-q', '--junitxml='+str(xml)], cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'closeout_tests.log').write_text(result.stdout+'\n'+result.stderr)
    print(result.stdout, flush=True); assert result.returncode == 0
    suites = list(ET.parse(xml).getroot())
    assert sum(int(s.attrib['tests']) for s in suites) == 4
    assert not any(int(s.attrib.get('failures', 0))+int(s.attrib.get('errors', 0)) for s in suites)
    bindings = dict(original['source_bindings']); bindings[audit_script] = run.digest(ROOT/audit_script)
    bindings[str(Path(__file__).relative_to(ROOT))] = run.digest(Path(__file__))
    artifacts = {str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*')
        if p.is_file() and p.name != 'final_verification.json'}
    assert all((run.PUBLIC/n).stat().st_size < 1024**2 for n in artifacts)
    run.immutable_json(run.PUBLIC/'final_verification.json', dict(all_passed=True,
        parent_replay=run.artifact(path), artifacts=artifacts, source_bindings=bindings,
        replay_status='cached_verified_unchanged_models_and_decisions', checkpoint_replays=72,
        scalar_and_vector_decisions=432, scoped_tests=431, fresh_post_update_tests=4,
        tests_are_not_additive=True, post_update_mass_checks=576,
        source_update_scope='audit.main independently cross-checks existing outcome mass; account() unchanged',
        previous_audit_source_sha256=original['source_bindings'][audit_script],
        current_audit_source_sha256=bindings[audit_script], deployment_changed=False))
    print(json.dumps(dict(all_passed=True, rebound_audit_only=True, models_retrained=False)))


if __name__ == '__main__': main()
