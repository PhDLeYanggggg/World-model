"""Check paired diagnostic identities and seal the completed, replayed readout."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_gradient_diagnostic import PUBLIC, PRIVATE, registration, digest, immutable
from scripts.report_m3w_easy_gradient_diagnostic import summarize


def check_pairs(groups):
    names = set(); sources = set(); comparisons = 0
    expected = set(itertools.product(('marginal', 'supervised'), ('initial', 'final'), range(4)))
    for group in groups:
        assert group['group'] not in names
        names.add(group['group'])
        assert group['parameter_updates'] == 0 and not group['held_outcomes_used']
        rows = {(r['arm'], r['point'], r['batch']): r for r in group['rows']}
        assert len(rows) == len(group['rows']) == 16 and set(rows) == expected
        roles = group['identity']['roles']
        role_sets = [set(roles[k]) for k in ('producer_sites', 'controller_sites', 'training_sites', 'held_sites')]
        assert list(map(len, role_sets)) == [4, 4, 2, 2]
        for first, second in itertools.combinations(role_sets, 2):
            assert not first & second
        assert len(set.union(*role_sets)) == 12
        sources.update(role_sets[2])
        for batch in range(4):
            a, b = rows['marginal', 'initial', batch], rows['supervised', 'initial', batch]
            assert a['losses'] == b['losses'] and a['gradients'] == b['gradients']
            assert len({rows[arm, point, batch]['sampled_rows'] for arm, point in expected_pairs()}) == 1
            comparisons += 1
    assert len(names) == 108 and len(sources) == 12
    return dict(groups=108, initial_pair_comparisons=comparisons, fitting_localities=12)


def expected_pairs():
    return itertools.product(('marginal', 'supervised'), ('initial', 'final'))


def main():
    reg = registration()
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    collected = json.loads((PRIVATE/'collected.json').read_text())
    checks = check_pairs(collected['groups'])
    first, replay = collected['receipt.json'], collected['replay_receipt.json']
    assert first['artifacts'] == replay['artifacts'] and replay['replay_exact']
    for label, doc in [('diagnose', first), ('replay', replay)]:
        account = collected[label+'_accounting']
        assert account['code'] == 0
        assert any(line.startswith(doc['job_id']+'|COMPLETED|0:0|') for line in account['stdout'].splitlines())
    before = {p.name: digest(p) for p in PUBLIC.iterdir() if p.suffix in ('.json', '.md', '.svg')}
    subprocess.run([sys.executable, str(ROOT/'scripts/report_m3w_easy_gradient_diagnostic.py')], cwd=ROOT, check=True)
    for name, sha in before.items():
        assert digest(PUBLIC/name) == sha
    computed = summarize(collected['groups'])
    summary = json.loads((PUBLIC/'summary.json').read_text())
    assert all(summary[k] == v for k, v in computed.items())
    paths = {**reg['code_bindings'], **reg['control_bindings']}
    for relative in (
        'scripts/report_m3w_easy_gradient_diagnostic.py',
        'scripts/verify_m3w_easy_gradient_diagnostic.py',
        'tests/test_m3w_easy_gradient_report.py',
    ):
        paths[relative] = digest(ROOT/relative)
    artifacts = {p.name: digest(p) for p in sorted(PUBLIC.iterdir())
                 if p.suffix in ('.json', '.md', '.svg') and p.name not in ('verification.json', 'verification_report.md')}
    output = dict(status='verified_fitting_diagnostic_not_efficacy', **checks,
        scalar_geometry_checks=computed['scalar_geometry_checks'],
        report_regeneration_byte_exact=True, all_group_replay_exact=True,
        first_job=first['job_id'], replay_job=replay['job_id'],
        source_bindings=paths, artifacts=artifacts,
        private_collected_sha256=digest(PRIVATE/'collected.json'),
        parameter_updates=0, held_outcomes_used=False, independent_roles_read=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False,
        full_legacy_suite='not_run', raw_data_rebuild='not_run')
    immutable(PUBLIC/'verification.json', output)
    print(json.dumps(dict(**checks, source_files=len(paths), artifacts=len(artifacts),
                         seal_sha256=digest(PUBLIC/'verification.json'))))


if __name__ == '__main__':
    main()
