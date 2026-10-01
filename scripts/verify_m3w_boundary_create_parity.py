"""Verify the collected CREATE diagnostic against the complete local summary."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import manage_m3w_boundary_diagnostic as parent


def main():
    public, private = parent.PUBLIC, parent.PRIVATE
    collected = json.loads((private/'collected.json').read_text())
    receipt = json.loads((public/'compute_receipt.json').read_text())
    assert receipt == collected['receipt']
    assert collected['accounting'] == dict(code=0, stdout='COMPLETED|0:0|00:01:47\n')
    assert receipt['job_id'] == '37602475' and receipt['groups'] == 288
    assert receipt['full_replay_exact'] and len(receipt['artifacts']) == 288
    assert receipt['accounting_checks'] == 64512
    assert not any(receipt[k] for k in ('parameter_updates', 'independent_roles_read', 'deployment_changed'))
    for name, field in [('registration.json', 'registration_sha256'),
                        ('packet_manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256')]:
        assert parent.digest(public/name) == receipt[field]
    remote = json.loads((public/'summary.json').read_text())
    local = json.loads((public/'local_summary.json').read_text())
    assert remote == collected['summary'] and local.keys() == remote.keys()
    assert local['result_source'] == 'fresh_run_local_verified_packet_reconstruction'
    assert remote['result_source'] == 'fresh_run_frozen_action_diagnostic_on_CREATE'
    compared = [key for key in local if key != 'result_source']
    for key in compared:
        parent.compare_tree(local[key], remote[key])
    local_receipt = json.loads((public/'local_compute_receipt.json').read_text())
    assert local_receipt['all_packet_bytes_match_committed_manifest']
    assert local_receipt['full_local_replay_exact']
    assert parent.digest(public/'local_summary.json') == local_receipt['summary_sha256']
    assert collected['local_parent_risk_checks'] == 1728
    assert collected['local_packet_parity_checks'] == 3
    paths = [public/name for name in ('registration.json', 'packet_manifest.json', 'compute_receipt.json',
        'summary.json', 'local_summary.json', 'local_compute_receipt.json')]
    paths += [Path(__file__).resolve(), ROOT/'scripts/manage_m3w_boundary_diagnostic.py']
    out = dict(status='verified_CREATE_replication_no_new_training', job_id=receipt['job_id'],
        scheduler_state='COMPLETED', exit_code='0:0', scheduler_seconds=107,
        groups=288, fitting_groups=72, transfer_groups=216,
        remote_first_pass_seconds=receipt['first_pass_seconds'], remote_replay_seconds=receipt['replay_seconds'],
        remote_full_replay_exact=True, local_full_replay_exact=True,
        complete_summary_fields_compared=compared, rtol=1e-10, atol=1e-9,
        only_excluded_field='result_source_exact_expected_execution_provenance',
        whole_summary_byte_identical=False, remote_native_accounting_checks=64512,
        collected_parent_risk_checks=1728, individual_local_remote_packet_comparisons=3,
        all288_local_packet_bytes_match_export_manifest=True,
        independent_confirmation=False, deployment_changed=False, parameter_updates=0,
        artifacts={str(p.relative_to(ROOT)): parent.digest(p) for p in paths})
    parent.immutable(public/'create_parity_verification.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
