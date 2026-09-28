"""Freeze completed paired training without making a held-efficacy claim."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_risk_priority import PUBLIC, PRIVATE, registration, digest, immutable


def remote_config_digest(path):
    value = json.loads(Path(path).read_text())
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    return hashlib.sha256(encoded.encode()).hexdigest()


def training_summary(receipt):
    rows = receipt['fitting_summary']
    assert receipt['groups'] == 108 and receipt['heads'] == len(rows) == 216
    assert receipt['updates_per_head'] == 2000 and receipt['control_parent_states_exact'] == 108
    index = {(r['group'], r['arm']): r for r in rows}
    assert len(index) == 216
    names = sorted({r['group'] for r in rows})
    assert len(names) == 108
    for name in names:
        a, b = index[name, 'uncapped'], index[name, 'risk_priority']
        for key in ('sample_hash', 'query_draws', 'row_draws', 'first_monitor'):
            assert a[key] == b[key], (name, key)
        assert a['step'] == b['step'] == 2000
        assert a['unknown_rows_sampled'] == b['unknown_rows_sampled'] == 0
        assert b['min_risk_projection'] is None or b['min_risk_projection'] >= .5-1e-6
        assert 0 <= b['cap_updates'] <= 2000 and 0 <= b['mean_alpha'] <= 1
        for r in (a, b):
            assert all(np.isfinite(v) for v in r['last_monitor'].values())
    loss = {}
    for arm in ('uncapped', 'risk_priority'):
        loss[arm] = {term: dict(
            first_mean=float(np.mean([index[n, arm]['first_monitor'][term] for n in names])),
            final_mean=float(np.mean([index[n, arm]['last_monitor'][term] for n in names])),
            improved_heads=sum(index[n, arm]['last_monitor'][term] < index[n, arm]['first_monitor'][term] for n in names))
            for term in ('marginal', 'occurrence', 'conditional', 'supervised')}
    repaired = [index[n, 'risk_priority'] for n in names]
    return dict(groups=108, heads=216, total_updates=432000,
        loss=loss, cap_active_updates=sum(r['cap_updates'] for r in repaired),
        repaired_updates=216000, mean_alpha_over_updates=float(np.mean([r['mean_alpha'] for r in repaired])),
        minimum_pre_optimizer_risk_projection=min(r['min_risk_projection'] for r in repaired if r['min_risk_projection'] is not None),
        repaired_lower_final_direct_risk_heads=sum(index[n, 'risk_priority']['last_monitor']['marginal'] < index[n, 'uncapped']['last_monitor']['marginal'] for n in names))


def main():
    reg = registration()
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    evidence = json.loads((PRIVATE/'collected.json').read_text())
    first, replay = evidence['training_complete.json'], evidence['replay.json']
    for phase, doc in [('train', first), ('replay', replay)]:
        assert doc['registration_sha256'] == digest(PUBLIC/'registration.json')
        assert doc['config_sha256'] == remote_config_digest(ROOT/'configs/m3w_european_easy_risk_priority_v1.json')
        assert not doc['held_outcomes_used'] and not doc['independent_roles_read']
        accounting = evidence[phase+'_accounting']
        assert accounting['returncode'] == 0
        assert any(line.startswith(doc['job_id']+'|COMPLETED|0:0|') for line in accounting['stdout'].splitlines())
    assert replay['groups'] == 1 and replay['repair_first_pair_replay_exact']
    result = training_summary(first)
    result.update(result_source='fresh_run_Torch_training', inputs='cached_verified_fitting_only_packets',
        registration_sha256=digest(PUBLIC/'registration.json'), registration_commit='f22e98c8',
        train_job=first['job_id'], replay_job=replay['job_id'],
        train_seconds=first['seconds'], train_peak_RSS_KiB=first['peak_RSS_KiB'],
        replay_seconds=replay['seconds'], all108_uncapped_parent_states_exact=True,
        repair_first_pair_replay_exact=True, all108_repaired_heads_retrained_for_replay=False,
        fitting_loss_is_not_efficacy=True, held_readout='not_run',
        independent_confirmation='not_run', deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    immutable(PUBLIC/'training_summary.json', result)
    immutable(PUBLIC/'create_training_freeze.json', dict(
        registration_sha256=digest(PUBLIC/'registration.json'), receipt=first, replay_receipt=replay,
        collection_sha256=digest(PRIVATE/'collected.json'),
        collector_sha256=digest(ROOT/'scripts/manage_m3w_easy_risk_priority.py'),
        reporting_sha256=digest(Path(__file__))))
    text = ['# Paired Risk-Priority Training', '',
        'Source: fresh_run Torch training on cached_verified fitting-only CREATE packets. '
        'No new held or independent-source outcome has been read. Training completion is not efficacy.', '',
        f"All108 groups /216 heads /432,000 cumulative updates completed in job {first['job_id']} "
        f"({first['seconds']:.2f} seconds). All108 uncapped states reproduce the original "
        'supervised states except identity/elapsed time. Initial states and query sample chains match between arms. '
        f"The first full pair replays exactly in job {replay['job_id']}; this is not a replay of all108 repaired fits.", '',
        '| Arm | Fitting objective component | Initial mean | Final mean | Improved heads |',
        '|---|---|---:|---:|---:|']
    for arm, losses in result['loss'].items():
        for term, row in losses.items():
            text.append(f"| {arm} | {term} | {row['first_mean']:.8g} | {row['final_mean']:.8g} | {row['improved_heads']}/108 |")
    text += ['', f"The cap was active in {result['cap_active_updates']:,}/216,000 repaired updates; "
        f"mean coefficient {result['mean_alpha_over_updates']:.6g}. Minimum pre-optimizer direct-risk projection: "
        f"{result['minimum_pre_optimizer_risk_projection']:.6g}. Lower final direct-risk monitor loss than uncapped: "
        f"{result['repaired_lower_final_direct_risk_heads']}/108 heads. This is a fitting monitor, not generalization.", '',
        'Next: freeze causal actions, then the preregistered same-query-count development comparison, including '
        'selected-risk/easy/zero-CV failures and locality intervals. Do not promote from fitting loss, cap activation '
        'or source-role replay alone. Independent selection/calibration/confirmation stay closed. '
        'Obs8/pred12 raw-frame image-local detector silver only, no seconds/metric/human-gold/physical-safety/true3D/'
        'foundation/submission-ready claim. Stage5C and SMC remain disabled.']
    path = PUBLIC/'training_report.md'; content = '\n'.join(text)+'\n'
    if path.exists():
        assert path.read_text() == content
    else:
        path.write_text(content)
    print(json.dumps({k: v for k, v in result.items() if k != 'loss'}))


if __name__ == '__main__':
    main()
