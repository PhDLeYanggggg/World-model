"""Render the completed, replay-verified DroneCrowd calibration readout."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.world_model.m3w_dronecrowd_calibration import GATES, MARGINS, winner

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/dronecrowd_calibration_readout_v1'
MANIFEST = ROOT/'configs/m3w_dronecrowd_calibration_confirmation_v2.json'
BASE, OLS = 'constant_velocity_causal_fd', 'history_ols4_velocity'
CONTROLS = ('independent', 'scene', 'pair_additive', 'joint')
ARMS = (BASE, OLS, *CONTROLS, *GATES)
RISK = ('switches', 'switch_rate', 'supported_views', 'undefined_easy_views', 'all_risk_over_2_views',
        'signed_easy_over_2_views', 'worst_signed_easy_percent', 'worst_recording_p95_known_ade_pixel')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(root, receipt, expected):
    path = root/receipt['artifact']['path']
    if (path != expected or path.stat().st_size != receipt['artifact']['bytes']
            or digest(path) != receipt['artifact']['sha256']):
        raise ValueError('Execution receipt does not bind the expected artifact')
    return receipt['artifact']['sha256']


def verify_artifacts(root, public, manifest_path):
    paths = {p: public/f'{p}_execution.json' for p in ('infer', 'score', 'replay')}
    if not all(p.is_file() for p in paths.values()):
        raise ValueError('Full calibration inference, scoring and replay must finish first')
    freeze, analysis = (public/k for k in ('forecast_freeze.json', 'analysis.json'))
    infer = verify_receipt(root, read(paths['infer']), freeze)
    score = verify_receipt(root, read(paths['score']), analysis)
    verify_receipt(root, read(paths['replay']), analysis)
    registration = public/'registration.json'
    reg = digest(registration)
    registered = read(registration)
    bound = [p for p in registered['parents'] if root/p['path'] == manifest_path]
    if len(bound) != 1:
        raise ValueError('Registration does not bind the approved role file')
    verify_receipt(root, {'artifact': bound[0]}, manifest_path)
    manifest = read(manifest_path)
    f = read(freeze)
    clips = f['clips']
    role = manifest['assignments']['calibration']['recordings']
    denied = set(manifest['assignments']['confirmation']['recordings'])
    counts = manifest['input_only_counts']['calibration']
    if (len(role) != len(set(role)) or denied.intersection(role) or f['registration'] != reg
            or f['future_labels_read'] is not False or [c['sequence'] for c in clips] != role
            or any(c['registration'] != reg for c in clips)
            or (len(clips), sum(c['queries'] for c in clips), sum(c['rows'] for c in clips))
                != (counts['recordings'], counts['queries'], counts['eligible'])):
        raise ValueError('Frozen roster differs from the approved calibration role')
    return read(analysis), dict(calibration_registration_sha256=reg, score_replay_sha256=score,
                                frozen_forecasts_sha256=infer, calibration_counts=counts)


def summarize(doc):
    if (doc['confirmation_outcome_access'] or doc['independent_confirmation'] or not doc['original_rules_unchanged']
            or not doc['calibration_outcomes_opened'] or doc['submission_ready'] or doc['deployment_changed']):
        raise ValueError('Expected calibration-only evidence with the original rules')
    s = doc['summary']
    if ((s['recordings'], s['source_groups'], s['views']) != (40, 15, 120)
            or s['seeds'] != [17, 29, 43] or s['bootstrap_draws'] != 3000 or set(s['risk']) != set(ARMS)):
        raise ValueError('Unexpected calibration summary roster')
    rows = []
    for arm in ARMS:
        risk = s['risk'][arm]
        if (risk['required_views'] != 120 or risk['scored_views'] != 120
                or not 0 <= risk['supported_views'] <= 120
                or risk['supported_views']+risk['undefined_easy_views'] > 120):
            raise ValueError(f'Incomplete or invalid recording risk support for {arm}')
        margin = MARGINS[GATES.index(arm)] if arm in GATES else None
        cv, ols = (s['effects'].get(f'{arm}_vs_{ref}') for ref in ('cv', 'ols4'))
        lower = cv['completion_lower_pp']['interval95'] if cv else None
        fail = []
        if arm in GATES:
            if risk['supported_views'] != 120:
                fail.append('recording_risk_or_support')
            if not risk['switches']:
                fail.append('zero_switches')
            if lower is None or lower[0] <= 0.:
                fail.append('cv_lower_ci_not_positive')
        rows.append(dict(arm=arm, margin=margin, **{k: risk[k] for k in RISK}, vs_cv=cv, vs_ols4=ols,
                         eligible=arm in GATES and not fail, disqualification=fail))
    if len({s['risk'][k]['switches'] for k in CONTROLS}) != 1:
        raise ValueError('Registered matched controls do not have equal intervention totals')
    chosen = winner(s)
    eligible = [r for r in rows if r['eligible']]
    independent = min(eligible, key=lambda r: (
        -r['vs_cv']['completion_lower_pp']['point'], r['switches'], GATES.index(r['arm'])), default=None)
    margin = MARGINS[GATES.index(chosen)] if chosen else None
    if (chosen != (independent['arm'] if independent else None)
            or doc['selected_arm'] != chosen or doc['calibration_pass'] is not (chosen is not None)
            or doc['selected_normalized_gain_margin'] != margin):
        raise ValueError('Reported outcome disagrees with the locked winner rule')
    return dict(selected_arm=chosen, selected_margin=margin, eligible_arms=[r['arm'] for r in eligible],
                rows=rows, recordings=40, groups=15, views=120, seeds=s['seeds'],
                confirmation_outcome_access=False, independent_confirmation=False, deployment_changed=False,
                submission_ready=False, original_rules_unchanged=True)


def lower(row, key):
    return row[key]['completion_lower_pp'] if row[key] is not None else dict(point=None, interval95=None)


def render(result):
    def effect(row, ref):
        m = lower(row, ref)
        ci = m['interval95']
        return f'{m["point"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci is not None else 'NA'

    chosen = result['selected_arm']
    state = (f'**Eligible for the separately locked confirmation: {chosen}**, '
             f'margin {result["selected_margin"]:.2f}.' if chosen else
             '**No registered threshold qualifies.** Independent confirmation remains sealed.')
    lines = ['# DroneCrowd Calibration Outcome', '', state, '',
             '40 recordings, 15 exclusion groups, three frozen seeds and 120 repeated recording-seed views.',
             'The 3,000 paired bootstrap draws resample groups after seed averaging.',
             'This is calibration, not independent confirmation or deployment evidence.', '',
             '| Arm | Margin | Switches | Risk passes / 120 | Undefined easy | All harm >2% | Signed easy >2% |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result['rows']:
        margin = 'NA' if r['margin'] is None else f'{r["margin"]:.2f}'
        lines.append('| '+' | '.join((r['arm'], margin, f'{r["switches"]:,}', *(str(r[k]) for k in RISK[2:6])))+' |')
    lines.extend(('', 'Risk-diagnostic counts can overlap. An undefined selected reference is not a pass.', '',
        '| Arm | Conservative gain vs CV, pp [95% CI] | Conservative gain vs OLS4, pp [95% CI] | Eligibility |',
        '| --- | ---: | ---: | --- |'))
    for r in result['rows']:
        status = ('eligible' if r['eligible'] else ', '.join(r['disqualification'])) if r['arm'] in GATES else 'control'
        lines.append('| '+' | '.join((r['arm'], effect(r, 'vs_cv'), effect(r, 'vs_ols4'), status))+' |')
    lines.extend(('', 'Gains use the observed CV error mass as denominator. Distances are image pixels;',
        '8/12 refers to annotation steps. OLS4 is not a calibration gate.', ''))
    return '\n'.join(lines)


def table(result):
    buf = io.StringIO()
    fields = ('arm', 'margin', *RISK, 'eligible', 'disqualification')
    refs = ('vs_cv', 'vs_ols4')
    effect = ('point', 'ci_low', 'ci_high')
    w = csv.writer(buf, lineterminator='\n')
    w.writerow((*fields, *(f'{r}_{k}' for r in refs for k in effect)))
    for row in result['rows']:
        values = [row[k] if k != 'disqualification' else ';'.join(row[k]) for k in fields]
        for ref in refs:
            e = lower(row, ref)
            values += [e['point'], *(e['interval95'] or [None, None])]
        w.writerow(values)
    return buf.getvalue()


def write_once(path, contents):
    if path.is_file():
        if path.read_text() != contents:
            raise ValueError(f'Report already exists with different evidence: {path.name}')
        return
    tmp = path.with_suffix('.pending')
    tmp.write_text(contents)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PUBLIC/'readout')
    a = p.parse_args()
    try:
        doc, bindings = verify_artifacts(ROOT, PUBLIC, MANIFEST)
        result = dict(summarize(doc), bindings=bindings)
    except (FileNotFoundError, ValueError) as exc:
        p.exit(2, f'{exc}\n')
    a.output.mkdir(parents=True, exist_ok=True)
    write_once(a.output/'report.md', render(result))
    write_once(a.output/'table.csv', table(result))
    write_once(a.output/'readout.json', json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
