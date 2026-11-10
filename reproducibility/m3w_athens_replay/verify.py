"""Offline scalar replay; no fitting, target join, or deployment certification."""

from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ATHENS = ('registration', 'freeze', 'confirmation', 'recipient_path_attribution',
          'stepwise_budget_sensitivity')
MEMBERS = {'README.md', 'verify.py', 'prior_evidence.zip', 'supplement.pdf',
           'comparison.json', *(f'{k}.json' for k in ATHENS)}


def date_envelope(metrics, counts):
    fields = ('lower_pp', 'observed_pp', 'upper_pp')
    for field in fields:
        m = metrics[field]
        if (m['member_count_by_date'] != dict(counts) or set(m['by_date']) != set(counts)
                or m['bootstrap_unit'] != 'acquisition_date_after_seed_then_member_average'
                or not m['one_city'] or m['status'] != 'descriptive_date_bootstrap'
                or not all(isfinite(v) for v in m['by_date'].values())
                or not isclose(mean(m['by_date'].values()), m['point'], abs_tol=1e-12)):
            raise ValueError('Invalid date aggregation')
    rows = {day: [metrics[f]['by_date'][day] for f in fields] for day in sorted(counts)}
    if any(not a <= b <= c for a, b, c in rows.values()):
        raise ValueError('Invalid missing-position envelope')
    return dict(by_date=rows, mean=[mean(v[i] for v in rows.values()) for i in range(3)],
                positive_dates=sum(v[0] > 0 for v in rows.values()),
                unresolved_dates=sum(v[0] <= 0 <= v[2] for v in rows.values()))


def paired_interval(original, matched):
    delta = [b[1]-a[1] for a, b in zip(original, matched, strict=True)]
    rng = Random(20261110)
    draws = sorted(mean(rng.choices(delta, k=len(delta))) for _ in range(3000))
    def quantile(p):
        j = p*(len(draws)-1)
        i = int(j)
        return draws[i]+(j-i)*(draws[i+1]-draws[i])
    return mean(delta), [quantile(.025), quantile(.975)]


def replay(raw):
    d = {k: json.loads(v) for k, v in raw.items()}
    reg, freeze, primary = (d[k] for k in ATHENS[:3])
    counts = Counter(reg['dates'].values())
    h = {k: sha256(raw[k]).hexdigest() for k in ATHENS}
    spec = reg['protocol']
    if (len(counts) != 4 or len(reg['dates']) != 6 or primary['dates'] != reg['dates']
            or set(primary['members']) != set(reg['dates'])
            or reg['member_roles'] != dict.fromkeys(reg['dates'], 'confirmation')
            or spec['date_roles'] != dict.fromkeys(counts, 'confirmation')
            or (spec['observed_steps'], spec['prediction_steps']) != (8, 12)
            or spec['mode'] != 'zero_shot' or spec['risk_budget'] != .02
            or freeze['registration'] != h['registration'] or freeze['thresholds'] is not None
            or freeze['label_access'] or not primary['roles_frozen_before_read']
            or primary['binding']['registration'] != h['registration']
            or primary['binding']['freeze'] != h['freeze'] or primary['binding']['role'] != 'confirmation'
            or primary['mode'] != 'zero_shot' or primary['official_risk_budget'] != .02
            or primary['new_fit'] or primary['deployment_changed']
            or primary['release'] != 'not_run_zero_shot_fixed_count_only'):
        raise ValueError('Athens is a frozen zero-shot utility comparison, not a release test')
    attr, steps = (d[k] for k in ATHENS[3:])
    if (attr['primary_confirmation'] != h['confirmation'] or not attr['no_threshold_changes']
            or steps['primary_confirmation'] != h['confirmation']
            or steps['attribution_receipt'] != h['recipient_path_attribution']
            or not steps['primary_unchanged'] or steps['release_recomputed']
            or steps['thresholds_changed'] or steps['source_date_count'] != 4):
        raise ValueError('Secondary controls must keep the primary result frozen')
    m = primary['matched']
    contrasts = {'joint_vs_cv': m['by_arm']['joint']['vs_reference']['cv'],
                 'same_family_vs_cv': m['by_arm']['same_family']['vs_reference']['cv'],
                 **{f'joint_vs_{k}': v for k, v in m['joint_minus'].items()},
                 'equal_steps': steps['by_comparator']['independent']['matched_steps']}
    result = {k: date_envelope(v, counts) for k, v in contrasts.items()}
    comp = d['comparison']
    if (comp['pooling'] or comp['new_outcomes_read'] or comp['new_fit']
            or comp['release_recomputed'] or comp['original_risk_fraction'] != .02
            or comp['source_sha256']['athens'] != h['confirmation']
            or comp['source_sha256']['athens_steps'] != h['stepwise_budget_sensitivity']):
        raise ValueError('Comparison roles or Athens bindings differ')
    comparisons = {}
    for key, expected_n in (('wep', 10), ('songdo', 14), ('athens', 4)):
        row = comp['cohorts'][key]
        original, matched = row['original'], row['equal_steps']
        if (original['eligible_indices'] != matched['eligible_indices']
                or len(set(original['eligible_indices'])) != expected_n
                or row['units_resampled'] != expected_n or row['bootstrap_draws'] != 3000):
            raise ValueError('Changed matched source roster')
        for arm in (original, matched):
            scores = arm['group_scores']
            if (len(scores) != expected_n or any(not all(isfinite(x) for x in v)
                    or not v[0] <= v[1] <= v[2] for v in scores)):
                raise ValueError('Invalid source envelope')
            center = [mean(v[i] for v in scores) for i in range(3)]
            if not all(isclose(a, b, abs_tol=1e-12) for a, b in zip(center,
                    [arm['envelope_pp'][0], arm['observed_pp'], arm['envelope_pp'][1]], strict=True)):
                raise ValueError('Source means differ from the reported contrast')
        value, interval = paired_interval(original['group_scores'], matched['group_scores'])
        if value != row['equal_steps_minus_original_observed_pp'] or interval != row['paired_source_ci95_pp']:
            raise ValueError('Paired interval did not reproduce')
        comparisons[key] = dict(change_pp=value, ci95_pp=interval, source_units=expected_n)
    for key, arm in (('joint_vs_independent', 'original'), ('equal_steps', 'equal_steps')):
        if list(result[key]['by_date'].values()) != comp['cohorts']['athens'][arm]['group_scores']:
            raise ValueError('Athens source alignment differs between reports')
    return dict(athens=result, comparison=comparisons, athens_release_evaluated=False,
                joint_advantage_established=result['joint_vs_independent']['mean'][0] > 0,
                original_risk_fraction=.02,
                replay_scope='source_scalar_aggregation_not_training_or_raw_annotation_reconstruction')


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_bytes())
    if set(manifest) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for name, expected in manifest.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed archive member')
    raw = {k: (root/f'{k}.json').read_bytes() for k in (*ATHENS, 'comparison')}
    output = replay(raw)
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        folder = Path(tmp)
        for n in names:
            (folder/n).write_bytes(z.read(n))
        output['inherited'] = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return output


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
