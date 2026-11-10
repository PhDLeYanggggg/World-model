"""Two-sided missing-label and source-sampling sensitivity on frozen WEPDTOF."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.focused_review_supplement.wepdtof_completion import CONTROLS, group_bounds
from reproducibility.wepdtof_scalar_replay.verify import exact_interval, interval, mean

PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
REPLAY = PUBLIC/'wepdtof_scalar_replay_v1/readout.json'
ARCHIVE = ROOT/'data/stage_cvpr2027_experiments/submission_admin/wepdtof_scalar_replay.zip'
OUT = PUBLIC/'wepdtof_missing_label_interval_v1/receipt.json'

def verdict(low, high, tol=1e-12):
    if low > tol:
        return 'positive_under_all_missing_completions'
    if high < -tol:
        return 'negative_under_all_missing_completions'
    if abs(low) <= tol and abs(high) <= tol:
        return 'score_tie'
    return 'sign_not_resolved_by_bounds'


def analyze(groups, total, draws):
    if len(groups) < 2:
        raise ValueError('Need at least two estimable original source groups')
    cols = list(zip(*groups, strict=True))
    parts = {k: interval(list(col), total, draws) | dict(exact_ci95=exact_interval(list(col)))
             for k, col in zip(('lower', 'observed', 'upper'), cols, strict=True)}
    low, high = (parts[k]['estimate'] for k in ('lower', 'upper'))
    loo = [(mean([r[0] for i, r in enumerate(groups) if i != j]),
            mean([r[2] for i, r in enumerate(groups) if i != j])) for j in range(len(groups))]
    state = tuple(verdict(l, h) for l, _, h in groups)
    return dict(estimates=parts, identification_envelope_pp=[low, high],
                empirical_bootstrap_outer_pp=[parts['lower']['exact_ci95'][0], parts['upper']['exact_ci95'][1]],
                estimable_source_states={s: state.count(s) for s in sorted(set(state))},
                mean_sign=verdict(low, high), leave_one_source_out=dict(
                    lower_range_pp=[min(x[0] for x in loo), max(x[0] for x in loo)],
                    upper_range_pp=[min(x[1] for x in loo), max(x[1] for x in loo)],
                    sign_unresolved=sum(verdict(l, h) == 'sign_not_resolved_by_bounds' for l, h in loo)))


def run():
    prior = json.loads(REPLAY.read_text())
    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    if (prior['archive_sha256'] != digest or not prior['no_new_label_access'] or not prior['original_2pct']
            or prior['independent_confirmation'] or prior['arms']['fixed_joint']['dual_passes'] != 0):
        raise ValueError('Expected the original verified scalar-only archive')
    with ZipFile(ARCHIVE) as z:
        e, r = (json.loads(z.read(n)) for n in ('evidence.json', 'resamples.json'))
    groups = e['groups']
    if len(groups) != 13 or sum(map(len, groups)) != 15 or any(
            [v['seed'] for v in c] != [17, 29, 43] for g in groups for c in g):
        raise ValueError('The original source and seed roster changed')
    comparisons = {}
    for arm in CONTROLS:
        rows = group_bounds(e, arm)
        if len(rows) != 10:
            raise ValueError('Matched-source support changed')
        new = analyze(rows, len(groups), r)
        a, b = new['estimates']['lower'], prior['matched'][arm]
        for x, y in zip((a['estimate'], *a['ci95'], *a['exact_ci95']),
                        (b['estimate'], *b['ci95'], *b['exact_ci95']), strict=True):
            if not isclose(x, y, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError('Original conservative effect did not replay')
        comparisons[arm] = new
    return dict(sources=13, matched_estimable_sources=10, clips=15, views=45, comparisons=comparisons,
                scalar_archive_sha256=digest, original_fixed_joint_dual_passes=0,
                original_2pct_unchanged=True, outcome_access=False, submission_ready=False,
                result_source='fresh_two_sided_sensitivity_of_frozen_scalar_readout')


if __name__ == '__main__':
    raw = (json.dumps(run(), indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_bytes() != raw:
        raise FileExistsError('Refusing to replace a different sensitivity receipt')
    if not OUT.exists():
        with OUT.open('xb') as f:
            f.write(raw)
    print(raw.decode(), end='')
