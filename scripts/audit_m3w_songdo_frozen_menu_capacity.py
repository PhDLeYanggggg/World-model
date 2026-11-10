"""Measure hindsight release headroom from the archived Songdo score scalars."""

from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.songdo_review_supplement import replay_songdo as old


PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
ARCHIVE = ROOT/'data/stage_cvpr2027_experiments/submission_admin/review_supplement_v3.zip'
OUT = PUBLIC/'songdo_frozen_menu_capacity_v1/receipt.json'
MENUS = {**{a: (a,) for a in old.ARMS}, 'without_joint': old.ARMS[:-1], 'all': old.ARMS}


def best(view, menu):
    """Highest positive CV-relative lower score among dual-passing proposals."""
    result = 0.
    for arm in menu:
        cell = view['arms'][arm]
        cv, ols = cell['gains']
        if any(x is not None and not isfinite(x) for x in (cv, ols)):
            raise ValueError('A conservative score must be finite or undefined')
        if old.ratio_state(cell['risk']) and cv is not None and ols is not None and min(cv, ols) > 0.:
            result = max(result, cv)
    return result


def select_counts(groups, predicate):
    flags = [[[bool(predicate(view)) for view in flight] for flight in group] for group in groups]
    return dict(views=sum(sum(f) for g in flags for f in g),
                flights_any_seed=sum(any(f) for g in flags for f in g),
                flights_all_seeds=sum(all(f) for g in flags for f in g),
                source_components=sum(any(any(f) for f in g) for g in flags))


def diagnostic(evidence, frozen):
    groups = evidence['confirmation']
    draws = evidence['source_resamples']
    scored = [[[{**v, 'capacity': {k: best(v, a) for k, a in MENUS.items()}} for v in f] for f in g] for g in groups]
    by_menu = {}
    for key in MENUS:
        by_menu[key] = dict(dual_opportunity=select_counts(scored, lambda v: v['capacity'][key] > 0.),
                            conservative_cv_gain_ceiling_pp=old.interval(scored, lambda v: v['capacity'][key], draws))
    for a in old.ARMS:
        if by_menu[a]['dual_opportunity']['views'] != frozen['matched_risk'][a]['dual_pass']:
            raise ValueError('A single-arm ceiling must recover the original proposal counts')

    def increment(more, less):
        delta = lambda v: v['capacity'][more]-v['capacity'][less]
        return dict(additional_dual_opportunity=select_counts(scored, lambda v: v['capacity'][more] > 0. and v['capacity'][less] == 0.),
                    strictly_higher_ceiling=select_counts(scored, lambda v: delta(v) > 0.),
                    delta_of_conservative_cv_ceiling_pp=old.interval(scored, delta, draws))

    threshold = frozen['thresholds']
    actual = {}
    for arm in old.ARMS:
        def edit(v):
            return old.released(v['arms'][arm]['score'], threshold[str(v['seed'])][arm])

        good = lambda v: edit(v) and v['capacity'][arm] > 0.
        missed = lambda v: not edit(v) and v['capacity'][arm] > 0.
        counts = frozen['actual_risk'][arm]
        captured, uncaptured = select_counts(scored, good), select_counts(scored, missed)
        if (captured['views'] != counts['dual_pass'] or captured['views'] + uncaptured['views']
                != by_menu[arm]['dual_opportunity']['views']):
            raise ValueError('Original releases do not partition the verified opportunities')
        actual[arm] = dict(released_views=counts['edited'], captured_dual_opportunities=captured,
                           missed_dual_opportunities=uncaptured,
                           released_without_dual_pass=counts['edited']-counts['dual_pass'])
    return dict(views=frozen['views'], flights=frozen['flights'], source_components=frozen['source_components'],
                seeds=old.SEEDS, bootstrap_draws=len(draws), menu=by_menu, frozen_release=actual,
                other_nonjoint_added_to_independent=increment('without_joint', 'independent'),
                joint_added_to_other_arms=increment('all', 'without_joint'))


def verified_archive(archive):
    receipt = json.loads((PUBLIC/'focused_review_supplement_v1/receipt.json').read_text())
    source = archive.read_bytes()
    digest = sha256(source).hexdigest()
    if receipt['archive_sha256'] != digest or not receipt['original_2pct_unchanged'] or receipt['external_labels_reopened']:
        raise ValueError('Expected the frozen v3 anonymous replay')
    with ZipFile(BytesIO(source)) as z:
        raw = z.read('songdo_replay.zip')
        if sha256(raw).hexdigest() != json.loads(z.read('manifest.json'))['songdo_replay.zip']:
            raise ValueError('Songdo nested archive has changed')
    with ZipFile(BytesIO(raw)) as z:
        data = z.read('evidence.json')
        if sha256(data).hexdigest() != json.loads(z.read('manifest.json'))['evidence.json']:
            raise ValueError('Songdo scores have changed')
    evidence = json.loads(data)
    frozen = old.replay(evidence)
    published = json.loads((PUBLIC/'songdo_calibrated_20261110/confirmation_receipt.json').read_text())
    if (frozen['actual_risk'] != receipt['songdo_actual_release'] or not published['no_fitting_on_confirmation']
            or any(frozen['matched_risk'][a] != published['matched'][a]['risk'] for a in old.ARMS)):
        raise ValueError('Archived scalars do not reproduce the frozen result')
    return evidence, frozen, dict(archive_sha256=digest, songdo_scalar_sha256=sha256(data).hexdigest()), data


def run(archive):
    evidence, frozen, binding, _ = verified_archive(archive)
    return dict(**diagnostic(evidence, frozen), **binding,
                result_source='fresh_post_hoc_finite_menu_analysis_of_frozen_score_scalars',
                original_2pct_unchanged=True, outcomes_reopened=False, test_threshold_selected=False,
                new_confirmation=False, learned_policy=False, deployment_changed=False, submission_ready=False)


if __name__ == '__main__':
    result = run(ARCHIVE)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_bytes() != raw:
        raise FileExistsError('Refusing to overwrite a different analysis receipt')
    if not OUT.exists():
        with OUT.open('xb') as f:
            f.write(raw)
    print(json.dumps({k: result[k] for k in ('menu', 'joint_added_to_other_arms')}, indent=2))
