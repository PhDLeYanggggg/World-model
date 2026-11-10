"""Bound the one-shot WEPDTOF result from frozen paths and score aggregates."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.world_model.m3w_wepdtof_matched_policy import bank


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/stage_cvpr2027_experiments/wepdtof_external_transfer_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
STUDY = PUBLIC/'wepdtof_external_transfer_v1'
OUT = PUBLIC/'wepdtof_frozen_capacity_v1/receipt.json'
READOUT = '7f07eecce91c8bb6e72f792a6e0dbe4d0b228f17c97cf0ac87d8a961173062d8'
FREEZE = '59087d08426e6018a9ea71c822d745d12ff044b0494bf128d29f7c6895869a7e'
SEEDS = (17, 29, 43)
MATCHED = ('fixed_joint', 'fixed_independent', 'fixed_uniform', 'fixed_no_pair')
KEYS = ('arbitrary_radius', 'menu_cv_anchor', 'menu_multi_anchor')
EPS = 1e-7


def grouping(query, distance):
    groups = []
    radii = distance.max(-1)
    legal = (radii > 0.).all(1)
    for q in np.unique(query):
        ix = np.flatnonzero(query == q)
        live = ix[legal[ix]]
        n = min(2, len(live), len(ix)-1)
        radius = radii[live].max() if n else 0.
        if n and not np.allclose(radii[live], radius, atol=1e-8, rtol=1e-8):
            raise ValueError('Candidate radii no longer match within a query')
        groups.append((ix, live, n, float(radius)))
    return groups


def menu_distance(cv, paths, anchor, groups):
    """Max weighted path distance to an anchor over all at-most-count edits."""
    d = np.linalg.norm(cv-anchor, axis=-1).sum(1)
    delta = np.linalg.norm(paths-anchor[:, None], axis=-1).sum(2).max(1)-d
    total = 0.
    for ix, live, n, _ in groups:
        gain = np.sort(np.maximum(delta[live], 0.))[-n:].sum() if n else 0.
        total += (d[ix].sum()+gain)/len(ix)
    return float(total)


def archived_actions(cv, paths, output, groups, selected):
    edited = np.any(output != cv, axis=(1, 2))
    supported = np.isclose(output[:, None], paths, atol=1e-8, rtol=1e-8).all((2, 3)).any(1)
    if int(edited.sum()) != selected or not supported[edited].all():
        raise ValueError('Archived action count or frozen path does not match')
    for ix, live, n, _ in groups:
        active = ix[edited[ix]]
        if len(active) != n or not np.isin(active, live).all():
            raise ValueError('Archived action violates the fixed query budget')


def upper_bounds(cv, paths, groups, anchors, mass):
    unrestricted = sum(len(cv[0])*n*r/len(ix) for ix, _, n, r in groups)
    distances = {k: 100*menu_distance(cv, paths, path, groups)/mass for k, (path, _) in anchors.items()}
    result = {}
    for field in ('known_vs_ols4', 'lower_vs_ols4'):
        bounds = {k: row[field]+distances[k] for k, (_, row) in anchors.items()}
        result[field] = dict(zip(KEYS, (anchors['cv'][1][field]+100*unrestricted/mass,
                                      bounds['cv'], min(bounds.values())), strict=True))
    return result


def checked(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f'Frozen artifact changed: {path.name}')
    return raw


def summarize(rows):
    active = [r for r in rows if r['edits'] and r['bounds'] is not None]
    risk = [r for r in active if r['actual_risk_pass']]
    result = {}
    for label, part in (('edited_estimable', active), ('actual_joint_risk_pass', risk)):
        clips = {r['clip'] for r in part}
        sources = {r['source'] for r in part}
        result[label] = dict(views=len(part), clips=len(clips), sources=len(sources))
        for field in ('known_vs_ols4', 'lower_vs_ols4'):
            result[label][field] = {}
            for k in KEYS:
                x = np.array([r['bounds'][field][k] for r in part])
                possible = [r for r, v in zip(part, x, strict=True) if v >= -EPS]
                result[label][field][k] = dict(excluded=int((x < -EPS).sum()),
                                              clips_all_edited_views_excluded=len(clips-{r['clip'] for r in possible}),
                                              sources_all_edited_views_excluded=len(sources-{r['source'] for r in possible}),
                                              median_upper_pp=float(np.median(x)) if len(x) else None)
    result.update(views=len(rows), edits=sum(r['edits'] for r in rows), no_edit_views=sum(not r['edits'] for r in rows),
                  unestimable_views=sum(r['bounds'] is None for r in rows),
                  positive_actual_ols4_lower=sum(r['actual_ols4_lower'] is not None and r['actual_ols4_lower'] > 0
                                                for r in rows),
                  positive_actual_ols4_lower_with_edits=sum(r['actual_ols4_lower'] is not None and r['actual_ols4_lower'] > 0
                                                           for r in active))
    return result


def run():
    scores = json.loads(checked(STUDY/'external_readout.json', READOUT))
    frozen = json.loads(checked(STUDY/'augmented_freeze.json', FREEZE))
    if (scores['forecast_freeze'] != FREEZE or not scores['external_scoring_completed']
            or not scores['original_2pct'] or scores['external_tuning'] or scores['dronecrowd_confirmation_access']
            or not frozen['outcomes_unopened'] or not frozen['original_2pct'] or frozen['seeds'] != list(SEEDS)):
        raise ValueError('Expected the original one-shot readout and pre-outcome forecast freeze')
    score_rows = {r['clip']: r for r in scores['rows']}
    if (len(score_rows) != 15 or len({r['source'] for r in score_rows.values()}) != 13
            or len(frozen['rows']) != 15 or set(score_rows) != {r['clip'] for r in frozen['rows']}):
        raise ValueError('External roster changed')
    rows = []
    for item in frozen['rows']:
        clip, artifact = item['clip'], item['artifact']
        record = score_rows[clip]
        if (record['source'] != item['source'] or record['online'] != item['online']
                or record['freeze'] != FREEZE or Path(artifact['path']) != (DATA/f'{clip}_augmented.npz').relative_to(ROOT)):
            raise ValueError('Archived forecast and readout do not align')
        with np.load(BytesIO(checked(ROOT/artifact['path'], artifact['sha256'])), allow_pickle=False) as z:
            q = z['q']
            for s in SEEDS:
                view = record['seeds'][str(s)]
                if len(q) != record['online'] or view['online'] != len(q):
                    raise ValueError('Query roster differs from the recorded view')
                arms, mass = view['arms'], view['reference_mass']
                actual = arms.get('fixed_joint', {})
                row = dict(clip=clip, source=item['source'], edits=actual.get('risk', {}).get('selected', 0), bounds=None,
                           actual_ols4_lower=actual.get('lower_vs_ols4'),
                           actual_risk_pass=actual.get('risk', {}).get('supported', False))
                if len(q):
                    cv, ols, spline = (z[f'{s}_{k}'] for k in ('cv', 'ols4', 'spline'))
                    if cv.shape != (len(q), 12, 2) or cv.shape != ols.shape or cv.shape != spline.shape:
                        raise ValueError('Forecasts no longer use the registered 12-step grid')
                    paths = bank(cv, ols, spline, q)
                    groups = grouping(q, np.linalg.norm(paths-cv[:, None], axis=-1))
                    for k in MATCHED:
                        archived_actions(cv, paths, z[f'{s}_{k}'], groups, arms[k]['risk']['selected'])
                    if mass > 0.:
                        anchors = {k: (z[f'{s}_{k}'], v) for k, v in arms.items()}
                        row['bounds'] = upper_bounds(cv, paths, groups, anchors, mass)
                        for field, bounds in row['bounds'].items():
                            if max(arms[k][field] for k in MATCHED) > min(bounds.values()) + EPS:
                                raise ValueError('An upper bound excludes an actual archived action')
                rows.append(row)
    summary = summarize(rows)
    if (summary['views'], summary['edits'], summary['no_edit_views'], summary['positive_actual_ols4_lower'],
            summary['actual_joint_risk_pass']['views']) != (45, 366, 9, 10, 4):
        raise ValueError('Archived external summary changed')
    return dict(**summary, sources=13, clips=15, seeds=list(SEEDS), forecast_freeze_sha256=FREEZE,
                readout_sha256=READOUT, result_source='fresh_bound_cached_verified_frozen_predictions_and_aggregates',
                original_2pct_unchanged=True, outcomes_reopened=False, policy_changed=False,
                independent_confirmation=False, submission_ready=False)


if __name__ == '__main__':
    receipt = (json.dumps(run(), indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_bytes() != receipt:
        raise ValueError('Refusing to replace a different analysis receipt')
    if not OUT.exists():
        with OUT.open('xb') as f:
            f.write(receipt)
    print(receipt.decode(), end='')
