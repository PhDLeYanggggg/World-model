"""Audit current source event support; reuse fixed forecasts, train nothing."""
import json
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_sdd_state_support import load_source
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_motion_quality import trajectory_quality
from src.evaluation.m3w_source_temporal_information import describe
from src.evaluation.m3w_source_event_support import (
    past_episode_context, future_change_labels, nonoverlapping_window_count,
    neighbor_support, concentration,
)
from src.world_model.m3w_sdd_state_support import frame_lookup
import numpy as np
import torch

CONFIG = Path('configs/m3w_source_event_support_v1.json')


def grouped_mean(values, groups):
    _, inv = np.unique(groups, return_inverse=True)
    return float((np.bincount(inv, weights=values)/np.bincount(inv)).mean())


def fixed_score_diagnostic(a, episodes, evidence):
    base = ROOT/'outputs/publication_readiness_2026_09'
    outputs = {}
    for folder, arms in [('source_pretrained_temporal_v1', ['geometry', 'current', 'sequence']),
                         ('source_temporal_centered_v1', ['centered', 'centered_unit'])]:
        vp = base/folder/'verification.json'
        verification = json.loads(vp.read_text())
        rp = base/folder/'report.json'
        assert file_digest(rp) == verification['artifact_hashes'][str(rp.relative_to(ROOT))]
        evidence[str(vp.relative_to(ROOT))] = file_digest(vp)
        evidence[str(rp.relative_to(ROOT))] = file_digest(rp)
        report = json.loads(rp.read_text())
        for arm in arms:
            scores = []
            for seed in (17, 29, 43):
                item = next(x for x in report['oof_labels'] if x['arm'] == arm and x['seed'] == seed)
                pp = ROOT/item['path']
                assert file_digest(pp) == item['sha256']
                evidence[item['path']] = item['sha256']
                with np.load(pp, allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'], a['ids'])
                    error, cv = p['ade'].copy(), p['cv_ade'].copy()
                weightings = {}
                for name, grouping in [('windows', a['ids']), ('episodes', episodes), ('tracks', a['tracks'])]:
                    pairs = [(grouped_mean(error[a['sites'] == s], grouping[a['sites'] == s]),
                              grouped_mean(cv[a['sites'] == s], grouping[a['sites'] == s]))
                             for s in sorted(set(a['sites']))]
                    weightings[name] = dict(equal_site_gain_percent=float(100*(1-np.mean(pairs, axis=0)[0]/np.mean(pairs, axis=0)[1])),
                        sites={s:float(100*(1-x/y)) for s,(x,y) in zip(sorted(set(a['sites'])), pairs)})
                scores.append(dict(seed=seed, **weightings))
            outputs[arm] = dict(result_source='cached_verified_reweighted_no_new_forecasts', seeds=scores,
                mean_seed_gain={k:float(np.mean([x[k]['equal_site_gain_percent'] for x in scores]))
                                for k in ('windows', 'episodes', 'tracks')})
    return outputs


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    r = json.loads(CONFIG.read_text())
    assert r['rows'] == 15430 and r['sites'] == ['coupa', 'deathCircle', 'gates', 'hyang']
    assert not any(r[k] for k in ('new_training', 'new_forecasts', 'cohort_change', 'main_outer_processing'))
    public, private = ROOT/r['reports'], ROOT/r['output']
    evidence = {str(CONFIG):file_digest(CONFIG)}
    for name in ('scripts/audit_m3w_source_event_support.py', 'src/evaluation/m3w_source_event_support.py',
                 'tests/test_m3w_source_event_support.py',
                 'outputs/publication_readiness_2026_09/source_event_support_decision.md'):
        evidence[name] = file_digest(ROOT/name)
    read = lambda name:json.loads((ROOT/name).read_text())
    verification = read(r['quality_verification'])
    for name, sha in verification['artifact_hashes'].items():
        assert file_digest(ROOT/name) == sha, name
    prep = read(r['quality_preparation'])
    evidence[r['quality_preparation']] = file_digest(ROOT/r['quality_preparation'])
    evidence[r['quality_verification']] = file_digest(ROOT/r['quality_verification'])
    assert file_digest(ROOT/r['quality_rows']) == prep['rows_sha256']
    evidence[r['quality_rows']] = prep['rows_sha256']
    with np.load(ROOT/r['quality_rows'], allow_pickle=False) as z:
        a = {k:z[k].copy() for k in z.files}
    assert len(a['ids']) == r['rows'] and sorted(set(a['sites'])) == r['sites']
    audit = read(r['input_audit']); ip = audit['row_archive']['path']
    assert file_digest(ROOT/ip) == audit['row_archive']['sha256']
    evidence[ip] = audit['row_archive']['sha256']
    evidence[r['input_audit']] = file_digest(ROOT/r['input_audit'])
    with np.load(ROOT/ip, allow_pickle=False) as z:
        images = {k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(a['ids'], images.pop('ids'))
    n = len(a['ids']); values = {}; seen = np.zeros(n, bool); checks = 0
    for source in prep['raw_sources']:
        name = source['path']; assert file_digest(ROOT/name) == source['sha256']
        evidence[name] = source['sha256']
        raw, labels = load_source(ROOT/name)
        starts = np.r_[0, np.flatnonzero(np.diff(raw[:, 0]))+1]
        ends = np.r_[starts[1:], len(raw)]
        lookup = {int(raw[lo, 0]):(lo, hi) for lo,hi in zip(starts, ends)}
        selected = np.flatnonzero(a['records'] == source['recording'])
        assert len(selected) == source['rows']
        for agent in np.unique(a['query_keys'][selected, 1]):
            idx = selected[a['query_keys'][selected, 1] == agent]
            queries = a['query_keys'][idx, 0]
            lo, hi = lookup[int(agent)]; track = raw[lo:hi]
            hist = frame_lookup(track, queries[:, None]-np.arange(7, -1, -1)*12)
            fut = frame_lookup(track, queries[:, None]+np.arange(1, 13)*12)
            assert (hist >= 0).all() and (fut >= 0).all()
            assert not track[hist, 6].any() and not track[fut, 6].any()
            assert (labels[lo:hi][hist[:, -1]] == 'Pedestrian').all()
            xy = (track[:, 1:3]+track[:, 3:5])/2
            assert (xy[hist] == xy[hist[:, -1], None]).all()
            np.testing.assert_array_equal(xy[fut]-xy[hist[:, -1], None], a['future_native'][idx])
            past = past_episode_context(track, queries)
            future = future_change_labels(track, queries)
            for k,v in {**past, **future}.items():
                if k not in values: values[k] = np.empty(n, v.dtype)
                values[k][idx] = v
            for q in np.unique([queries.min(), queries.max()]):
                before = past_episode_context(track, [q])
                altered = track.copy(); altered[altered[:, 5] > q, 1:5] = np.nan
                for alternative in (altered, track[track[:, 5] <= q]):
                    after = past_episode_context(alternative, [q])
                    for k in before: np.testing.assert_array_equal(before[k], after[k])
                checks += 1
            seen[idx] = True
        print(json.dumps(dict(recording=source['recording'], completed_rows=int(seen.sum()))), flush=True)
    assert seen.all()
    values.update(neighbor_support(a['geometry']))
    episodes = np.array([f'{track}@{frame}' for track,frame in zip(a['tracks'],values['plateau_start_frame'])])
    quality = trajectory_quality(a['future_native'], a['past_box_scale'])
    subsets = dict(all=np.ones(n, bool), zero=~quality['nonzero'], nonzero=quality['nonzero'],
        above_ten_pixels=quality['max_pixel_displacement'] > 10,
        half_box_excursion=quality['half_box_excursion'],
        final_four_outside_half_box=quality['final_four_outside_half_box'])
    def summarize(m):
        rows = int(m.sum())
        if not rows: return dict(rows=0)
        groups = episodes[m]
        _, counts = np.unique(groups, return_counts=True)
        result = dict(rows=rows, tracks=len(set(a['tracks'][m])), recordings=len(set(a['records'][m])),
            sites=len(set(a['sites'][m])), episode_groups=len(counts),
            disjoint_track_window_spans=nonoverlapping_window_count(a['tracks'][m], a['query_keys'][m,0]),
            windows_per_episode=describe(counts), episode_window_concentration=concentration(np.ones(rows), groups),
            complete_raw_static_history=int(values['full_raw_history_static'][m].sum()),
            complete_raw_future=int(values['future_raw_complete'][m].sum()),
            raw_changed_future=int((values['first_changed_raw_frame'][m] >= 0).sum()),
            past_later_control=int(a['past_has_later_control'][m].sum()),
            no_current_neighbor=int((values['current_neighbor_count'][m] == 0).sum()),
            no_moving_neighbor=int((values['moving_neighbor_count'][m] == 0).sum()),
            no_full_eight_neighbor=int((values['full_eight_neighbor_count'][m] == 0).sum()),
            past_unoccluded=int((~a['annotation_flags'][m,:8,0].any(1)).sum()),
            past_control_count=describe(values['past_nonlost_control_rows'][m]),
            past_plateau_age=describe(values['plateau_age_raw_frames'][m]),
            observed_image={k:describe(images[k][m]) for k in ('mean_coverage', 'box_width_output_pixels',
                'box_height_output_pixels', 'box_pixel_change', 'temporal_embedding_energy_fraction')})
        return result
    strata = {k:summarize(m) for k,m in subsets.items()}
    by_site = {s:{k:summarize(m & (a['sites'] == s)) for k,m in subsets.items()} for s in r['sites']}
    mixed = sum(bool(quality['nonzero'][episodes == g].any()) and
                bool((~quality['nonzero'][episodes == g]).any()) for g in np.unique(episodes))
    scores = fixed_score_diagnostic(a, episodes, evidence)
    path = private/'event_rows.npz'
    save_arrays(path, dict(ids=a['ids'], episodes=episodes, **values))
    result = dict(result_source='fresh_run_raw_event_audit_cached_verified_context_and_forecasts',
        rows=n, scopes=strata, by_site=by_site, episode_groups_containing_both_labels=mixed,
        forecast_reweighting=scores, input_mutation_and_truncation_queries=checks,
        row_archive=dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path)), evidence_hashes=evidence,
        groups_are_independent_physical_events=False, no_new_fits=True, no_new_predictions=True,
        rows_dropped=0, main_outer_rows_processed=0, model_or_threshold_selection=False,
        sensor_asof_certified=False, coordinate_unit='annotation_pixel_raw_frame_only',
        new_deployment=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(public/'audit.json', result)
    print(json.dumps(dict(scopes={k:{f:v[f] for f in ('rows','tracks','episode_groups','disjoint_track_window_spans',
        'no_current_neighbor','no_moving_neighbor','complete_raw_static_history')} for k,v in strata.items()},
        reweighted={k:v['mean_seed_gain'] for k,v in scores.items()}, checks=checks), indent=2))


if __name__ == '__main__': main()
