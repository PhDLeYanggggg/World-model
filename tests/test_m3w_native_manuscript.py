import json

import pytest

from scripts import build_m3w_native_paper_v10 as paper
from scripts.build_m3w_native_paper_v10 import latex_document


def test_ast_preserves_aligned_math_and_escapes_table_cells():
    text = """# Forecast Study

## Abstract

Only measured results.

## 1. Method

$$
\\begin{aligned}
B&=1,\\\\
H&=2.
\\end{aligned}
$$

### Table 1. Observed development data

| Input | Gain |
|---|---:|
| No_image | 2% |

## References

1. https://example.org/paper
"""
    tex = latex_document(text)
    assert r'\begin{aligned}' in tex
    assert r'\end{aligned}' in tex
    assert r'No\_image & 2\%' in tex
    assert r'\captionof{table}{Observed development data}' in tex
    assert r'\begin{minipage}{\columnwidth}' in tex
    assert r'\section{Method}' in tex
    assert r'\section*{References}' in tex
    assert r'\clearpage\section*{References}' not in tex
    assert tex.count(r'\begin{abstract}') == tex.count(r'\end{abstract}') == 1


def test_paper_figures_reject_external_or_data_images():
    with pytest.raises(ValueError, match='generated scalar figures'):
        latex_document('# Study\n\n![Private image](data/image.png)\n')


def test_tables_require_explicit_captions():
    with pytest.raises(ValueError, match='caption'):
        latex_document('# Study\n\n| Input | Gain |\n|---|---|\n| A | B |\n')


def camera_fixture(tmp_path, monkeypatch, with_replay=True):
    home = tmp_path/'public'; folder = home/'moving_camera_frame_v1'
    folder.mkdir(parents=True)
    monkeypatch.setattr(paper, 'HOME', home)
    monkeypatch.setattr(paper, 'ROOT', tmp_path)
    records = [dict(contrast=contrast, summary=[dict(paired_known_pp=1., paired_completion_lower_pp=-2.) for _ in range(4)])
        for contrast in ('primary', 'camera_cv_effect', 'neural_over_camera_cv') for _ in range(14)]
    files = {'analysis.json': dict(primary=dict(known_pp=1., completion_lower_pp=-2., interval95=[-3., -1.],
        positive_known_sources=9, component_progression_screen_pass=False), sources=records,
        recordings=[dict(contrast='primary', summary=[dict(treatment=dict(known_easy_positive_harm_percent=3.))]) for _ in range(34)]),
        'training.json': dict(models=6), 'verification.json': dict(models=1, full_refits_exact=True)}
    for name, value in files.items(): (folder/name).write_text(json.dumps(value))
    (folder/'fits').mkdir()
    control = home/'moving_producer_crossfit_v1/fits'; control.mkdir(parents=True)
    for i in range(6):
        (folder/'fits'/f'job{i}.json').write_text(json.dumps(dict(result=dict(losses=[dict(loss=2.)]))))
        (control/f'job{i}.json').write_text(json.dumps(dict(result=dict(losses=[dict(loss=1.)]))))
    if with_replay:
        replays = tmp_path/'data/stage_cvpr2027_experiments/moving_camera_frame_v1/replays'
        replays.mkdir(parents=True); (replays/'replay_unit.json').write_text('{}')
    return folder, files


def test_camera_results_require_completed_exact_replay(tmp_path, monkeypatch):
    camera_fixture(tmp_path, monkeypatch, with_replay=False)
    with pytest.raises(ValueError, match='completed exact replay'):
        paper.camera_update()


def test_camera_table_keeps_negative_bound_and_distinct_controls(tmp_path, monkeypatch):
    camera_fixture(tmp_path, monkeypatch)
    text = paper.camera_update()
    assert 'not supported' in text and '-2.000' in text
    assert 'Camera CV - raw CV' in text and 'Actor - camera CV' in text
    assert 'not six complete treatment refits' in text
    assert 'not neural added value' in text


def test_camera_table_never_uses_defined_only_source_mean(tmp_path, monkeypatch):
    folder, files = camera_fixture(tmp_path, monkeypatch)
    files['analysis.json']['sources'][0]['summary'][0]['paired_known_pp'] = None
    (folder/'analysis.json').write_text(json.dumps(files['analysis.json']))
    with pytest.raises(ValueError, match='defined sources only'):
        paper.camera_update()


def test_aggregate_reporting_artifact_never_becomes_full_model_reproduction(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, 'HOME', tmp_path)
    folder = tmp_path/'native_actor_reproduction_v1'; folder.mkdir()
    value = dict(clean_execution=dict(exit_code=0, stdlib_only=True, site_packages_disabled=True),
        archive_members=6, archive_bytes=253746, new_training=False, new_prediction_or_target_readout=False,
        new_random_bootstrap=False, full_training_reproduction=False, independent_confirmation=False, submission_ready=False)
    path = folder/'verification.json'; path.write_text(json.dumps(value))
    text = paper.native_portable_update()
    assert '253,746' in text and 'not new independent evidence' in text
    value['full_training_reproduction'] = True; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='full model replication'):
        paper.native_portable_update()


def dual_view_fixture(tmp_path, monkeypatch, with_replay=True):
    monkeypatch.setattr(paper, 'HOME', tmp_path)
    folder = tmp_path/'moving_dual_view_actor_v1'; folder.mkdir()
    primary = dict(known_pp=.828, completion_lower_pp=-4.979, interval95=[-7.692, -2.333],
        positive_known_sources=10, component_progression_screen_pass=False)
    raw = dict(known_pp=-.321, completion_lower_pp=-5.995, interval95=[-8.461, -3.530],
        positive_known_sources=6, component_progression_screen_pass=False)
    a = dict(primary=primary, camera_bundle=raw, component_progression_screen_pass=False,
        sources=[dict(contrast=c, summary=[dict(rows=8967+(i < 2))])
            for c in ('primary', 'camera_bundle') for i in range(14)],
        recordings=[dict(contrast='primary', summary=[dict(treatment=dict(known_easy_positive_harm_percent=3.))])
            for _ in range(34)])
    for name, value in {
            'analysis.json': a, 'training.json': dict(models=12, steps_each=4000),
            'verification.json': dict(models=2, full_refits_exact=True),
            'forecast_freeze.json': dict(all_new_predictions_frozen_before_scoring=True)}.items():
        (folder/name).write_text(json.dumps(value))
    if with_replay:
        (folder/'replay_verification.json').write_text(json.dumps(dict(
            analysis_sha256=paper.sha(folder/'analysis.json'),
            forecast_freeze_sha256=paper.sha(folder/'forecast_freeze.json'),
            new_prediction_banks_exact=12, all_scalar_results_exact=True, replay_exit_code_observed=0)))
    return folder, a


def test_dual_view_requires_bound_completed_replay(tmp_path, monkeypatch):
    dual_view_fixture(tmp_path, monkeypatch, with_replay=False)
    with pytest.raises(ValueError, match='bound exact replay'):
        paper.dual_view_update()


def test_dual_view_preserves_both_required_contrasts(tmp_path, monkeypatch):
    dual_view_fixture(tmp_path, monkeypatch)
    text = paper.dual_view_update()
    assert 'Camera - native node | +0.828 | -4.979' in text
    assert 'Camera - original | -0.321 | -5.995' in text
    assert 'Both comparisons must pass' in text and 'not supported' in text
    assert 'not the full-grid error' in text


def test_dual_view_rejects_changed_results_after_replay(tmp_path, monkeypatch):
    folder, a = dual_view_fixture(tmp_path, monkeypatch)
    a['camera_bundle']['known_pp'] = 1.
    (folder/'analysis.json').write_text(json.dumps(a))
    with pytest.raises(ValueError, match='exact replay evidence required'):
        paper.dual_view_update()


def test_manuscript_disclosure_precedes_references_and_retains_negative_scope():
    template = (paper.PUBLIC/'manuscript.template.md').read_text()
    assert template.index('## LLM Usage Statement') < template.index('## References')
    tex = paper.latex_document((paper.PUBLIC/'manuscript.md').read_text())
    assert tex.index(r'\section*{LLM Usage Statement}') < tex.index(r'\section*{References}')
    assert 'The component fails' in paper.fixed_count_update()
    assert 'not estimable' in paper.three_seed_update()
    assert 'neither a three-seed intervention policy nor independent confirmation' in paper.three_seed_update()
    assert '-3.535 pp' in paper.seed_dispersion_update()
    assert 'full risk policy is not evaluated' in paper.seed_dispersion_update()
    assert '0/102' in paper.recording_risk_update() and 'undefined in 17' in paper.recording_risk_update()
    assert '101/102 feasible views' in paper.bank_ceiling_update()
    assert 'without demonstrating a learnable selector' in paper.bank_ceiling_update()
    assert '-7.68 pp' in paper.easy_forecast_update() and 'Neither new trajectory bank' in paper.easy_forecast_update()
    assert '-40.03 pp and raw risk passes 0/102' in paper.easy_forecast_update()
    assert 'feasibility to 102/102, but neither raw variant passes any view' in paper.easy_forecast_update()
    assert '-7.71 pp [-12.51, -3.12]' in paper.easy_forecast_update()
    assert '-0.061 pp [-0.157, -0.006]' in paper.easy_forecast_update()
    assert '+0.077 pp versus CV but passes only 2/102' in paper.easy_forecast_update()
    assert 'calibration passes 19/102 by retaining 474 of 6,184 switches' in paper.easy_forecast_update()
