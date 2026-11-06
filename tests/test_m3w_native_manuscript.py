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
