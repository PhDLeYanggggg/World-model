import json
from pathlib import Path

import pytest

from reproducibility.m3w_athens_replay import verify

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'


def raw_inputs():
    return {**{k: (PUBLIC/'pneuma_confirmation_v1'/f'{k}.json').read_bytes() for k in verify.ATHENS},
            'comparison': (PUBLIC/'external_budget_comparison_v1/readout.json').read_bytes()}


def mutate(raw, key, fn):
    obj = json.loads(raw[key])
    fn(obj)
    raw[key] = json.dumps(obj).encode()


def test_athens_replay_distinguishes_baseline_gain_and_joint_gain():
    r = verify.replay(raw_inputs())
    assert r['athens']['joint_vs_cv']['positive_dates'] == 4
    assert r['athens']['same_family_vs_cv']['positive_dates'] == 4
    assert r['athens']['joint_vs_independent']['unresolved_dates'] == 4
    assert r['athens']['equal_steps']['unresolved_dates'] == 4
    assert not r['joint_advantage_established']
    assert not r['athens_release_evaluated']
    assert r['comparison']['athens']['ci95_pp'][1] < 0


def test_reject_calibration_or_deployment_relabeling():
    raw = raw_inputs()
    mutate(raw, 'confirmation', lambda d: d.update(release='passed'))
    with pytest.raises(ValueError, match='not a release test'):
        verify.replay(raw)


def test_reject_test_threshold_changes():
    raw = raw_inputs()
    mutate(raw, 'stepwise_budget_sensitivity', lambda d: d.update(thresholds_changed=True))
    with pytest.raises(ValueError, match='keep the primary result frozen'):
        verify.replay(raw)


def test_reject_changed_pairing():
    raw = raw_inputs()
    mutate(raw, 'comparison', lambda d: d['cohorts']['wep']['equal_steps']['eligible_indices'].reverse())
    with pytest.raises(ValueError, match='source roster'):
        verify.replay(raw)


def test_reject_changed_interval():
    raw = raw_inputs()
    mutate(raw, 'comparison', lambda d: d['cohorts']['athens'].update(paired_source_ci95_pp=[0., 1.]))
    with pytest.raises(ValueError, match='interval did not reproduce'):
        verify.replay(raw)


def test_manuscript_preserves_title_abstract_and_disclosure():
    pytest.importorskip('pypdf')
    from scripts import build_m3w_athens_candidate as c
    data, _ = c.inputs()
    old = (c.PARENT/'manuscript.tex').read_text()
    new, markers = c.revise(old, data)
    assert old.split('\\section{Introduction}', 1)[0] == new.split('\\section{Introduction}', 1)[0]
    disclosure = lambda s: s.split('\\section*{LLM Usage Statement}', 1)[1].split('\\end{samepage}', 1)[0]
    assert disclosure(old) == disclosure(new)
    assert 'Athens adds a positive' in new
    assert all(x in new for x in markers)
    assert new.count('\\bibitem{pneuma}') == 1
    with pytest.raises(ValueError, match='manuscript anchors'):
        c.revise(new, data)
