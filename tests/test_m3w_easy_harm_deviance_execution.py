import ast
import json
from pathlib import Path
import pytest

from scripts import train_m3w_easy_harm_deviance as run
from scripts import manage_m3w_easy_harm_deviance as manager


def test_remote_templates_parse_and_do_not_use_numerical_login_work():
    for text in (manager.INSTALL,manager.SUBMIT,manager.OBSERVE):
        ast.parse(text)
        assert 'import torch' not in text and 'import numpy' not in text
    assert '--hold' in manager.SUBMIT and 'submission_intent.json' in manager.SUBMIT


def test_new_config_preserves_original_settings_and_budget():
    cfg=json.loads(manager.CONFIG.read_text());old=json.loads(manager.base.CONFIG.read_text())
    assert cfg['head_training']==old['head_training'] and cfg['risk_budget']==old['risk_budget']==.02
    assert cfg['effective_auxiliary_weight']==0 and cfg['neural_fits']==144
    assert not cfg['threshold_search'] and not cfg['independent_roles_read']


def test_four_disjoint_complete_matched_shards():
    manifest={'heads':[dict(group='g'+str(g),identity=dict(seed=s)) for g in range(24) for s in (17,29,43)]}
    parts=[{run.key(r,a) for r in manifest['heads'][i::4] for a in ('quadratic','easy_deviance')} for i in range(4)]
    assert all(len(p)==36 for p in parts)
    assert len(set.union(*parts))==144
    assert all(not parts[i]&parts[j] for i in range(4) for j in range(i))


def test_immutable_receipt_preserved(tmp_path):
    p=tmp_path/'receipt.json';run.once(p,{'x':1});run.once(p,{'x':1})
    with pytest.raises(ValueError):run.once(p,{'x':2})
    assert json.loads(p.read_text())=={'x':1}


def test_unowned_root_rejected_before_artifact_reads(tmp_path):
    with pytest.raises(ValueError,match='Owned'):run.verify(tmp_path)
