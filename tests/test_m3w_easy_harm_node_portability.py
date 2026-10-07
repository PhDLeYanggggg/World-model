import ast
import copy
import inspect

import pytest

from scripts import diagnose_m3w_easy_harm_node_portability as run
from scripts import manage_m3w_easy_harm_node_portability as manager


def rows():
    return [dict(name=str(i),comparison=dict(all_control_fields_exact=True)) for i in range(17)]


def test_only_complete_exact_grid_allows_alternate_node():
    data=rows();expected=[r['name'] for r in data]
    out=run.summarize(data,expected)
    assert out['alternate_node_admissible'] and out['verification_updates']==34000
    assert out['new_scientific_fits']==0 and not out['references_changed']
    assert not out['tolerance_relaxed'] and not out['independent_roles_read']
    data[5]['comparison']['all_control_fields_exact']=False
    out=run.summarize(data,expected)
    assert not out['alternate_node_admissible'] and out['frozen_reference_exact']==16


@pytest.mark.parametrize('change',['missing','duplicate','foreign'])
def test_no_partial_or_different_reference_acceptance(change):
    data=rows();expected=[r['name'] for r in data]
    if change=='missing':data.pop()
    elif change=='duplicate':data[1]=copy.deepcopy(data[0])
    else:data[1]['name']='foreign'
    with pytest.raises(ValueError):run.summarize(data,expected)


def test_submission_has_no_login_numerics_and_preserves_queued_job():
    ast.parse(manager.SUBMIT)
    assert 'import torch' not in manager.SUBMIT and 'import numpy' not in manager.SUBMIT
    assert 'scancel' not in manager.SUBMIT
    assert 'unknown_timeout_inspect_do_not_resubmit' in manager.SUBMIT
    assert 'existing_training_job_untouched' in manager.SUBMIT
    assert '#SBATCH --cpus-per-task=4' in manager.SUBMIT and '#SBATCH --mem=8G' in manager.SUBMIT
    source=inspect.getsource(run.main)
    assert "arm='quadratic'" in source and "arm='easy_deviance'" not in source
    assert 'checkpoint_write.lock' in source and 'frozen_preprocess' in source
