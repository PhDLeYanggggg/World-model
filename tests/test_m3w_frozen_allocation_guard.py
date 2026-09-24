from copy import deepcopy

import pytest

from src.evaluation.m3w_frozen_allocation_guard import require_frozen_repair_identity


@pytest.fixture
def identities():
    parent={'source_bindings':{'parent.py':'frozen-a','model.pt':'frozen-b'},'config':{'seeds':[17,29,43]}}
    repair={'source_bindings':dict(parent['source_bindings'], **{'repair.py':'frozen-c'}),
            'config':{'rho':.02},'numpy':'frozen-version'}
    return deepcopy(repair), parent, repair


def test_matching_frozen_chain_passes_without_mutation(identities):
    before=deepcopy(identities)
    require_frozen_repair_identity(*identities)
    assert before==identities


@pytest.mark.parametrize('change',['changed','missing'])
def test_changed_parent_source_binding_rejected(identities,change):
    current,parent,repair=identities
    if change=='changed':current['source_bindings']['parent.py']='new-hash'
    else:del current['source_bindings']['parent.py']
    with pytest.raises(ValueError,match='Frozen parent source mismatch'):
        require_frozen_repair_identity(current,parent,repair)


@pytest.mark.parametrize('change',['extra_source','repair_source','config','runtime'])
def test_complete_repair_identity_must_match(identities,change):
    current,parent,repair=identities
    if change=='extra_source':current['source_bindings']['extra.py']='new'
    elif change=='repair_source':current['source_bindings']['repair.py']='new'
    elif change=='config':current['config']['rho']=.03
    else:current['numpy']='new-version'
    with pytest.raises(ValueError,match='Reconstructed repair identity'):
        require_frozen_repair_identity(current,parent,repair)
