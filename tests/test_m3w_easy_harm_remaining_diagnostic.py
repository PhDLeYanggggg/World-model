import copy
import pytest

from scripts.diagnose_m3w_easy_harm_remaining_controls import remaining_refs, summary


def fixture():
    heads = [dict(group='g'+str(i), identity=dict(seed=17)) for i in range(3)]
    names = lambda i: ['g'+str(i)+'_head17_'+a for a in ('quadratic','easy_deviance')]
    observation = dict(missing=names(1)+names(2), fits=[dict(path='fits/'+v+'.json') for v in names(0)])
    return dict(heads=heads), observation


def test_only_unaccepted_pairs_in_registered_order():
    manifest, observation = fixture()
    assert remaining_refs(manifest, observation) == manifest['heads'][1:]


@pytest.mark.parametrize('mutation', ['duplicate','foreign','partial','overlap','missing_accepted'])
def test_refuses_incomplete_or_ambiguous_inventory(mutation):
    manifest, row = fixture()
    if mutation == 'duplicate': row['missing'].append(row['missing'][0])
    if mutation == 'foreign': row['missing'][0] = 'outside'
    if mutation == 'partial': row['missing'].pop()
    if mutation == 'overlap': row['fits'].append(dict(path='fits/'+row['missing'][0]+'.json'))
    if mutation == 'missing_accepted': row['fits'].pop()
    with pytest.raises(ValueError): remaining_refs(manifest,row)


def test_diagnostic_equality_does_not_grant_acceptance():
    keys = ('initial_model','preprocess','settings','input_hashes','seed','step',
            'sampler_rng','torch_rng','draw_hash','row_draws','queries')
    cmp = {k:dict(exact=True) for k in keys}
    cmp['all_control_fields_exact'] = True
    rows = [dict(name='g', comparisons={k:copy.deepcopy(cmp) for k in (
        'old_direct_vs_new_direct','old_direct_vs_historical','new_direct_vs_historical')})]
    rows[0]['comparisons']['old_direct_vs_historical']['all_control_fields_exact'] = False
    out = summary(rows)
    assert out['original_new_exact']==1 and out['original_historical_exact']==0
    assert not out['acceptance_changed'] and not out['scientific_lift_established']
    assert not out['metadata_failures']
    rows[0]['comparisons']['new_direct_vs_historical']['sampler_rng']['exact'] = False
    assert summary(rows)['metadata_failures']==['g']
