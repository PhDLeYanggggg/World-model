import numpy as np

from src.evaluation import m3w_prefix_capacity as model


def fixture():
    c = np.zeros((4, 1, 2))
    y = c.copy()
    y[...,0] = 1.
    bank = np.repeat(c[:,None], 2, axis=1)
    bank[:,0,:,0], bank[:,1,:,0] = .9,.2
    direct_choice = np.array([1,-1,1,-1])
    direct = c.copy()
    direct[[0,2]] = bank[[0,2],1]
    valid = np.ones((4,1),bool)
    groups = [np.array([0,1]),np.array([2,3])]
    legal = np.array([[True,True],[False,False],[True,True],[False,False]])
    pred = np.array([[1.,-.1],[0.,0.],[-1.1,-1.1],[0.,0.]])
    return c,bank,direct,direct_choice,y,valid,groups,legal,np.stack((pred,pred))


def test_later_credit_cannot_pay_for_earlier_switch():
    c,b,d,ch,y,valid,g,ok,pred = fixture()
    p = model.prepare(c,b,d,ch,y,valid,g,1.5,pred,ok)
    solve = lambda mode: model.solve(p,c,b,d,y,valid,'r',1.5,2.,mode,2.)
    terminal,prefix = solve('fixed_terminal'),solve('fixed_prefix')
    assert terminal['status'] == prefix['status'] == 'optimal'
    assert terminal['risk']['supported'] and prefix['risk']['supported']
    np.testing.assert_allclose(terminal['paired_lower_mass'], .7)
    np.testing.assert_allclose(prefix['paired_lower_mass'], .35)
    np.testing.assert_allclose(terminal['paired_lower_mass'],terminal['paired_upper_mass'])


def test_fixed_counts_can_block_a_supported_recording():
    c,b,d,ch,y,valid,g,ok,pred = fixture()
    ch[:2] = -1
    d[:2] = c[:2]
    valid[2:] = False
    p = model.prepare(c,b,d,ch,y,valid,g,1.5,pred,ok)
    solve = lambda mode: model.solve(p,c,b,d,y,valid,'r',1.5,2.,mode,2.)
    assert solve('free_capped')['status'] == 'optimal'
    assert solve('fixed') == dict(status='infeasible',reason='no_eligible_easy_reference')


def test_unknown_objective_uses_direct_path_disagreement_not_cv_radius():
    c,b,d,ch,y,valid,g,ok,pred = fixture()
    valid[0] = False
    y[0] = np.nan
    p = model.prepare(c,b,d,ch,y,valid,g,1.5,pred,ok)
    np.testing.assert_allclose(p['gain'][0],[-.25,.1])


def test_dominance_is_same_row_and_respects_every_predicted_cost():
    legal = np.array([[True,True,True,False],[True,True,True,True]])
    gain = np.array([[2.,2.,2.,100.],[0.,1.,1.,1.]])
    cost = np.array([[[2.,2.,0.,-1.],[0.,1.,1.,0.]],
                     [[0.,0.,1.,0.],[1.,0.,0.,0.]]])
    np.testing.assert_array_equal(model.undominated(legal,gain,cost),
                                  [[True,False,True,False],[False,False,False,True]])
