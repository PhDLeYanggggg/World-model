from scripts.report_m3w_cross_critic_veto import benefit_loss


def test_algebra_uses_full_strategy_not_differently_normalized_primary():
    row = dict(secondary_full_strategy_paired=dict(known_pp=-2.), matched_known_harm_reduction_pp=.5,
        primary_nonempty_matched=dict(contrast=dict(known_pp=-100.)))
    assert benefit_loss(row) == 2.5


def test_unavailable_support_does_not_become_zero_benefit():
    for utility, harm in ((None, .5), (-2., None)):
        assert benefit_loss(dict(secondary_full_strategy_paired=dict(known_pp=utility),
            matched_known_harm_reduction_pp=harm)) is None
