from scripts import audit_m3w_av2_release_transport as audit


def row(clip, score, *, supported=True, risk=True, gain=True):
    cause = "supported" if risk else "observed_positive_harm" if supported else "unsupported_reference"
    return dict(clip=clip, scores={"arm": score}, support=supported, cause=cause,
                result=dict(eligible=risk and gain, risk=risk, gain=(1., 1. if gain else -1.)))


def test_hindsight_prefix_cannot_split_a_score_tie():
    highest = row("top", 4.)
    tie = [row("pass", 3.), row("fail", 3., risk=False)]
    low = row("low", 2.)
    assert audit.prefix([low, *tie, highest], "arm", audit.eligible) == ([highest], tie)
    matched = audit.count_match([low, *tie, highest], "arm", 2)
    assert matched["n"] == 2
    assert matched["qualified"] == 1.5
    assert matched["risk_fail"] == .5


def test_threshold_count_and_ranking_failures_are_separate():
    cal = [row("cal-ok", 4.), row("cal-bad", 3., risk=False)]
    held = [row("missing", 6., supported=False, risk=False), row("accuracy-fail", 5., gain=False),
            row("missed", 3.), row("blocked", None)]
    res = audit.comparison(cal, held, "arm", audit.run.calibrate(cal, "arm"))
    assert res["actual"]["n"] == 2
    assert res["matched"]["n"] == 1
    assert res["matched"]["qualified"] == 0
    assert res["matched_oracle"] == 1
    assert res["retrospective_known_support_strict"] == 0
    assert res["retrospective_known_support_risk"] == 2
    assert res["misses"] == dict(gate_excluded=1, below_safe_prefix=1, inside_safe_prefix=0)


def test_no_release_is_not_a_pass():
    cal = [row("bad", 2., risk=False)]
    held = [row("good", 1.)]
    res = audit.comparison(cal, held, "arm", audit.run.calibrate(cal, "arm"))
    assert res["actual"]["n"] == res["matched"]["n"] == res["matched_oracle"] == 0
    assert res["retrospective_strict"] == res["misses"]["inside_safe_prefix"] == 1
