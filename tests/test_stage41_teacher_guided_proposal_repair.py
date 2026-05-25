from src import stage41_teacher_guided_proposal_repair as repair


def test_score_penalizes_collision_delta():
    metrics = {"all_improvement": 1.0, "t50_improvement": 1.0, "t100_improvement": 1.0, "hard_failure_improvement": 1.0, "easy_degradation": 0.0}
    safe = repair._score(metrics, collision_delta=0.0)
    unsafe = repair._score(metrics, collision_delta=0.2)
    assert safe > unsafe


def test_jsonable_handles_scalars():
    assert repair._jsonable({"a": [1, 2]}) == {"a": [1, 2]}
