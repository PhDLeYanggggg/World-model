from src.stage42_group_consistency_policy_freeze import run_stage42_freeze_group_consistency_policy


if __name__ == "__main__":
    payload = run_stage42_freeze_group_consistency_policy()
    gate = payload["stage42_dj_gate"]
    print(f"Stage42-DJ frozen group-consistency policy: {gate['verdict']} ({gate['passed']}/{gate['total']})")
