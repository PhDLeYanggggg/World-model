from src.stage42_fe_policy_freeze_replay import run_stage42_fe_policy_freeze_replay


if __name__ == "__main__":
    result = run_stage42_fe_policy_freeze_replay()
    gate = result["stage42_ff_gate"]
    print(f"Stage42-FF gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
