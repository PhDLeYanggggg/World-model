from src.stage42_group_level_risk_repair import run_stage42_group_level_risk_repair


if __name__ == "__main__":
    payload = run_stage42_group_level_risk_repair()
    gate = payload["stage42_ex_gate"]
    print(f"Stage42-EX group-level risk repair: {gate['passed']}/{gate['total']} {gate['verdict']}")
