from src.stage42_continuous_group_risk_repair import run_stage42_continuous_group_risk_repair


if __name__ == "__main__":
    payload = run_stage42_continuous_group_risk_repair()
    gate = payload["stage42_ey_gate"]
    print(f"Stage42-EY continuous group-risk repair: {gate['passed']}/{gate['total']} {gate['verdict']}")
