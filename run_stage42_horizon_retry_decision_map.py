from src.stage42_horizon_retry_decision_map import run_stage42_horizon_retry_decision_map


if __name__ == "__main__":
    payload = run_stage42_horizon_retry_decision_map()
    gate = payload["stage42_fy_gate"]
    print(f"Stage42-FY gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
