from src.stage42_group_consistency_full_waypoint_repair import run_stage42_group_consistency_full_waypoint_repair


if __name__ == "__main__":
    payload = run_stage42_group_consistency_full_waypoint_repair()
    gate = payload["stage42_di_gate"]
    print(f"Stage42-DI gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
    print(f"Decision: {payload['deployment_decision']['decision']}")
