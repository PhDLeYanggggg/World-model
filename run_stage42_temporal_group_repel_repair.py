from src.stage42_temporal_group_repel_repair import run_stage42_temporal_group_repel_repair


if __name__ == "__main__":
    payload = run_stage42_temporal_group_repel_repair()
    gate = payload["stage42_ez_gate"]
    print(f"Stage42-EZ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
    print(f"Decision: {payload['deployment_decision']['decision']}")
