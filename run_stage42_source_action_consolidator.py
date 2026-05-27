from src.stage42_source_action_consolidator import run_stage42_source_action_consolidator


if __name__ == "__main__":
    payload = run_stage42_source_action_consolidator()
    gate = payload["stage42_fw_gate"]
    print(f"Stage42-FW gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
