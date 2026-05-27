from src.stage42_fh_horizon_row_switch_specialist import run_stage42_fh_horizon_row_switch_specialist


if __name__ == "__main__":
    payload = run_stage42_fh_horizon_row_switch_specialist()
    gate = payload["stage42_fm_gate"]
    print(f"Stage42-FM gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
