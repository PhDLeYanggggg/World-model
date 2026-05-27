from src.stage42_fh_horizon_gain_harm_specialist import run_stage42_fh_horizon_gain_harm_specialist


if __name__ == "__main__":
    payload = run_stage42_fh_horizon_gain_harm_specialist()
    gate = payload["stage42_fo_gate"]
    print(f"Stage42-FO gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
