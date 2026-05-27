from src.stage42_fh_horizon_conservative_easy_guard import run_stage42_fh_horizon_conservative_easy_guard


if __name__ == "__main__":
    payload = run_stage42_fh_horizon_conservative_easy_guard()
    gate = payload["stage42_fn_gate"]
    print(f"Stage42-FN gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
