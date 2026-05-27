from src.stage42_fh_horizon_weak_slice_repair import run_stage42_fh_horizon_weak_slice_repair


if __name__ == "__main__":
    payload = run_stage42_fh_horizon_weak_slice_repair()
    gate = payload["stage42_fk_gate"]
    print(f"Stage42-FK gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
