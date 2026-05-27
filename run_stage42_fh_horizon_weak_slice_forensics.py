from src.stage42_fh_horizon_weak_slice_forensics import run_stage42_fh_horizon_weak_slice_forensics


if __name__ == "__main__":
    payload = run_stage42_fh_horizon_weak_slice_forensics()
    gate = payload["stage42_fl_gate"]
    print(f"Stage42-FL gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
