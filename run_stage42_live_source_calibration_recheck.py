from src.stage42_live_source_calibration_recheck import run_stage42_live_source_calibration_recheck


if __name__ == "__main__":
    payload = run_stage42_live_source_calibration_recheck()
    gate = payload["stage42_ga_gate"]
    print(f"Stage42-GA live source/calibration recheck: {gate['verdict']} ({gate['passed']}/{gate['total']})")
