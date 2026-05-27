from src.stage42_calibration_hint_intake_bridge import run_stage42_calibration_hint_intake_bridge


if __name__ == "__main__":
    payload = run_stage42_calibration_hint_intake_bridge()
    gate = payload["stage42_gd_gate"]
    print(f"Stage42-GD calibration hint intake bridge: {gate['verdict']} ({gate['passed']}/{gate['total']})")
