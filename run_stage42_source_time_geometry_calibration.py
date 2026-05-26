from src.stage42_source_time_geometry_calibration import run_stage42_source_time_geometry_calibration


if __name__ == "__main__":
    result = run_stage42_source_time_geometry_calibration()
    gate = result["stage42_bn_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
