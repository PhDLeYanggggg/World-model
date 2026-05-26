from src.stage42_calibrated_subset_safety_repair import run_stage42_calibrated_subset_safety_repair


if __name__ == "__main__":
    result = run_stage42_calibrated_subset_safety_repair()
    gate = result["stage42_bp_gate"]
    print(f"Stage42-BP calibrated subset safety repair: {gate['passed']} / {gate['total']} {gate['verdict']}")
