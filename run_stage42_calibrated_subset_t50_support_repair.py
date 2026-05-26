from src.stage42_calibrated_subset_t50_support_repair import run_stage42_calibrated_subset_t50_support_repair


if __name__ == "__main__":
    result = run_stage42_calibrated_subset_t50_support_repair()
    gate = result["stage42_bq_gate"]
    print(f"Stage42-BQ calibrated subset t50 support repair: {gate['passed']} / {gate['total']} {gate['verdict']}")
