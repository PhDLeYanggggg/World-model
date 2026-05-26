from src.stage42_calibrated_subset_eval import run_stage42_calibrated_subset_eval


if __name__ == "__main__":
    result = run_stage42_calibrated_subset_eval()
    gate = result["stage42_bo_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
