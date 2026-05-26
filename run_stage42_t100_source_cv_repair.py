from src.stage42_t100_source_cv_repair import run_stage42_t100_source_cv_repair


if __name__ == "__main__":
    result = run_stage42_t100_source_cv_repair()
    gate = result["stage42_ba_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
