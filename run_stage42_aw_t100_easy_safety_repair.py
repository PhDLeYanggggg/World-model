from src.stage42_aw_t100_easy_safety_repair import run_stage42_aw_t100_easy_safety_repair


if __name__ == "__main__":
    result = run_stage42_aw_t100_easy_safety_repair()
    gate = result["stage42_ay_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
