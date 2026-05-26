from src.stage42_local_t100_easy_guard_repair import run_stage42_local_t100_easy_guard_repair


if __name__ == "__main__":
    result = run_stage42_local_t100_easy_guard_repair()
    gate = result["stage42_bi_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
