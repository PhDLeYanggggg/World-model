from src.stage42_local_t100_protected_policy import run_stage42_local_t100_protected_policy


if __name__ == "__main__":
    result = run_stage42_local_t100_protected_policy()
    gate = result["stage42_bg_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
