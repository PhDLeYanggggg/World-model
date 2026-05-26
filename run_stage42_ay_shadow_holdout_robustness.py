from src.stage42_ay_shadow_holdout_robustness import run_stage42_ay_shadow_holdout_robustness


if __name__ == "__main__":
    result = run_stage42_ay_shadow_holdout_robustness()
    gate = result["stage42_az_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
