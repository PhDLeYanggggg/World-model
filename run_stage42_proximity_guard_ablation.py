from src.stage42_proximity_guard_ablation import run


if __name__ == "__main__":
    result = run()
    gate = result["stage42_cr_gate"]
    print(f"Stage42-CR proximity guard ablation: {gate['verdict']} ({gate['passed']}/{gate['total']})")
