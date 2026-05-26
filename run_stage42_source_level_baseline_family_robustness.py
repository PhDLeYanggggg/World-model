from src.stage42_source_level_baseline_family_robustness import run_stage42_source_level_baseline_family_robustness


if __name__ == "__main__":
    result = run_stage42_source_level_baseline_family_robustness()
    gate = result["stage42_av_gate"]
    print(f"Stage42-AV complete: {gate['verdict']} ({gate['passed']}/{gate['total']})")
