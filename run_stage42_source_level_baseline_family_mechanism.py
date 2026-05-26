from src.stage42_source_level_baseline_family_mechanism import run_stage42_source_level_baseline_family_mechanism


if __name__ == "__main__":
    result = run_stage42_source_level_baseline_family_mechanism()
    gate = result["stage42_au_gate"]
    print(f"Stage42-AU complete: {gate['verdict']} ({gate['passed']}/{gate['total']})")
