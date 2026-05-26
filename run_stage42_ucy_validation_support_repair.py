from src.stage42_ucy_validation_support_repair import run_stage42_ucy_validation_support_repair


if __name__ == "__main__":
    result = run_stage42_ucy_validation_support_repair()
    gate = result["stage42_aw_gate"]
    print(f"Stage42-AW complete: {gate['verdict']} ({gate['passed']}/{gate['total']})")
