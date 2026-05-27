from src.stage42_ucy_h100_terms_prefill_from_integrity import run_stage42_ucy_h100_terms_prefill_from_integrity


if __name__ == "__main__":
    result = run_stage42_ucy_h100_terms_prefill_from_integrity()
    gate = result["stage42_gy_gate"]
    print(f"Stage42-GY gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
