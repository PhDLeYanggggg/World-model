from src.stage42_source_terms_prefill import run_stage42_source_terms_prefill


if __name__ == "__main__":
    payload = run_stage42_source_terms_prefill()
    gate = payload["stage42_gb_gate"]
    print(f"Stage42-GB source terms prefill: {gate['verdict']} ({gate['passed']}/{gate['total']})")
