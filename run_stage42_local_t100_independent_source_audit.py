from src.stage42_local_t100_independent_source_audit import run_stage42_local_t100_independent_source_audit


if __name__ == "__main__":
    result = run_stage42_local_t100_independent_source_audit()
    gate = result["stage42_bh_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
