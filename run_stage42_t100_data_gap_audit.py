from src.stage42_t100_data_gap_audit import run_stage42_t100_data_gap_audit


if __name__ == "__main__":
    result = run_stage42_t100_data_gap_audit()
    gate = result["stage42_bb_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
