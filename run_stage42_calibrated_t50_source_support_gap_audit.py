from src.stage42_calibrated_t50_source_support_gap_audit import run_stage42_calibrated_t50_source_support_gap_audit


if __name__ == "__main__":
    result = run_stage42_calibrated_t50_source_support_gap_audit()
    gate = result["stage42_br_gate"]
    print(f"Stage42-BR calibrated t50 source-support gap audit: {gate['passed']} / {gate['total']} {gate['verdict']}")
