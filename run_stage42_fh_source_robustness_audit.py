from src.stage42_fh_source_robustness_audit import run_stage42_fh_source_robustness_audit


if __name__ == "__main__":
    payload = run_stage42_fh_source_robustness_audit()
    gate = payload["stage42_fj_gate"]
    print(f"Stage42-FJ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
