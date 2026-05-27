from src.stage42_h100_weak_horizon_source_support_audit import run_stage42_h100_weak_horizon_source_support_audit


if __name__ == "__main__":
    payload = run_stage42_h100_weak_horizon_source_support_audit()
    gate = payload["stage42_fp_gate"]
    print(f"Stage42-FP gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
