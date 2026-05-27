from src.stage42_current_claim_evidence_closure import run_stage42_current_claim_evidence_closure


if __name__ == "__main__":
    result = run_stage42_current_claim_evidence_closure()
    gate = result["stage42_ic_gate"]
    print(f"Stage42-IC current claim/evidence closure: {gate['verdict']} ({gate['passed']}/{gate['total']})")
