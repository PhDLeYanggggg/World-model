from src.stage42_paper_claim_contract import run_stage42_paper_claim_contract


if __name__ == "__main__":
    result = run_stage42_paper_claim_contract()
    gate = result["stage42_id_gate"]
    print(f"Stage42-ID paper claim contract: {gate['verdict']} ({gate['passed']}/{gate['total']})")
