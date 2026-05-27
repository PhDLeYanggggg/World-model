from src.stage42_paper_contract_compliance import run_stage42_paper_contract_compliance


if __name__ == "__main__":
    result = run_stage42_paper_contract_compliance()
    gate = result["stage42_ie_gate"]
    print(f"Stage42-IE paper contract compliance: {gate['verdict']} ({gate['passed']}/{gate['total']})")
