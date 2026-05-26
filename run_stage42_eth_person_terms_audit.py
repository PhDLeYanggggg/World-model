from src.stage42_eth_person_terms_audit import run_stage42_eth_person_terms_audit


if __name__ == "__main__":
    result = run_stage42_eth_person_terms_audit()
    gate = result["stage42_bm_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
