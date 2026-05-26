from src.stage42_post_repair_claim_refresh import build_post_repair_claim_refresh


if __name__ == "__main__":
    result = build_post_repair_claim_refresh()
    gate = result["stage42_ah_gate"]
    print(f"Stage42-AH post-repair claim refresh: {gate['passed']} / {gate['total']} {gate['verdict']}")
