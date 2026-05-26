from src.stage42_eth_t50_fde_source_repair import build_eth_t50_fde_source_repair


if __name__ == "__main__":
    result = build_eth_t50_fde_source_repair()
    gate = result["stage42_ag_gate"]
    print(f"Stage42-AG ETH_UCY t50/FDE repair: {gate['passed']} / {gate['total']} {gate['verdict']}")
