from src.stage42_dual_domain_group_consistency_statistics import run_stage42_dual_domain_group_consistency_statistics


if __name__ == "__main__":
    payload = run_stage42_dual_domain_group_consistency_statistics()
    gate = payload["stage42_ea_gate"]
    print(
        "Stage42-EA dual-domain group-consistency statistics:",
        f"{gate['passed']}/{gate['total']}",
        gate["verdict"],
    )
