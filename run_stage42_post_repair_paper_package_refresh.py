from src.stage42_post_repair_paper_package_refresh import build_post_repair_paper_package_refresh


if __name__ == "__main__":
    result = build_post_repair_paper_package_refresh()
    gate = result["stage42_aj_gate"]
    print(f"Stage42-AJ post-repair paper package refresh: {gate['passed']} / {gate['total']} {gate['verdict']}")
