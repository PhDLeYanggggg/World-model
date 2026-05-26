from src.stage42_post_bi_t100_source_package import run_stage42_post_bi_t100_source_package


if __name__ == "__main__":
    result = run_stage42_post_bi_t100_source_package()
    gate = result["stage42_bj_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
